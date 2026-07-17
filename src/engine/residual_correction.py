"""Multistage residual-correction training CLI.

Freezes a converged stage-1 PINN and trains a second, smaller network whose
output is added to it: psi_final = psi_stage1 + scale * psi_stage2. Wang 2024/2025
("Spectrum-Informed Multistage Neural Networks", arXiv 2507.16636, 2407.17213)
report this drives the GS residual down by orders of magnitude versus
single-stage training on this exact PDE.

Stage-2 is trained with the ordinary GS physics loss evaluated on the *composed*
field. Since stage-1 is wrapped in a FoundationModel (a frozen constant not part
of the traced params pytree), gradients only flow into stage-2.

The composed NetworkManager is built as::

    prior = FoundationModel(stage1.model, stage1.state.params)
    stage2 = NetworkManager(stage2_cfg, prior=prior, scale=args.stage2_scale)
    stage2.train()

and saved under ``<stage1_run_dir>/stage2/`` with ``stage2_meta.json``.
"""

import argparse
import json

from src.engine.network import (
    BASE_SEED,
    FoundationModel,
    NetworkManager,
)
from src.lib.logger import get_logger
from src.lib.network_config import HyperParams
from src.streamlit.network_utils import resolve_run_directory

logger = get_logger(name="ResidualCorrection")


def load_combined(stage1_name: str) -> NetworkManager:
    """Load a saved stage-1 checkpoint plus its stage-2 correction net.

    Returns a NetworkManager whose ``make_psi_fn()`` returns the composed field
    ``psi_stage1(frozen) + scale * psi_stage2(params)``.

    ``stage1_name`` is a ``commit/run`` slug accepted by
    ``resolve_run_directory``; the stage-2 net is expected at
    ``<run_dir>/stage2/`` (see ``NetworkManager.to_disk(stage1_run_dir=...)``).
    Reads ``stage2_meta.json`` if present to recover the output scale
    (defaults to 1.0 for checkpoints saved before this field existed).
    """
    stage1_dir = resolve_run_directory(stage1_name)
    stage2_dir = stage1_dir / "stage2"
    if not (stage2_dir / "network.flax").exists():
        raise FileNotFoundError(f"no stage2 correction net found under {stage2_dir}")

    meta_path = stage2_dir / "stage2_meta.json"
    scale: float = (
        json.loads(meta_path.read_text()).get("scale", 1.0) if meta_path.exists() else 1.0
    )

    hp1 = HyperParams.from_json(str(stage1_dir / "config.json"))
    stage1 = NetworkManager(hp1)
    stage1.state = stage1.state.replace(params=stage1.from_disk(stage1_dir / "network.flax"))

    hp2 = HyperParams.from_json(str(stage2_dir / "config.json"))
    prior = FoundationModel(model=stage1.model, params=stage1.state.params)
    # Build the composed manager and load stage-2 params from disk.
    mgr = NetworkManager(hp2, prior=prior, scale=scale, seed=BASE_SEED)
    mgr.state = mgr.state.replace(params=mgr.from_disk(stage2_dir / "network.flax"))
    mgr.artifact_stem = f"{stage1_name.split('/')[-1]}/stage2"
    return mgr


if __name__ == "__main__":
    from src.engine.model_evaluation import (
        build_kpi_record,
        evaluate_plasma_kpis,
        kpi_benchmark_configs,
    )
    from src.lib.config import KPI_EVAL_CONFIGS, KPI_POINTS_PER_CONFIG

    parser = argparse.ArgumentParser(
        description="Train a stage-2 residual-correction PINN on top of a frozen stage-1 "
        "checkpoint and score the composed field with the standard KPI protocol."
    )
    parser.add_argument(
        "stage1",
        help="Stage-1 run slug accepted by resolve_run_directory "
        "(e.g. '2026_07_17_14_54_43_arch-pirate2_f3dd0c8').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Total stage-2 training epochs; split 1:5 warmup:decay (default 600).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="128,128,128,128",
        help="Comma-separated hidden layer widths for the stage-2 net (default '128,128,128,128').",
    )
    parser.add_argument(
        "--arch",
        choices=["mlp", "piratenet"],
        default="mlp",
        help="Stage-2 architecture: 'mlp' (default) or 'piratenet'.",
    )
    parser.add_argument(
        "--rwf",
        action="store_true",
        help="Random Weight Factorization on the stage-2 net (Wang et al. arXiv 2210.01274).",
    )
    parser.add_argument(
        "--fourier-features",
        type=int,
        default=64,
        help="Random Fourier features on (r,z) for the stage-2 net; 0 = off (default 64).",
    )
    parser.add_argument(
        "--fourier-sigma",
        type=float,
        default=2.0,
        help="Bandwidth of the random Fourier feature projection (default 2.0).",
    )
    parser.add_argument(
        "--stage2-scale",
        type=float,
        default=0.01,
        help="Output scale ε for the stage-2 contribution: psi = psi1 + ε·psi2. "
        "Keeps the freshly-initialised corrector from perturbing the converged "
        "stage-1 field during warmup (default 0.01).",
    )
    args = parser.parse_args()

    # --- resolve stage-1 ---
    stage1_dir = resolve_run_directory(args.stage1)
    logger.info(f"stage-1 dir: {stage1_dir}")

    hp1 = HyperParams.from_json(str(stage1_dir / "config.json"))
    stage1_mgr = NetworkManager(hp1)
    stage1_mgr.state = stage1_mgr.state.replace(
        params=stage1_mgr.from_disk(stage1_dir / "network.flax")
    )
    logger.info(
        f"stage-1 loaded: arch={hp1.arch} hidden_dims={hp1.hidden_dims} soft_bc={hp1.soft_bc}"
    )

    # --- build stage-2 HyperParams (1:5 warmup:decay, same split as network.py) ---
    warmup = max(1, args.epochs // 6)
    decay = args.epochs - warmup
    # The envelope convention must match stage-1 so psi=0 at the boundary still holds.
    hp2 = HyperParams(
        hidden_dims=tuple(int(d) for d in args.hidden_dims.split(",")),
        arch=args.arch,
        rwf=args.rwf,
        n_fourier_features=args.fourier_features,
        fourier_sigma=args.fourier_sigma,
        warmup_epochs=warmup,
        decay_epochs=decay,
        soft_bc=hp1.soft_bc,
    )
    logger.info(
        f"stage-2 config: arch={hp2.arch} hidden_dims={hp2.hidden_dims} "
        f"nff={hp2.n_fourier_features} epochs={args.epochs} "
        f"(warmup={warmup} decay={decay})"
    )

    # --- build composed manager and train ---
    prior = FoundationModel(model=stage1_mgr.model, params=stage1_mgr.state.params)
    stage2 = NetworkManager(hp2, prior=prior, scale=args.stage2_scale)
    stage2.train(save_to_disk=False)

    # --- save under <stage1_run_dir>/stage2/ ---
    import flax.serialization

    stage2_dir = stage1_dir / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    (stage2_dir / "network.flax").write_bytes(flax.serialization.to_bytes(stage2.state.params))
    hp2.to_json(path=str(stage2_dir / "config.json"))
    (stage2_dir / "stage2_meta.json").write_text(json.dumps({"scale": args.stage2_scale}) + "\n")
    logger.info(f"stage-2 saved to {stage2_dir}")

    # --- KPI eval on the composed field ---
    configs = kpi_benchmark_configs(stage2, KPI_EVAL_CONFIGS)
    kpis = evaluate_plasma_kpis(stage2, configs, sample_size=KPI_POINTS_PER_CONFIG)

    logger.info(
        f"composed KPIs — median={kpis['loss_median']:.4e} "
        f"p95={kpis['loss_p95']:.4e} "
        f"core_median={kpis['core_loss_median']:.4e} "
        f"bnd_leak={kpis['boundary_leak_max']:.4e}"
    )

    stage2.artifact_stem = f"{args.stage1}/stage2"
    record = build_kpi_record(
        stage2,
        kpis,
        n_configs=KPI_EVAL_CONFIGS,
        n_points=KPI_POINTS_PER_CONFIG,
        core_rho=0.85,
        network_name=f"{args.stage1}/stage2",
    )
    kpis_path = stage2_dir / "kpis.json"
    kpis_path.write_text(json.dumps(record, indent=2) + "\n")
    logger.info(f"kpis.json written to {kpis_path}")
