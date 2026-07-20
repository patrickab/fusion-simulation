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

and saved under ``<stage1_run_dir>/stage2/`` with its scale in ``run.json``.
"""

import argparse
import json
from pathlib import Path

from src.engine.network import FoundationModel
from src.engine.network_manager import NetworkManager
from src.lib.config import KPI_EVAL_CONFIGS
from src.lib.logger import get_logger
from src.lib.network_config import HyperParams
from src.lib.run_artifacts import load_config, load_run
from src.streamlit.network_utils import resolve_run_directory

logger = get_logger(name="ResidualCorrection")


def load_foundation(
    stage1: str | Path, *, soft_bc: bool | None = None, allow_nested: bool = False
) -> tuple[FoundationModel, HyperParams, Path]:
    """Load a plain checkpoint as a frozen foundation."""
    stage1_dir = stage1 if isinstance(stage1, Path) else resolve_run_directory(stage1)
    if not allow_nested and (stage1_dir / "stage2" / "network.flax").exists():
        raise ValueError("A corrector checkpoint cannot be used as a stage-1 foundation")
    hp = HyperParams.from_dict(load_config(stage1_dir))
    if soft_bc is not None and hp.soft_bc != soft_bc:
        raise ValueError(
            f"Foundation soft_bc={hp.soft_bc} does not match corrector soft_bc={soft_bc}"
        )
    manager = NetworkManager(hp)
    params = manager.from_disk(stage1_dir / "network.flax")
    return FoundationModel(model=manager.model, params=params), hp, stage1_dir


def _load_composed(
    stage1: str | Path, stage2_dir: Path, n_validation_size: int = KPI_EVAL_CONFIGS
) -> NetworkManager:
    """Load a frozen foundation plus a stage-2 directory."""
    if not (stage2_dir / "network.flax").exists():
        raise FileNotFoundError(f"no stage2 correction net found under {stage2_dir}")

    scale = float(load_run(stage2_dir).get("result", {}).get("stage2_scale", 1.0))

    prior, hp1, stage1_dir = load_foundation(stage1, allow_nested=True)
    hp2 = HyperParams.from_dict(load_config(stage2_dir))
    if hp1.soft_bc != hp2.soft_bc:
        raise ValueError(
            f"Foundation soft_bc={hp1.soft_bc} does not match corrector soft_bc={hp2.soft_bc}"
        )
    # Build the composed manager and load stage-2 params from disk.
    mgr = NetworkManager(
        hp2,
        prior=prior,
        scale=scale,
        n_validation_size=n_validation_size,
    )
    mgr.state = mgr.state.replace(params=mgr.from_disk(stage2_dir / "network.flax"))
    mgr.artifact_stem = f"{stage1_dir.name}/stage2"
    return mgr


def load_checkpoint(
    name: str | Path, *, n_validation_size: int = KPI_EVAL_CONFIGS
) -> NetworkManager:
    """Load a plain, nested-corrector, or retained HPO-corrector checkpoint."""
    run_dir = name if isinstance(name, Path) else resolve_run_directory(name)
    if (run_dir / "stage2" / "network.flax").exists():
        return _load_composed(name, run_dir / "stage2", n_validation_size)

    foundation_path = run_dir.parent / "foundation.json"
    stage2_run = load_run(run_dir)
    has_stage2_metadata = "stage2_scale" in stage2_run.get("result", {})
    if foundation_path.exists() or has_stage2_metadata:
        if not foundation_path.exists() or not has_stage2_metadata:
            raise ValueError(f"incomplete corrector metadata for {run_dir}")
        stage1_name = json.loads(foundation_path.read_text()).get("stage1_run")
        if not stage1_name:
            raise ValueError(f"invalid foundation metadata in {foundation_path}")
        local_foundation = run_dir.parent / "_foundation"
        local_artifacts = [
            (local_foundation / "run.json").exists(),
            (local_foundation / "network.flax").exists(),
        ]
        if any(local_artifacts) and not all(local_artifacts):
            raise ValueError(f"incomplete bundled foundation in {local_foundation}")
        manager = _load_composed(
            local_foundation if all(local_artifacts) else stage1_name,
            run_dir,
            n_validation_size,
        )
        manager.artifact_stem = str(name)
        return manager

    hp = HyperParams.from_dict(load_config(run_dir))
    manager = NetworkManager(hp, n_validation_size=n_validation_size)
    manager.state = manager.state.replace(params=manager.from_disk(run_dir / "network.flax"))
    manager.artifact_stem = str(name)
    return manager


if __name__ == "__main__":
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
    prior, hp1, stage1_dir = load_foundation(args.stage1)
    logger.info(f"stage-1 dir: {stage1_dir}")
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
    stage2 = NetworkManager(hp2, prior=prior, scale=args.stage2_scale, stage1_run_dir=stage1_dir)
    stage2.train()
    logger.info(f"stage-2 benchmark saved to {stage1_dir / 'stage2'}")
