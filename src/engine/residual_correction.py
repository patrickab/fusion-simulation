"""Multistage residual-correction training.

Freezes a converged stage-1 PINN and trains a second, smaller network whose
output is added to it: psi_final = psi_stage1 + psi_stage2. Wang 2024/2025
("Spectrum-Informed Multistage Neural Networks", arXiv 2507.16636, 2407.17213)
report this drives the GS residual down by orders of magnitude versus
single-stage training on this exact PDE.

There is no ground-truth psi to regress stage2 against directly, so instead
stage2 is trained with the ordinary GS physics loss evaluated on the *composed*
field. Since stage1 is a frozen constant (not part of the traced params
pytree), gradients only flow into stage2 — minimizing the composed-field
residual this way is exactly stage2 learning to correct whatever residual
stage1's frozen field left behind.

Composes NetworkManager's public API (Sampler, model init, optimizer,
from_disk) rather than modifying it, per this branch's other in-flight work
on network.py.
"""

import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Callable

import flax.serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from src.engine.network import (
    BASE_SEED,
    LOG_FREQUENCY,
    RESAMPLING_FREQUENCY,
    NetworkManager,
    apply_psi_fn,
)
from src.engine.physics import pinn_loss_function
from src.lib.geometry_config import PlasmaConfig
from src.lib.logger import get_logger
from src.lib.network_config import FluxInput, HyperParams
from src.streamlit.network_utils import resolve_run_directory

logger = get_logger(name="ResidualCorrection")


def make_correction_psi_fn(
    stage1_apply_fn: Callable,
    stage1_params: any,
    stage2_apply_fn: Callable,
    *,
    soft_bc: bool,
    scale: float = 1.0,
) -> Callable[[any, jnp.ndarray, jnp.ndarray, PlasmaConfig], jnp.ndarray]:
    """psi_final(params, R, Z, cfg) = psi_stage1(frozen) + scale * psi_stage2(params).

    ``stage1_params`` is closed over as a constant, not the traced ``params``
    argument — jax.grad w.r.t. ``params`` therefore only differentiates the
    stage2 branch. R/Z derivatives (needed by the Shafranov operator) still
    flow through both terms, since the sum is what the PDE residual is taken
    of.
    ``scale`` anchors the stage-2 contribution at the stage-1 error magnitude
    so the freshly-initialised corrector does not perturb the converged
    composed field during warmup (arXiv 2407.17213 / 2507.16636).
    """

    def psi_fn(params: any, R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> jnp.ndarray:
        psi1 = apply_psi_fn(stage1_apply_fn, stage1_params, R, Z, cfg, soft_bc=soft_bc)
        psi2 = apply_psi_fn(stage2_apply_fn, params, R, Z, cfg, soft_bc=soft_bc)
        return psi1 + scale * psi2

    return psi_fn


class ResidualCorrectionManager:
    """Trains a stage-2 correction net against a frozen stage-1 NetworkManager.

    Builds its own stage-2 NetworkManager purely to reuse its model init,
    Sampler and optimizer schedule (all public); training itself runs a
    separate loop here since NetworkManager.train_step binds the loss to a
    single raw network, not the psi_stage1+psi_stage2 composition.
    """

    def __init__(
        self,
        stage1: NetworkManager,
        stage2_config: HyperParams,
        seed: int = BASE_SEED + 500,
        scale: float = 1.0,
    ) -> None:
        self.stage1 = stage1
        self.scale = scale
        # The envelope (or lack of it) must match stage1's convention: if
        # stage1 is hard-BC, stage2's own output must also vanish at the edge
        # for the sum to still satisfy psi=0 there.
        stage2_config = stage2_config.replace(soft_bc=stage1.config.soft_bc)
        self.stage2 = NetworkManager(stage2_config, seed=seed)
        self.psi_fn = make_correction_psi_fn(
            stage1.state.apply_fn,
            stage1.state.params,
            self.stage2.model.apply,
            soft_bc=stage2_config.soft_bc,
            scale=scale,
        )
        self._train_step_jit = jax.jit(self._train_step, static_argnames=("soft_bc",))

    @property
    def config(self) -> HyperParams:
        return self.stage2.config

    def _train_step(
        self,
        state: train_state.TrainState,
        inputs: FluxInput,
        weight_boundary_condition: float,
        huber_delta: float,
        weight_flux_scale: float,
        soft_bc: bool,
    ) -> tuple[
        train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        @jax.checkpoint
        def loss_wrapper(
            params: any,
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            # rba_eta=0.0 (default): RBA point-reweighting stays off for the
            # correction stage. pinn_loss_function's return arity depends on
            # rba_eta (5-tuple with rba_weights only when RBA is active), so
            # slice defensively instead of unpacking a fixed count.
            result = pinn_loss_function(
                self.psi_fn,
                params,
                inputs.R_sample,
                inputs.Z_sample,
                inputs.config,
                weight_boundary_condition=weight_boundary_condition,
                huber_delta=huber_delta,
                weight_flux_scale=weight_flux_scale,
                soft_bc=soft_bc,
            )
            total, l_res, l_dir, l_per_cfg = result[:4]
            return total, (l_res, l_dir, l_per_cfg)

        (loss, (l_res, l_dir, l_per_cfg)), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(
            state.params
        )
        grad_norm = optax.tree_utils.tree_norm(grads)
        return state.apply_gradients(grads=grads), loss, l_res, l_dir, l_per_cfg, grad_norm

    def train_epoch(self, epoch: int) -> tuple[float, float, float, float]:
        stage2 = self.stage2
        stage2.sampler.precompute_coordinate_samples()
        loss = l_res = l_dir = grad_norm = 0.0
        all_losses = []

        for i in range(0, len(stage2.train_set), stage2.config.batch_size):
            batch = stage2.train_set[i : i + stage2.config.batch_size]
            inputs = stage2.sampler.sample_flux_input(plasma_configs=batch)
            stage2.state, loss, l_res, l_dir, per_config_loss, grad_norm = self._train_step_jit(
                stage2.state,
                inputs,
                stage2.config.weight_boundary_condition,
                stage2.config.huber_delta,
                stage2.config.weight_flux_scale,
                stage2.config.soft_bc,
            )
            all_losses.append(per_config_loss)

        if epoch % RESAMPLING_FREQUENCY == 0 and epoch > 0:
            stage2.train_set = stage2.sampler.resample_train_set(
                train_set=stage2.train_set, epoch=epoch, per_config_losses=all_losses
            )

        return float(loss), float(l_res), float(l_dir), float(grad_norm)

    def train(self, epochs: int | None = None) -> None:
        """Run the stage-2 training loop for ``epochs`` (default: stage2.epochs)."""
        total_epochs = self.stage2.epochs if epochs is None else epochs
        for epoch in range(total_epochs):
            start = time.perf_counter()
            loss, l_res, l_dir, grad_norm = self.train_epoch(epoch)
            elapsed = time.perf_counter() - start
            if (epoch + 1) % LOG_FREQUENCY == 0 or epoch + 1 == total_epochs:
                logger.info(
                    f"[stage2] epoch {epoch + 1}/{total_epochs} loss={loss:.4f} "
                    f"residual={l_res:.4f} boundary={l_dir:.4f} "
                    f"||grad||={grad_norm:.3f} ({elapsed:.2f}s)"
                )

    def to_disk(self, stage1_run_dir: Path) -> Path:
        """Persist the correction net into ``<stage1_run_dir>/stage2/``.

        Nested one level below the stage-1 run dir (not a sibling under the
        commit dir) so the generic checkpoint scanner — which only looks for
        network.flax one level below <commit>/ — never mistakes this partial
        net for a standalone single-stage checkpoint.

        Also writes ``stage2_meta.json`` (``{"scale": <float>}``) so the
        output scale round-trips through ``load_combined`` without touching
        HyperParams.
        """

        out_dir = stage1_run_dir / "stage2"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "network.flax").write_bytes(
            flax.serialization.to_bytes(self.stage2.state.params)
        )
        self.stage2.config.to_json(path=str(out_dir / "config.json"))
        (out_dir / "stage2_meta.json").write_text(json.dumps({"scale": self.scale}) + "\n")
        return out_dir


class CombinedManager:
    """Duck-typed NetworkManager stand-in for the shared model_evaluation path.

    Exposes exactly the surface evaluate_plasma_grids / evaluate_plasma_kpis /
    compute_gs_residual_on_points / build_kpi_record need: .config,
    .state.params, .sampler, .make_psi_fn(), .artifact_stem. ``state.params``
    is stage2's params (the traced argument of the composed psi_fn); stage1 is
    baked into the psi_fn closure as a frozen constant.
    """

    def __init__(self, stage2: NetworkManager, psi_fn: Callable, artifact_stem: str) -> None:
        self.config = stage2.config
        self.sampler = stage2.sampler
        self.state = SimpleNamespace(params=stage2.state.params)
        self.artifact_stem = artifact_stem
        self._psi_fn = psi_fn

    def make_psi_fn(self) -> Callable:
        return self._psi_fn


def load_combined(stage1_name: str) -> CombinedManager:
    """Load a saved stage1 checkpoint plus its stage2 correction net.

    ``stage1_name`` is a ``commit/run`` name (as accepted by
    ``resolve_run_directory``); the stage2 net is expected at
    ``<run_dir>/stage2/`` (see ``ResidualCorrectionManager.to_disk``).

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
    stage2 = NetworkManager(hp2)
    stage2.state = stage2.state.replace(params=stage2.from_disk(stage2_dir / "network.flax"))

    psi_fn = make_correction_psi_fn(
        stage1.state.apply_fn,
        stage1.state.params,
        stage2.model.apply,
        soft_bc=hp2.soft_bc,
        scale=scale,
    )
    run_name = stage1_name.split("/")[-1]
    return CombinedManager(stage2, psi_fn, artifact_stem=f"{run_name}/stage2")


if __name__ == "__main__":
    import argparse
    import json

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
    hp2 = HyperParams(
        hidden_dims=tuple(int(d) for d in args.hidden_dims.split(",")),
        arch=args.arch,
        rwf=args.rwf,
        n_fourier_features=args.fourier_features,
        fourier_sigma=args.fourier_sigma,
        warmup_epochs=warmup,
        decay_epochs=decay,
    )
    logger.info(
        f"stage-2 config: arch={hp2.arch} hidden_dims={hp2.hidden_dims} "
        f"nff={hp2.n_fourier_features} epochs={args.epochs} "
        f"(warmup={warmup} decay={decay})"
    )

    # --- train ---
    manager = ResidualCorrectionManager(stage1_mgr, hp2, scale=args.stage2_scale)
    manager.train(epochs=args.epochs)

    # --- save ---
    stage2_dir = manager.to_disk(stage1_dir)
    logger.info(f"stage-2 saved to {stage2_dir}")

    # --- KPI eval on the composed field via CombinedManager (duck-typed NetworkManager) ---
    # Reconstruct CombinedManager from the in-memory state (avoids a round-trip through disk).
    composed_psi_fn = manager.psi_fn
    combined = CombinedManager(
        stage2=manager.stage2,
        psi_fn=composed_psi_fn,
        artifact_stem=f"{args.stage1}/stage2",
    )
    configs = kpi_benchmark_configs(combined, KPI_EVAL_CONFIGS)
    kpis = evaluate_plasma_kpis(combined, configs, sample_size=KPI_POINTS_PER_CONFIG)

    logger.info(
        f"composed KPIs — median={kpis['loss_median']:.4e} "
        f"p95={kpis['loss_p95']:.4e} "
        f"core_median={kpis['core_loss_median']:.4e} "
        f"bnd_leak={kpis['boundary_leak_max']:.4e}"
    )

    record = build_kpi_record(
        combined,
        kpis,
        n_configs=KPI_EVAL_CONFIGS,
        n_points=KPI_POINTS_PER_CONFIG,
        core_rho=0.85,
        network_name=f"{args.stage1}/stage2",
    )
    kpis_path = stage2_dir / "kpis.json"
    kpis_path.write_text(json.dumps(record, indent=2) + "\n")
    logger.info(f"kpis.json written to {kpis_path}")
