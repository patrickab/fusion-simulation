import argparse
from collections import deque
from contextlib import nullcontext, suppress
import csv
from datetime import datetime
import functools
from functools import partial
import json
import logging
from pathlib import Path
import re
import shutil
import time
from typing import Callable, Literal

from flax import linen as nn
import flax.serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from scipy.stats import qmc

from src.engine.physics import pinn_loss_function
from src.engine.plasma import (
    boundary_normalized_radius,
    calculate_poloidal_boundary,
    get_poloidal_points,
)
from src.lib.config import KPI_EVAL_CONFIGS, KPI_POINTS_PER_CONFIG, Filepaths, current_commit
from src.lib.geometry_config import (
    PlasmaConfig,
    PlasmaGeometry,
    PlasmaState,
)
from src.lib.logger import get_logger
from src.lib.network_config import DomainBounds, FluxInput, HyperParams

console = Console()
logger = get_logger(
    name="Network",
)


def denormalize_psi(
    psi_n: jnp.ndarray,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    cfg: PlasmaConfig,
    *,
    soft_bc: bool = False,
) -> jnp.ndarray:
    """Map raw network output to physical psi.

    Hard-enforces psi=0 (and, since the envelope is theta-independent along
    the boundary, dpsi/dtheta=0) at the plasma edge via a multiplicative
    envelope, so boundary conditions hold by construction instead of via a
    soft loss penalty.

    ``soft_bc=True`` skips the envelope multiply: soft-BC checkpoints train
    and evaluate on the raw physical flux, with boundary conditions enforced
    by Dirichlet/Neumann penalties instead.
    """
    scaled = psi_n.squeeze() * cfg.State.F_axis * cfg.Geometry.a
    if soft_bc:
        return scaled
    envelope = 1.0 - boundary_normalized_radius(R, Z, cfg.Boundary) ** 2
    return envelope * scaled


def apply_psi_fn(
    apply_fn: Callable,
    params: any,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    cfg: PlasmaConfig,
    *,
    soft_bc: bool = False,
) -> jnp.ndarray:
    """Adapter converting neural network output to physical psi flux.

    Shared by training, inference, and evaluation call sites; bind ``apply_fn``
    with ``functools.partial`` to get a ``(params, R, Z, cfg) -> psi`` callable.
    ``soft_bc`` is forwarded to :func:`denormalize_psi`.
    """
    inp = FluxInput(R_sample=R, Z_sample=Z, config=cfg)
    p_n, r_n, z_n = inp.normalize()
    psi_n = apply_fn(params, r=r_n, z=z_n, **p_n)
    return denormalize_psi(psi_n, R, Z, cfg, soft_bc=soft_bc)


# --- Network (simple MLP) ---
class FluxPINN(nn.Module):
    hidden_dims: tuple[int, ...]
    # Random Fourier features on the spatial (r, z) inputs to counter spectral
    # bias (Wang et al. 2021). 0 disables; the projection matrix is drawn from
    # a fixed PRNG key so train and reload see the identical embedding.
    n_fourier_features: int = 0
    fourier_sigma: float = 2.0

    @nn.compact
    def __call__(
        self,
        r: jnp.ndarray,  # r coordinates of shape (B, N)
        z: jnp.ndarray,  # z coordinates of shape (B, N)
        r0: jnp.ndarray,  # major radius (reactor center)
        a: jnp.ndarray,  # minor radius (poloidal plasma)
        kappa: jnp.ndarray,  # elongation factor
        delta: jnp.ndarray,  # triangularity factor
        p0: jnp.ndarray,  # central plasma pressure
        f_axis: jnp.ndarray,  # toroidal field on psi-axis (tesla-meters)
        alpha: jnp.ndarray,  # pressure profile shaping parameter
        exponent: jnp.ndarray,  # current profile shaping parameter
    ) -> jnp.ndarray:
        """
        Note: JAX requires a stateless definition for neural networks.
              This prohibits passing the FluxInput dataclass directly.
              Therefore all inputs must be passed as arguments.

        Args:
            Batch of Normalized FluxInput parameters:

        Returns:
            Normalized Flux prediction of shape (B, N, 1)
        """
        # Broadcast all inputs to match coordinate shape (B, N)
        target_shape = r.shape
        inputs = [r, z, r0, a, kappa, delta, p0, f_axis, alpha, exponent]
        x = jnp.stack([jnp.broadcast_to(input, target_shape) for input in inputs], axis=-1)

        if self.n_fourier_features > 0:
            proj_matrix = self.fourier_sigma * jax.random.normal(
                jax.random.PRNGKey(0), (2, self.n_fourier_features)
            )
            proj = 2.0 * jnp.pi * (x[..., :2] @ proj_matrix)
            x = jnp.concatenate([x, jnp.cos(proj), jnp.sin(proj)], axis=-1)

        for dim in self.hidden_dims:
            x = nn.Dense(features=dim, dtype=jnp.float32)(x)
            x = nn.swish(x)

        psi_hat = nn.Dense(features=1)(x)
        return psi_hat


# --- Sampler ---
BASE_SEED = 42
RESAMPLING_FREQUENCY = 10  # Resample training set every N epochs
LOG_FREQUENCY = 10  # Log training metrics every N epochs
N_VALIDATION_SIZE = KPI_EVAL_CONFIGS  # Number of validation plasma configs
VALIDATION_FREQUENCY = 5 * LOG_FREQUENCY  # Evaluate validation set every N epochs
# Cap rows shown in the live table; unbounded growth desyncs Rich's redraw
# once it overflows the terminal, corrupting scrollback.
LIVE_TABLE_MAX_ROWS = 15


class Sampler:
    """Data generation engine for PINN training.

    Uses quasi-random Sobol sequences to guarantee even parameter space coverage without clustering
    Uses adaptive sampling to focus on high-loss regions, improving model convergence.
    """

    def __init__(self, config: HyperParams, seed: int = BASE_SEED) -> None:
        self.config = config
        self.seed = seed
        self._domain_lower_bounds, self._domain_upper_bounds = DomainBounds.get_bounds()

        # Cache compiled batch compute function to avoid JAX recompilation
        # Vectorize over axis 0 (plasma_configs)
        self._compute_batch = jax.jit(
            jax.vmap(Sampler._compute_single_config, in_axes=(0, None, None, None))
        )

        # Instantiate separate Sobol samplers for domain, interior points, and boundary points.
        # Avoids repetitive instatiation.
        self._sobol_domain = qmc.Sobol(
            d=len(self._domain_lower_bounds),
            scramble=True,
            seed=self.seed,
        )
        self._sobol_inner = qmc.Sobol(d=2, scramble=True, seed=self.seed + 1)
        self._sobol_boundary = qmc.Sobol(d=1, scramble=True, seed=self.seed + 2)

        self.precompute_coordinate_samples()

    def _get_sobol_sample(
        self,
        n_samples: int,
        lower_bounds: jnp.ndarray | None = None,
        upper_bounds: jnp.ndarray | None = None,
        sobol_sampler: Literal["interior", "boundary", "domain"] | None = None,
    ) -> jnp.ndarray:
        """Generate Sobol sequence samples within specified bounds."""

        sobol_sampler = {
            "interior": self._sobol_inner,
            "boundary": self._sobol_boundary,
            "domain": self._sobol_domain,
        }.get(sobol_sampler, self._sobol_domain)

        sample_unit = jnp.array(sobol_sampler.random(n_samples), dtype=jnp.float32)
        return sample_unit * (upper_bounds - lower_bounds) + lower_bounds

    def precompute_coordinate_samples(
        self,
        n_inner_samples: int | None = None,
        n_boundary_samples: int | None = None,
    ) -> None:
        """Precompute Sobol interior and boundary coordinates for one epoch."""
        n_inner = self.config.n_rz_inner_samples if n_inner_samples is None else n_inner_samples
        n_boundary = (
            self.config.n_rz_boundary_samples if n_boundary_samples is None else n_boundary_samples
        )

        inner_samples = self._get_sobol_sample(
            n_samples=n_inner,
            lower_bounds=jnp.array([0.0, 0.0], dtype=jnp.float32),
            upper_bounds=jnp.array([2 * jnp.pi, 1.0], dtype=jnp.float32),
            sobol_sampler="interior",
        )
        self._theta_int = inner_samples[:, 0]
        self._rho_int = jnp.sqrt(inner_samples[:, 1])

        boundary_samples = self._get_sobol_sample(
            n_samples=n_boundary,
            lower_bounds=jnp.array([0.0], dtype=jnp.float32),
            upper_bounds=jnp.array([2 * jnp.pi], dtype=jnp.float32),
            sobol_sampler="boundary",
        )
        self._theta_b = boundary_samples[:, 0]

    def resample_train_set(
        self,
        train_set: jnp.ndarray,
        epoch: int,
        per_config_losses: list[jnp.ndarray],
    ) -> jnp.ndarray:
        """Generate new training set.

        50% drawn via Sobol Sequence: maintains global coverage.
        50% drawn via Adaptive Sampling: focuses on high-loss regions for improved learning.
        """
        n_sobol = self.config.n_train // 2
        n_adaptive = self.config.n_train - n_sobol

        sobol_samples = self._get_sobol_sample(
            n_samples=n_sobol,
            lower_bounds=self._domain_lower_bounds,
            upper_bounds=self._domain_upper_bounds,
        )

        all_losses = jnp.concatenate(per_config_losses)
        top_k_indices = jnp.argsort(all_losses)[-n_adaptive:]
        top_k_configs = train_set[top_k_indices]

        bounds_range = self._domain_upper_bounds - self._domain_lower_bounds
        key = jax.random.PRNGKey(self.seed + epoch)
        noise = jax.random.normal(key, top_k_configs.shape) * (
            self.config.sigma_residual_adaptive_sampling * bounds_range
        )

        adaptive_samples = jnp.clip(
            top_k_configs + noise,
            a_min=self._domain_lower_bounds,
            a_max=self._domain_upper_bounds,
        )

        return jnp.concatenate([sobol_samples, adaptive_samples], axis=0)

    @staticmethod
    def _compute_single_config(
        plasma_config: jnp.ndarray,
        theta_int: jnp.ndarray,
        rho_int: jnp.ndarray,
        theta_b: jnp.ndarray,
    ) -> tuple[PlasmaConfig, jnp.ndarray, jnp.ndarray]:
        geometry = PlasmaGeometry(
            R0=plasma_config[0],
            a=plasma_config[1],
            kappa=plasma_config[2],
            delta=plasma_config[3],
        )
        state = PlasmaState(
            p0=plasma_config[4],
            F_axis=plasma_config[5],
            pressure_alpha=plasma_config[6],
            field_exponent=plasma_config[7],
        )
        boundary = calculate_poloidal_boundary(theta_b, geometry)

        # Interior points
        r_interior, z_interior = jax.vmap(
            lambda theta, rho: get_poloidal_points(theta, geometry, rho)
        )(theta_int, rho_int)

        return (
            PlasmaConfig(Geometry=geometry, Boundary=boundary, State=state),
            r_interior,
            z_interior,
        )

    def sample_flux_input(
        self,
        plasma_configs: jnp.ndarray,
    ) -> FluxInput:
        """Sample interior and boundary points for a batch of plasma configurations.

        All configurations within each epoch share the same set of normalized RZ-coordinates.
        Normalized coordinates are then individually scaled to the plasma configuration on the GPU.
        This minimizes CPU-to-GPU memory transfer.

        Overfitting is avoided by resampling coordinates each epoch.
        """
        configs, R_int, Z_int = self._compute_batch(
            plasma_configs, self._theta_int, self._rho_int, self._theta_b
        )

        return FluxInput(R_sample=R_int, Z_sample=Z_int, config=configs)


# --- Manager for Training / Inference ---
class NetworkManager:
    # Optional consumer of finalized metrics-table rows (the HPO TUI appends them
    # to its sequential per-trial log). None = CLI mode, table rendering only.
    metrics_row_sink: Callable[[tuple[str, ...]], None] | None = None

    def __init__(
        self,
        config: HyperParams,
        seed: int = BASE_SEED,
        n_validation_size: int = N_VALIDATION_SIZE,
        test_mode: bool = False,
        output_dir: Path | None = None,
        name: str = "default",
    ) -> None:
        self.config = config
        self.seed = seed
        self.n_validation_size = n_validation_size
        self.test_mode = test_mode
        # None writes a flattened benchmark slug; HPO supplies its study dir.
        self.output_dir = output_dir
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(
                "Network name must contain only letters, numbers, underscores, or hyphens"
            )
        self.name = name
        self.model = FluxPINN(
            hidden_dims=config.hidden_dims,
            n_fourier_features=config.n_fourier_features,
            fourier_sigma=config.fourier_sigma,
        )
        self.sampler: Sampler = Sampler(config, seed=self.seed)
        self._validation_kpi_configs: list | None = None
        self.state = self._init_state()

        self._psi_fn_jit = jax.jit(self.make_psi_fn())

        self.train_set = self.sampler._get_sobol_sample(
            n_samples=self.config.n_train,
            lower_bounds=self.sampler._domain_lower_bounds,
            upper_bounds=self.sampler._domain_upper_bounds,
        )
        self.training_log: list[dict] = []
        self.artifact_stem: str | None = None

    def _new_artifact_stem(self) -> str:
        """Create a flat benchmark slug or an HPO-local trial stem."""
        # Second resolution: fast runs (HPO --test trials take ~8s) collided at
        # minute resolution and silently overwrote each other's artifacts.
        # ponytail: same-second parallel runs on one machine would still collide.
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.output_dir:
            return f"pinn_{timestamp}"
        return f"{timestamp}_{self.name}_{current_commit()}"

    def run_dir(self) -> Path:
        """Return this manager's direct benchmark or HPO trial directory."""
        if self.artifact_stem is None:
            self.artifact_stem = self._new_artifact_stem()
        base = self.output_dir or Filepaths.BENCHMARKS
        return base / self.artifact_stem

    def discard_unsaved_run(self) -> None:
        """Delete the benchmark run dir unless this run's checkpoint was saved.

        A run dir exists iff its checkpoint was kept: --test, pruned, failed
        and aborted runs leave nothing behind.
        """
        if self.artifact_stem is None:
            return
        run_dir = self.run_dir()
        if (run_dir / "network.flax").exists():
            return
        shutil.rmtree(run_dir, ignore_errors=True)
        if self.output_dir:
            with suppress(OSError):
                run_dir.parent.rmdir()

    def to_disk(self) -> str:
        if self.artifact_stem is None:
            self.artifact_stem = self._new_artifact_stem()

        run_dir = self.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

        artifact_flax_path = run_dir / "network.flax"
        artifact_json_path = run_dir / "config.json"
        artifact_log_path = run_dir / "training.csv"

        artifact_flax_path.write_bytes(flax.serialization.to_bytes(self.state.params))
        self.config.to_json(path=str(artifact_json_path))

        if self.training_log:
            with open(artifact_log_path, "w") as f:
                f.write(
                    "epoch,lr,moving_avg_loss,val_kpi_median,residual,boundary,grad_norm,epoch_time\n"
                )
                for entry in self.training_log:
                    vl = (
                        f"{entry['val_kpi_median']:.6f}"
                        if entry["val_kpi_median"] is not None
                        else ""
                    )
                    f.write(
                        f"{entry['epoch']},{entry['lr']:.2e},{entry['moving_avg_loss']:.6f},"
                        f"{vl},{entry['residual']:.6f},{entry['boundary']:.6f},"
                        f"{entry['grad_norm']:.6f},{entry['epoch_time']:.4f}\n"
                    )

        self._benchmark_network()
        self._save_training_curves_plot()
        return self.artifact_stem

    def from_disk(self, pinn_path) -> any:  # noqa
        """Load Flax model parameters from disk."""
        with open(pinn_path, "rb") as f:
            return flax.serialization.from_bytes(self.state.params, f.read())

    def _benchmark_network(self) -> None:
        """Save the residual montage, grids and kpis.json for this run."""
        if self.test_mode:
            return
        # model_evaluation imports NetworkManager from this module,
        # so a top-level import here would be circular.
        from src.engine.model_evaluation import (
            EVAL_RESOLUTION,
            N_PLOTS,
            build_kpi_record,
            evaluate_plasma_grids,
            evaluate_plasma_kpis,
            kpi_benchmark_configs,
            plot_plasma_grid_montage,
        )

        configs = kpi_benchmark_configs(self, KPI_EVAL_CONFIGS)

        kpis = evaluate_plasma_kpis(self, configs, sample_size=KPI_POINTS_PER_CONFIG)
        grids = evaluate_plasma_grids(
            self, configs[:N_PLOTS], resolution=EVAL_RESOLUTION, quantities=("residual",)
        )
        run_dir = self.run_dir()

        plot_plasma_grid_montage(
            grids,
            run_dir / "residual.png",
            quantity="residual",
            title=self.artifact_stem,
            metadata=self.config.to_dict(),
            display_parameters=(
                "huber_delta",
                "learning_rate_max",
                "n_fourier_features",
                "lbfgs_steps",
            ),
            kpis=kpis,
        )

        # Store KPIs as valid JSON (indent=2 for terminal readability).
        record = build_kpi_record(self, kpis, KPI_EVAL_CONFIGS, KPI_POINTS_PER_CONFIG, 0.85)
        (run_dir / "kpis.json").write_text(json.dumps(record, indent=2) + "\n")

        logger.info(f"residual plot saved to {run_dir}")

    def _save_training_curves_plot(self) -> None:
        """Loss/val-loss + LR/grad-norm overview, straight from training.csv."""
        if self.test_mode or not self.training_log:
            return
        from src.engine.model_evaluation import plot_training_curves

        run_dir = self.run_dir()
        plot_training_curves(
            run_dir / "training.csv", run_dir / "training_curves.png", title=self.artifact_stem
        )

    def _init_state(self) -> train_state.TrainState:
        """Initialize the training state with dummy data."""
        key = jax.random.PRNGKey(self.seed)
        d_rz = jnp.ones((1, self.config.n_rz_inner_samples))
        d_p = jnp.ones(1)

        geom = PlasmaGeometry(R0=d_p, a=d_p, kappa=d_p, delta=d_p)
        state = PlasmaState(p0=d_p, F_axis=d_p, pressure_alpha=d_p, field_exponent=d_p)
        boundary = calculate_poloidal_boundary(jnp.zeros(1), geom)

        dummy_config = PlasmaConfig(Geometry=geom, Boundary=boundary, State=state)
        dummy_input = FluxInput(R_sample=d_rz, Z_sample=d_rz, config=dummy_config)

        norm_params, r_n, z_n = dummy_input.normalize()
        params = self.model.init(key, r=r_n, z=z_n, **norm_params)

        steps_per_epoch = self.config.n_train // self.config.batch_size
        total_steps = (self.config.warmup_epochs + self.config.decay_epochs) * steps_per_epoch
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate_max,
            warmup_steps=self.config.warmup_epochs * steps_per_epoch,
            decay_steps=total_steps,
            end_value=self.config.learning_rate_min,
        )
        self._lr_schedule = schedule
        tx = optax.adamw(learning_rate=schedule, weight_decay=self.config.weight_decay)
        return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    @property
    def epochs(self) -> int:
        """Calculate total epochs."""
        return self.config.warmup_epochs + self.config.decay_epochs

    @staticmethod
    def compute_loss(
        params: any,
        apply_fn: Callable,
        inputs: FluxInput,
        weight_boundary_condition: float,
        huber_delta: float,
        weight_flux_scale: float,
        soft_bc: bool,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Pure function to compute physics loss. Reusable for both training and evaluation."""
        psi_fn = functools.partial(apply_psi_fn, apply_fn, soft_bc=soft_bc)

        total, l_res, l_dir, l_per_cfg = pinn_loss_function(
            psi_fn,
            params,
            inputs.R_sample,
            inputs.Z_sample,
            inputs.config,
            weight_boundary_condition=weight_boundary_condition,
            huber_delta=huber_delta,
            weight_flux_scale=weight_flux_scale,
            soft_bc=soft_bc,
        )
        return total, (l_res, l_dir, l_per_cfg)

    @staticmethod
    @partial(jax.jit, static_argnames=("soft_bc",))
    def eval_step(
        state: train_state.TrainState,
        inputs: FluxInput,
        weight_boundary_condition: float,
        huber_delta: float,
        weight_flux_scale: float,
        soft_bc: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Fast, gradient-free evaluation step returning all loss components."""
        total, (l_res, l_dir, l_per_cfg) = NetworkManager.compute_loss(
            state.params,
            state.apply_fn,
            inputs,
            weight_boundary_condition,
            huber_delta,
            weight_flux_scale,
            soft_bc,
        )
        return total, l_res, l_dir, l_per_cfg

    def validation_configs(self) -> list:
        """The manager's fixed validation configs: the first n_validation_size of the
        shared KPI config stream (kpi_benchmark_configs), so training-time tracking,
        HPO ranking and kpis.json all score identical PlasmaConfig objects.

        PlasmaConfig construction bakes the building sampler's boundary-theta draw
        into the Fourier fit, so configs must come from ``self.sampler`` — a
        separately-seeded sampler yields subtly different boundaries for the same
        domain vectors.
        """
        # model_evaluation imports NetworkManager from this module — lazy import avoids
        # the circular dependency at module load time.
        from src.engine.model_evaluation import kpi_benchmark_configs

        if self._validation_kpi_configs is None:
            self._validation_kpi_configs = kpi_benchmark_configs(self, self.n_validation_size)
        return self._validation_kpi_configs

    def _calculate_validation_kpi(self) -> float:
        """Median |R_GS| over the fixed validation configs at KPI_POINTS_PER_CONFIG.

        Runs the FULL global protocol, so when n_validation_size equals
        KPI_EVAL_CONFIGS the tracked val_kpi_median matches kpis.json's
        loss_median exactly.
        """
        from src.engine.model_evaluation import evaluate_plasma_kpis

        return evaluate_plasma_kpis(
            self, self.validation_configs(), sample_size=KPI_POINTS_PER_CONFIG
        )["loss_median"]

    @staticmethod
    @partial(jax.jit, static_argnames=("soft_bc",))
    def train_step(
        state: train_state.TrainState,
        inputs: FluxInput,
        weight_boundary_condition: float,
        huber_delta: float,
        weight_flux_scale: float,
        soft_bc: bool,
    ) -> tuple[
        train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """Perform a single training step using physics-informed gradients."""

        def loss_wrapper(
            params: any,
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            return NetworkManager.compute_loss(
                params,
                state.apply_fn,
                inputs,
                weight_boundary_condition,
                huber_delta,
                weight_flux_scale,
                soft_bc,
            )

        (loss, (l_res, l_dir, l_per_cfg)), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(
            state.params
        )
        grad_norm = optax.tree_utils.tree_norm(grads)
        return state.apply_gradients(grads=grads), loss, l_res, l_dir, l_per_cfg, grad_norm

    def train_epoch(self, epoch: int) -> tuple[float, float, float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (total_loss, residual_loss, boundary_loss, grad_norm).
        """
        loss, l_res, l_dir, b_grad_norm = 0.0, 0.0, 0.0, 0.0
        self.sampler.precompute_coordinate_samples()

        all_losses = []
        grad_norms = []

        for i in range(0, len(self.train_set), self.config.batch_size):
            train_batch = self.train_set[i : i + self.config.batch_size]
            inputs = self.sampler.sample_flux_input(plasma_configs=train_batch)
            self.state, loss, l_res, l_dir, per_config_loss, b_grad_norm = self.train_step(
                state=self.state,
                inputs=inputs,
                weight_boundary_condition=self.config.weight_boundary_condition,
                huber_delta=self.config.huber_delta,
                weight_flux_scale=self.config.weight_flux_scale,
                soft_bc=self.config.soft_bc,
            )
            all_losses.append(per_config_loss)
            grad_norms.append(b_grad_norm)

        avg_grad_norm = float(jnp.mean(jnp.array(grad_norms)))

        if epoch % RESAMPLING_FREQUENCY == 0 and epoch > 0:
            self.train_set = self.sampler.resample_train_set(
                train_set=self.train_set,
                epoch=epoch,
                per_config_losses=all_losses,
            )

        return float(loss), float(l_res), float(l_dir), avg_grad_norm

    @staticmethod
    def _new_table() -> Table:
        table = Table(title="Training Metrics", show_header=True, header_style="bold cyan")
        table.add_column("Epoch", justify="right", style="cyan")
        table.add_column("LR", justify="right", style="yellow")
        table.add_column("||∇L||", justify="right", style="magenta")
        table.add_column("Loss", justify="right", style="magenta")
        table.add_column("Val KPI", justify="right", style="green")
        table.add_column("Time/Ep", justify="right")
        return table

    def _init_metrics_display(self) -> None:
        """Set up Rich table, progress bar, and accumulators for the training display."""
        # ponytail: fixed-size rolling window, not the full history — an unbounded table
        # eventually taller than the terminal desyncs Rich's cursor-based redraw, corrupting
        # the display whenever the user scrolls
        self._table_rows: deque[tuple[str, ...]] = deque(maxlen=LIVE_TABLE_MAX_ROWS)
        self._table = self._new_table()

        self._progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(style="cyan", complete_style="bold cyan"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self._epoch_task = self._progress.add_task("Training", total=self.epochs)

        self._accumulated_loss = 0.0
        self._accumulated_residual = 0.0
        self._accumulated_boundary = 0.0
        self._accumulated_grad_norm = 0.0
        self._accumulated_time = 0.0

    def training_renderable(self) -> Panel:
        """Current metrics table + progress bar; rendered by Live or the HPO TUI."""
        return Panel(Group(self._table, self._progress), border_style="cyan")

    def _init_live_display(self) -> Live:
        self._init_metrics_display()
        return Live(
            self.training_renderable(),
            refresh_per_second=10,
            console=console,
            vertical_overflow="visible",
        )

    def _log_metric(
        self,
        epoch: int,
        loss: float,
        residual: float,
        boundary: float,
        val_kpi_median: float | None,
        lr: float,
        grad_norm: float,
        epoch_time: float,
    ) -> None:
        self._accumulated_loss += loss
        self._accumulated_residual += residual
        self._accumulated_boundary += boundary
        self._accumulated_grad_norm += grad_norm
        self._accumulated_time += epoch_time

        if (epoch + 1) % LOG_FREQUENCY == 0:
            moving_avg_loss = self._accumulated_loss / LOG_FREQUENCY
            moving_avg_residual = self._accumulated_residual / LOG_FREQUENCY
            moving_avg_boundary = self._accumulated_boundary / LOG_FREQUENCY
            moving_avg_grad_norm = self._accumulated_grad_norm / LOG_FREQUENCY
            moving_avg_time = self._accumulated_time / LOG_FREQUENCY

            self.training_log.append(
                {
                    "epoch": epoch + 1,
                    "lr": lr,
                    "moving_avg_loss": moving_avg_loss,
                    "val_kpi_median": val_kpi_median,
                    "residual": moving_avg_residual,
                    "boundary": moving_avg_boundary,
                    "grad_norm": moving_avg_grad_norm,
                    "epoch_time": moving_avg_time,
                }
            )

            row = _metrics_row(
                epoch=epoch + 1,
                total_epochs=self.epochs,
                lr=lr,
                grad_norm=moving_avg_grad_norm,
                loss=moving_avg_loss,
                val_kpi_median=val_kpi_median,
                epoch_time=moving_avg_time,
            )
            self._table_rows.append(row)
            if self.metrics_row_sink is not None:
                self.metrics_row_sink(row)
            # build fully, then swap: the TUI thread may render self._table mid-update
            table = self._new_table()
            for row in self._table_rows:
                table.add_row(*row)
            self._table = table

            self._accumulated_loss = 0.0
            self._accumulated_residual = 0.0
            self._accumulated_boundary = 0.0
            self._accumulated_grad_norm = 0.0
            self._accumulated_time = 0.0
        else:
            count = (epoch + 1) % LOG_FREQUENCY
            self.training_log.append(
                {
                    "epoch": epoch + 1,
                    "lr": lr,
                    "moving_avg_loss": self._accumulated_loss / count,
                    "val_kpi_median": val_kpi_median,
                    "residual": self._accumulated_residual / count,
                    "boundary": self._accumulated_boundary / count,
                    "grad_norm": self._accumulated_grad_norm / count,
                    "epoch_time": self._accumulated_time / count,
                }
            )

        self._progress.update(self._epoch_task, advance=1)
        if self._live is not None:
            self._live.update(self.training_renderable())

    def train(
        self,
        save_to_disk: bool = True,
        validation_callback: Callable[[int, float | None], None] | None = None,
        show_progress: bool = True,
    ) -> float:
        self.artifact_stem = self._new_artifact_stem()
        run_dir = self.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        # Full per-run log file; DEBUG epoch lines land here but not on the
        # Rich console handler (which stays at INFO next to the live table).
        file_handler = logging.FileHandler(run_dir / "train.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        previous_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.debug(f"run {self.artifact_stem} hyperparams: {self.config.to_dict()}")

        try:
            # Metrics (table + training_log CSV) are always tracked; Live only owns the
            # terminal in CLI mode. The HPO TUI renders training_renderable() itself.
            self._live = None
            if show_progress:
                live = self._init_live_display()
            else:
                self._init_metrics_display()
                live = nullcontext()
            val_kpi_median = None
            with live as active_live:
                if show_progress:
                    self._live = active_live
                for epoch in range(self.epochs):
                    start_time = time.perf_counter()
                    loss, residual, boundary, grad_norm = self.train_epoch(epoch)
                    epoch_time = time.perf_counter() - start_time

                    val_kpi_median = None
                    if (epoch + 1) % VALIDATION_FREQUENCY == 0:
                        val_kpi_median = self._calculate_validation_kpi()
                    if validation_callback is not None:
                        validation_callback(epoch + 1, val_kpi_median)

                    lr = float(self._lr_schedule(self.state.step))
                    self._log_metric(
                        epoch, loss, residual, boundary, val_kpi_median, lr, grad_norm, epoch_time
                    )

            if self.config.lbfgs_steps > 0:
                self.lbfgs(self.config.lbfgs_steps)
                val_kpi_median = self._calculate_validation_kpi()

            if val_kpi_median is None:
                val_kpi_median = self._calculate_validation_kpi()
            logger.debug(f"run {self.artifact_stem} final val_kpi_median={val_kpi_median:.3f}")
            with open(run_dir / "train.log", "a", encoding="utf-8") as log_f:
                Console(file=log_f, width=100, color_system=None).print(
                    Panel(self._table, border_style="cyan")
                )

            if save_to_disk:
                self.to_disk()
            return val_kpi_median
        finally:
            logger.removeHandler(file_handler)
            file_handler.close()
            logger.setLevel(previous_level)

    def lbfgs(self, steps: int) -> None:
        """Polish AdamW-trained params with L-BFGS on one fixed batch.

        Classic PINN two-stage optimization. Full-batch L-BFGS does not fit in
        VRAM, so this uses a single fixed training batch — same memory
        footprint as one train_step. ponytail: fixed-batch polish can overfit
        those configs; the held-out KPI eval is the guard.
        """
        self.sampler.precompute_coordinate_samples()
        batch = self.train_set[: self.config.batch_size]
        inputs = self.sampler.sample_flux_input(plasma_configs=batch)

        def loss_fn(params: any) -> jnp.ndarray:
            total, _ = NetworkManager.compute_loss(
                params,
                self.state.apply_fn,
                inputs,
                self.config.weight_boundary_condition,
                self.config.huber_delta,
                self.config.weight_flux_scale,
                self.config.soft_bc,
            )
            return total

        opt = optax.lbfgs()

        @jax.jit
        def step(params: any, opt_state: any) -> tuple[any, any, jnp.ndarray]:
            value, grad = optax.value_and_grad_from_state(loss_fn)(params, state=opt_state)
            updates, opt_state = opt.update(
                grad, opt_state, params, value=value, grad=grad, value_fn=loss_fn
            )
            return optax.apply_updates(params, updates), opt_state, value

        params = self.state.params
        opt_state = opt.init(params)
        for i in range(steps):
            params, opt_state, value = step(params, opt_state)
            if (i + 1) % 20 == 0 or i == 0:
                logger.info(f"L-BFGS polish step {i + 1}/{steps}: loss {float(value):.6f}")
        self.state = self.state.replace(params=params)

    def get_psi(self, R: jnp.ndarray, Z: jnp.ndarray, config: PlasmaConfig) -> jnp.ndarray:
        """Evaluate magnetic flux psi at physical coordinates.

        Convenience method for inference and visualization. Uses pre-compiled
        psi function from initialization. Handles input normalization and output
        denormalization internally.

        Args:
            R: Batch of major radial coordinates [m]
            Z: Batch of vertical coordinates [m]
            config: Plasma geometry and state parameters

        Returns:
            Array of flux values in Weber
        """
        return self._psi_fn_jit(self.state.params, R, Z, config)

    def make_psi_fn(self) -> Callable[[any, jnp.ndarray, jnp.ndarray, PlasmaConfig], jnp.ndarray]:
        """Factory returning a psi function bound to this network instance.

        This pattern is necessary due to JAX's functional paradigm, that requires stateless
        pure functions. The closure captures network state (apply_fn), providing a
        stateful callable interface while maintaining JIT compatibility.

        Returns:
            Callable[(params, R, Z, config) -> psi] for physics calculations.
        """
        return functools.partial(
            apply_psi_fn,
            self.model.apply,
            soft_bc=self.config.soft_bc,
        )


def _metrics_row(
    epoch: int,
    total_epochs: int,
    lr: float,
    grad_norm: float,
    loss: float,
    val_kpi_median: float | None,
    epoch_time: float,
) -> tuple[str, ...]:
    """One Training Metrics table row; shared by the live display and show_run replay.

    Loss alone (no residual/boundary breakdown) — both stay in training.csv for
    post-hoc analysis, this table is just the terminal-width-constrained live view.
    """
    return (
        f"{epoch}/{total_epochs}",
        f"{lr:.2e}",
        f"{grad_norm:.2e}",
        f"{loss:.2e}",
        f"{val_kpi_median:.2e}" if val_kpi_median is not None else "-",
        f"{epoch_time:.2f}s",
    )


def show_run(run: str) -> None:
    """Re-render the Training Metrics table for a stored run from training.csv.

    Accepts a run dir path, a flat benchmark slug, or a bare HPO trial stem.
    """
    run_dir = Path(run)
    if not run_dir.is_dir():
        candidates = [
            Filepaths.BENCHMARKS / run,
            *Filepaths.BENCHMARKS.glob(f"*/{run}"),
            *(Filepaths.DATA / "hpo").glob(f"*/{run}"),
        ]
        run_dir = next((p for p in candidates if p.is_dir()), run_dir)
    csv_path = run_dir / "training.csv"
    if not csv_path.exists():
        raise SystemExit(f"no training.csv found for run '{run}'")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    total_epochs = int(rows[-1]["epoch"])
    table = NetworkManager._new_table()
    # New CSVs have val_kpi_median; fall back to val_loss for legacy files.
    val_col = "val_kpi_median" if "val_kpi_median" in rows[0] else "val_loss"
    for r in rows:
        epoch = int(r["epoch"])
        # replay exactly what the live table showed: every LOG_FREQUENCY-th
        # epoch (full moving average), plus the final partial row if any
        if epoch % LOG_FREQUENCY != 0 and epoch != total_epochs:
            continue
        table.add_row(
            *_metrics_row(
                epoch=epoch,
                total_epochs=total_epochs,
                lr=float(r["lr"]),
                grad_norm=float(r["grad_norm"]),
                loss=float(r["moving_avg_loss"]),
                val_kpi_median=float(r[val_col]) if r[val_col] else None,
                epoch_time=float(r["epoch_time"]),
            )
        )
    console.print(Panel(table, border_style="cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINN network")
    parser.add_argument(
        "--test", action="store_true", help="Run with minimal parameters for rapid iteration"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate_max")
    parser.add_argument(
        "--name", default="default", help="Artifact name in <timestamp>_<name>_<commit>"
    )
    parser.add_argument(
        "--fourier-features",
        type=int,
        default=64,
        help="Random Fourier features on (r,z); 0 = off (default 64, per grid-2 ablation)",
    )
    parser.add_argument(
        "--lbfgs", type=int, default=0, help="L-BFGS polish steps after AdamW; 0 = off"
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="PDE loss: >0 = Huber with this delta (default 1.0), 0.0 = MSE",
    )
    parser.add_argument(
        "--weight-flux-scale",
        type=float,
        default=None,
        help="Weight of the interior-mean-ψ collapse-guard hinge (default 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total epochs; split 1:5 warmup:decay (default: HyperParams 100+500)",
    )
    parser.add_argument(
        "--soft-bc",
        action="store_true",
        help="Legacy soft-BC training: raw ψ + Dirichlet/Neumann penalties (no envelope)",
    )
    parser.add_argument(
        "--show",
        metavar="RUN",
        default=None,
        help="Render the stored Training Metrics table for a run "
        "(dir path, artifact slug, or pinn_<timestamp>) and exit",
    )
    args = parser.parse_args()

    if args.show:
        show_run(args.show)
        raise SystemExit

    manager = None
    try:
        if not args.test:
            config = HyperParams(
                huber_delta=args.huber_delta,
                n_fourier_features=args.fourier_features,
                lbfgs_steps=args.lbfgs,
                soft_bc=args.soft_bc,
            )
            if args.lr is not None:
                config = config.replace(learning_rate_max=args.lr)
            if args.weight_flux_scale is not None:
                config = config.replace(weight_flux_scale=args.weight_flux_scale)
            if args.epochs is not None:
                config = config.replace(
                    warmup_epochs=max(1, args.epochs // 6),
                    decay_epochs=args.epochs - max(1, args.epochs // 6),
                )
            manager = NetworkManager(config, test_mode=args.test, name=args.name)
            manager.train(save_to_disk=True)
        else:
            globals()["N_VALIDATION_SIZE"] = 16
            globals()["VALIDATION_FREQUENCY"] = 20
            config = HyperParams(
                huber_delta=args.huber_delta,
                soft_bc=args.soft_bc,
                lbfgs_steps=64,
                hidden_dims=(32, 32),
                batch_size=8,
                n_rz_inner_samples=64,
                n_rz_boundary_samples=16,
                n_train=64,
                warmup_epochs=20,
                decay_epochs=20,
            )
            manager = NetworkManager(config, test_mode=args.test)
            manager.train(save_to_disk=False)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Execution failed with an error: {e}", exc_info=True)
        raise
    finally:
        if manager is not None:
            manager.discard_unsaved_run()
