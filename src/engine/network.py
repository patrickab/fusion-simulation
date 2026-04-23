from datetime import datetime
import os
from pathlib import Path
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
from src.engine.plasma import calculate_poloidal_boundary, get_poloidal_points
from src.lib.config import Filepaths
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


# --- Network (simple MLP) ---
class FluxPINN(nn.Module):
    hidden_dims: tuple[int, ...]

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

        for dim in self.hidden_dims:
            x = nn.Dense(features=dim, dtype=jnp.float32)(x)
            x = nn.swish(x)

        psi_hat = nn.Dense(features=1)(x)
        return psi_hat


# --- Sampler ---
BASE_SEED = 42
RESAMPLING_FREQUENCY = 10  # Resample training set every N epochs
LOG_FREQUENCY = 10  # Log training metrics every N epochs


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
    def __init__(self, config: HyperParams, seed: int = BASE_SEED) -> None:
        self.config = config
        self.seed = seed
        self.model = FluxPINN(
            hidden_dims=config.hidden_dims,
        )
        self.sampler: Sampler = Sampler(config, seed=self.seed)
        self.state = self._init_state()

        # Pre-compile psi for efficient inference during evaluation.
        self._psi_fn_jit = jax.jit(self.make_psi_fn())

        self.train_set = self.sampler._get_sobol_sample(
            n_samples=self.config.n_train,
            lower_bounds=self.sampler._domain_lower_bounds,
            upper_bounds=self.sampler._domain_upper_bounds,
        )
        self.training_log: list[dict] = []

    def to_disk(self) -> None:
        """Save model parameters & config to disk."""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        latest_commit = os.popen("git rev-parse --short HEAD").read().strip() or "no_git"

        output_dir = Path(Filepaths.NETWORKS)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_stem = f"pinn_{timestamp}_{latest_commit}"
        artifact_flax_path = output_dir / f"{artifact_stem}.flax"
        artifact_json_path = output_dir / f"{artifact_stem}.json"
        artifact_log_path = output_dir / f"{artifact_stem}.csv"

        artifact_flax_path.write_bytes(flax.serialization.to_bytes(self.state.params))
        self.config.to_json(path=str(artifact_json_path))

        if self.training_log:
            with open(artifact_log_path, "w") as f:
                f.write("epoch,moving_avg_loss,residual,boundary\n")
                for entry in self.training_log:
                    f.write(
                        f"{entry['epoch']},{entry['moving_avg_loss']:.6f},"
                        f"{entry['residual']:.6f},{entry['boundary']:.6f}\n"
                    )

    def from_disk(self, pinn_path) -> any:  # noqa
        """Load Flax model parameters from disk."""
        with open(pinn_path, "rb") as f:
            return flax.serialization.from_bytes(self.state.params, f.read())

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
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate_max,
            warmup_steps=self.config.warmup_epochs * steps_per_epoch,
            decay_steps=self.config.decay_epochs * steps_per_epoch,
            end_value=self.config.learning_rate_min,
        )
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
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Pure function to compute physics loss. Reusable for both training and evaluation."""

        # Define psi_fn locally for JIT stability.
        # Using _make_psi_fn() would create new function objects per step, risking cache misses.
        def psi_fn(p: any, R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> jnp.ndarray:
            """Adapter converting neural network output to physical psi flux."""
            inp = FluxInput(R_sample=R, Z_sample=Z, config=cfg)
            p_n, r_n, z_n = inp.normalize()
            psi_n = apply_fn(p, r=r_n, z=z_n, **p_n)
            return (psi_n * cfg.State.F_axis * cfg.Geometry.a).squeeze()

        total, l_res, l_dir, l_per_cfg = pinn_loss_function(
            psi_fn,
            params,
            inputs.R_sample,
            inputs.Z_sample,
            inputs.config,
            weight_boundary_condition=weight_boundary_condition,
        )
        return total, (l_res, l_dir, l_per_cfg)

    @staticmethod
    @jax.jit
    def eval_step(
        state: train_state.TrainState,
        inputs: FluxInput,
        weight_boundary_condition: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Fast, gradient-free evaluation step returning all loss components."""
        total, (l_res, l_dir, l_per_cfg) = NetworkManager.compute_loss(
            state.params, state.apply_fn, inputs, weight_boundary_condition
        )
        return total, l_res, l_dir, l_per_cfg

    @staticmethod
    @jax.jit
    def train_step(
        state: train_state.TrainState,
        inputs: FluxInput,
        weight_boundary_condition: float,
    ) -> tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Perform a single training step using physics-informed gradients."""

        # Rematerializes activations during backprop. Trades compute for memory,
        # enabling training of larger networks with limited GPU memory.
        @jax.checkpoint
        def loss_wrapper(params):
            """Wrap compute loss to enable gradient computation"""
            return NetworkManager.compute_loss(
                params, state.apply_fn, inputs, weight_boundary_condition
            )

        (loss, (l_res, l_dir, l_per_cfg)), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(
            state.params
        )
        return state.apply_gradients(grads=grads), loss, l_res, l_dir, l_per_cfg

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (total_loss, residual_loss, boundary_loss).
        """
        loss, l_res, l_dir = 0.0, 0.0, 0.0
        self.sampler.precompute_coordinate_samples()

        all_losses = []

        for i in range(0, len(self.train_set), self.config.batch_size):
            train_batch = self.train_set[i : i + self.config.batch_size]
            inputs = self.sampler.sample_flux_input(plasma_configs=train_batch)
            self.state, loss, l_res, l_dir, per_config_loss = self.train_step(
                state=self.state,
                inputs=inputs,
                weight_boundary_condition=self.config.weight_boundary_condition,
            )
            all_losses.append(per_config_loss)

        if epoch % RESAMPLING_FREQUENCY == 0 and epoch > 0:
            self.train_set = self.sampler.resample_train_set(
                train_set=self.train_set,
                epoch=epoch,
                per_config_losses=all_losses,
            )

        return float(loss), float(l_res), float(l_dir)

    def _init_live_display(self) -> Live:
        """Set up Rich table, progress bar, and accumulators for live training display."""
        self._table = Table(title="Training Metrics", show_header=True, header_style="bold cyan")
        self._table.add_column("Epoch", justify="right", style="cyan")
        self._table.add_column("Loss", justify="right", style="magenta")
        self._table.add_column("Residual", justify="right")
        self._table.add_column("Boundary", justify="right")

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

        return Live(
            Panel(Group(self._table, self._progress), border_style="cyan"),
            refresh_per_second=10,
            console=console,
            vertical_overflow="visible",
        )

    def _log_metric(self, epoch: int, loss: float, residual: float, boundary: float) -> None:
        self._accumulated_loss += loss
        self._accumulated_residual += residual
        self._accumulated_boundary += boundary

        if (epoch + 1) % LOG_FREQUENCY == 0:
            moving_avg_loss = self._accumulated_loss / LOG_FREQUENCY
            moving_avg_residual = self._accumulated_residual / LOG_FREQUENCY
            moving_avg_boundary = self._accumulated_boundary / LOG_FREQUENCY

            self.training_log.append(
                {
                    "epoch": epoch + 1,
                    "moving_avg_loss": moving_avg_loss,
                    "residual": moving_avg_residual,
                    "boundary": moving_avg_boundary,
                }
            )

            self._table.add_row(
                f"{epoch + 1}/{self.epochs}",
                f"{moving_avg_loss:.3f}",
                f"{moving_avg_residual:.3f}",
                f"{moving_avg_boundary:.3f}",
            )

            self._accumulated_loss = 0.0
            self._accumulated_residual = 0.0
            self._accumulated_boundary = 0.0
        else:
            self.training_log.append(
                {
                    "epoch": epoch + 1,
                    "moving_avg_loss": self._accumulated_loss / ((epoch + 1) % LOG_FREQUENCY),
                    "residual": self._accumulated_residual / ((epoch + 1) % LOG_FREQUENCY),
                    "boundary": self._accumulated_boundary / ((epoch + 1) % LOG_FREQUENCY),
                }
            )

        self._progress.update(self._epoch_task, advance=1)
        self._live.update(Panel(Group(self._table, self._progress), border_style="cyan"))

    def train(self, save_to_disk: bool = True) -> None:
        live = self._init_live_display()
        with live as self._live:
            for epoch in range(self.epochs):
                loss, residual, boundary = self.train_epoch(epoch)
                self._log_metric(epoch, loss, residual, boundary)

        if save_to_disk:
            self.to_disk()

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
        apply_fn = self.model.apply

        def psi_fn(p: any, R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> jnp.ndarray:
            inp = FluxInput(R_sample=R, Z_sample=Z, config=cfg)
            p_n, r_n, z_n = inp.normalize()
            psi_n = apply_fn(p, r=r_n, z=z_n, **p_n)
            return (psi_n * cfg.State.F_axis * cfg.Geometry.a).squeeze()

        return psi_fn


if __name__ == "__main__":
    try:
        config = HyperParams()
        manager = NetworkManager(config)
        manager.train(save_to_disk=True)
        # params = manager.from_disk(manager.state.params)
        # manager.state = manager.state.replace(params=params)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Execution failed with an error: {e}", exc_info=True)
        raise
