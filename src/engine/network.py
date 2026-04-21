from datetime import datetime
import os
from pathlib import Path
from time import time
from typing import Callable, Literal

from flax import linen as nn
import flax.serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
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
        params = [r, z, r0, a, kappa, delta, p0, f_axis, alpha, exponent]
        x = jnp.stack([jnp.broadcast_to(p, target_shape) for p in params], axis=-1)

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
    def __init__(self, config: HyperParams, seed: int = BASE_SEED) -> None:
        self.config = config
        self.seed = seed
        self._domain_lower_bounds, self._domain_upper_bounds = self._build_domain_bounds()

        # Instatiate separate Sobol samplers for domain, interior points, and boundary points.
        # Avoids repetitive instatiation of samplers.
        self._sobol_domain = qmc.Sobol(
            d=len(self._domain_lower_bounds),
            scramble=True,
            seed=self.seed,
        )
        self._sobol_inner = qmc.Sobol(d=2, scramble=True, seed=self.seed + 1)
        self._sobol_boundary = qmc.Sobol(d=1, scramble=True, seed=self.seed + 2)

        # Pre-computed per-epoch coordinate samples.
        self._theta_int: jnp.ndarray | None = None
        self._rho_int: jnp.ndarray | None = None
        self._theta_b: jnp.ndarray | None = None

        self.precompute_coordinate_samples()

    def _build_domain_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Create lower and upper bounds arrays for parameter sampling."""
        # Get all field names from DomainBounds dataclass
        bound_names = list(DomainBounds.__dataclass_fields__.keys())
        l_bounds = jnp.array([getattr(DomainBounds, name)[0] for name in bound_names])
        u_bounds = jnp.array([getattr(DomainBounds, name)[1] for name in bound_names])
        return l_bounds, u_bounds

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

    def sample_flux_input(
        self,
        plasma_configs: jnp.ndarray,
    ) -> FluxInput:
        """Sample interior and boundary points for a batch of plasma configurations.

        Creates one set of coordinates, then casts them for each plasma config.
        More efficient than sampling separate coordinates per config.
        Overfitting avoided by resampling coordinates each epoch.
        """
        if self._theta_int is None or self._rho_int is None or self._theta_b is None:
            self.precompute_coordinate_samples()

        theta_int = self._theta_int
        rho_int = self._rho_int
        theta_b = self._theta_b

        # Define single case
        def compute_single_config(
            plasma_config: jnp.ndarray,
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

        # Vectorize, compile & execute over batch of plasma configs
        configs, R_int, Z_int = jax.jit(jax.vmap(compute_single_config))(plasma_configs)

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

    def to_disk(self) -> None:
        """Save model parameters & config to disk."""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        latest_commit = os.popen("git rev-parse --short HEAD").read().strip() or "no_git"

        output_dir = Path(Filepaths.NETWORKS)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_stem = f"pinn_{timestamp}_{latest_commit}"
        artifact_flax_path = output_dir / f"{artifact_stem}.flax"
        artifact_json_path = output_dir / f"{artifact_stem}.json"

        artifact_flax_path.write_bytes(flax.serialization.to_bytes(self.state.params))
        self.config.to_json(path=str(artifact_json_path))

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
    @jax.jit
    def train_step(
        state: train_state.TrainState,
        inputs: FluxInput,
        weight_boundary_condition: float,
    ) -> tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Perform a single training step using physics-informed gradients.

        Returns:
            Tuple of (updated_state, total_loss, residual_loss, boundary_loss, per_config_loss).
        """

        # Define psi_fn locally for JIT stability.
        # Using _make_psi_fn() would create new function objects per step, risking cache misses.
        def psi_fn(params: any, R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> jnp.ndarray:
            """Adapter converting neural network output to physical psi flux."""
            inp = FluxInput(R_sample=R, Z_sample=Z, config=cfg)
            p_n, r_n, z_n = inp.normalize()
            psi_n = state.apply_fn(params, r=r_n, z=z_n, **p_n)
            return (psi_n * cfg.State.F_axis * cfg.Geometry.a).squeeze()

        # @jax.checkpoint
        # Rematerializes activations during backprop. Trades compute for memory,
        # enabling training of larger networks with limited GPU memory.
        def loss_fn(
            params: any,
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            total, l_res, l_dir, l_per_cfg = pinn_loss_function(
                psi_fn,  # Passes callable directly
                params,
                inputs.R_sample,
                inputs.Z_sample,
                inputs.config,
                weight_boundary_condition=weight_boundary_condition,
            )
            return total, (l_res, l_dir, l_per_cfg)

        (loss, (l_res, l_dir, l_per_cfg)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        return state.apply_gradients(grads=grads), loss, l_res, l_dir, l_per_cfg

    def calculate_loss(self, inputs: FluxInput) -> float:
        """Calculate loss for given inputs without updating state."""
        _, loss, _, _, _ = self.train_step(
            self.state,
            inputs,
            self.config.weight_boundary_condition,
        )
        return float(loss)

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

        # Periodic dataset resampling
        if epoch % RESAMPLING_FREQUENCY == 0 and epoch > 0:
            n_sobol = self.config.n_train // 2
            n_adaptive = self.config.n_train - n_sobol

            sobol_samples = self.sampler._get_sobol_sample(
                n_samples=n_sobol,
                lower_bounds=self.sampler._domain_lower_bounds,
                upper_bounds=self.sampler._domain_upper_bounds,
            )

            all_losses_jnp = jnp.concatenate(all_losses)
            top_k_indices = jnp.argsort(all_losses_jnp)[-n_adaptive:]
            top_k_configs = self.train_set[top_k_indices]

            bounds_range = self.sampler._domain_upper_bounds - self.sampler._domain_lower_bounds
            key = jax.random.PRNGKey(self.seed + epoch)
            noise = jax.random.normal(key, top_k_configs.shape) * (
                self.config.sigma_residual_adaptive_sampling * bounds_range
            )

            adaptive_samples = jnp.clip(
                top_k_configs + noise,
                a_min=self.sampler._domain_lower_bounds,
                a_max=self.sampler._domain_upper_bounds,
            )

            self.train_set = jnp.concatenate([sobol_samples, adaptive_samples], axis=0)

        return float(loss), float(l_res), float(l_dir)

    def train(
        self,
        save_to_disk: bool = True,
    ) -> float:
        """
        Train the model.

        Args:
            save_to_disk: Whether to save model params to disk after training

        Returns:
            Final training loss
        """
        logger.info(f"Starting training for {self.epochs} epochs...")
        start_time = time()  # Initialize timer
        for epoch in range(self.epochs):
            loss_total, l_residual, l_boundary = self.train_epoch(epoch)
            if epoch % LOG_FREQUENCY == 0 and epoch > 0:
                elapsed = time() - start_time
                avg_speed = elapsed / LOG_FREQUENCY
                logger.info(
                    f"Epoch: {epoch:4d} | Loss: [bold magenta]{loss_total:6.3f}[/bold magenta] | "
                    f"(residual= {l_residual:5.2f}, boundary= {l_boundary:2.2f}) | "
                    f"sec/epoch: {avg_speed:.2f}"
                )
                start_time = time()  # Reset timer

        if save_to_disk:
            self.to_disk()

    def predict(self, inputs: FluxInput) -> jnp.ndarray:
        """Generate predictions for given inputs."""
        norm_params, r_n, z_n = inputs.normalize()
        psi_norm = self.model.apply(self.state.params, r=r_n, z=z_n, **norm_params)
        return psi_norm * inputs.get_physical_scale()

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
