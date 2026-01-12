from flax import linen as nn
from flax import struct
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from scipy.stats import qmc

from engine.plasma import get_poloidal_points
from src.lib.geometry_config import (
    PlasmaGeometry,
    PlasmaState,
)


# --- A. Configuration ---
@struct.dataclass
class HyperParams:
    """Central configuration for the experiment."""

    input_dim: int = 10  # 2 (RZ) + 8 (Params)
    output_dim: int = 1
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    learning_rate_max: float = 1e-3
    learning_rate_min: float = 1e-5
    batch_size: int = 64
    n_rz_samples: int = 2048
    n_train: int = 256
    n_test: int = 64
    n_val: int = 128
    warmup_steps: int = 500
    decay_steps: int = 10000


@struct.dataclass
class DomainBounds:
    """Define physical hypercube of domain for normalization and sampling."""

    R0: tuple[float, float] = (1.0, 8.0)  # Major radius (m)
    a: tuple[float, float] = (0.3, 3.0)  # Minor radius (m)
    kappa: tuple[float, float] = (1.0, 2.0)  # Elongation factor
    delta: tuple[float, float] = (0.0, 0.6)  # Triangularity factor
    p0: tuple[float, float] = (1e4, 1e6)  # Central pressure (Pa)
    F_axis: tuple[float, float] = (1.0, 50.0)  # Toroidal field function at axis (T*m)
    alpha: tuple[float, float] = (0.5, 3.0)  # Pressure profile shape
    exponent: tuple[float, float] = (0.5, 3.0)  # Current profile shape


def min_max_scale(val: jnp.ndarray, bounds: tuple[float, float]) -> jnp.ndarray:
    min_v, max_v = bounds
    return 2.0 * (val - min_v) / (max_v - min_v) - 1.0


# --- B. Physics Containers (JAX Pytrees) ---
@struct.dataclass
class FluxInput:
    """Batch-first Pytree container for physics inputs."""

    R: jnp.ndarray  # Shape (B, N) or (N,)
    Z: jnp.ndarray  # Shape (B, N) or (N,)
    geometry: PlasmaGeometry
    state: PlasmaState

    def normalize(self) -> dict[str, jnp.ndarray]:
        """
        Dual-Space Normalization
        Physics preserving transformation using min-max scaling.
        Returns a dictionary for **kwargs unpacking into FluxPINN.
        """
        R = jnp.atleast_2d(self.R)
        Z = jnp.atleast_2d(self.Z)

        def norm_param(val: jnp.ndarray, bounds: tuple[float, float]) -> jnp.ndarray:
            v = jnp.atleast_1d(val)
            v_norm = min_max_scale(v, bounds)
            return v_norm[:, jnp.newaxis]

        # Use geometry parameters for coordinate normalization (Physics scale)
        R0_phys = jnp.atleast_1d(self.geometry.R0)[:, jnp.newaxis]
        a_phys = jnp.atleast_1d(self.geometry.a)[:, jnp.newaxis]

        r_norm = (R - R0_phys) / a_phys
        z_norm = Z / a_phys

        return {
            "r": r_norm,
            "z": z_norm,
            "r0": norm_param(self.geometry.R0, DomainBounds.R0),
            "a": norm_param(self.geometry.a, DomainBounds.a),
            "kappa": norm_param(self.geometry.kappa, DomainBounds.kappa),
            "delta": norm_param(self.geometry.delta, DomainBounds.delta),
            "p0": norm_param(self.state.p0, DomainBounds.p0),
            "f_axis": norm_param(self.state.F_axis, DomainBounds.F_axis),
            "alpha": norm_param(self.state.pressure_alpha, DomainBounds.alpha),
            "exponent": norm_param(self.state.field_exponent, DomainBounds.exponent),
        }

    def get_physical_scale(self) -> jnp.ndarray:
        """Denormalization factor to map network outputs to physical psi units."""
        return (jnp.atleast_1d(self.state.F_axis) * jnp.atleast_1d(self.geometry.a))[
            :, jnp.newaxis, jnp.newaxis
        ]


# --- C. The Neural Network ---
class FluxPINN(nn.Module):
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(
        self,
        r: jnp.ndarray,
        z: jnp.ndarray,
        r0: jnp.ndarray,
        a: jnp.ndarray,
        kappa: jnp.ndarray,
        delta: jnp.ndarray,
        p0: jnp.ndarray,
        f_axis: jnp.ndarray,
        alpha: jnp.ndarray,
        exponent: jnp.ndarray,
    ) -> jnp.ndarray:
        # Broadcast all inputs to match coordinate shape (B, N)
        target_shape = r.shape
        params = [r, z, r0, a, kappa, delta, p0, f_axis, alpha, exponent]
        x = jnp.stack([jnp.broadcast_to(p, target_shape) for p in params], axis=-1)

        for dim in self.hidden_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.tanh(x)

        psi_hat = nn.Dense(features=1)(x)
        return psi_hat


# --- D. The Sampler ---
BASE_SEED = 42


class Sampler:
    def __init__(self, config: HyperParams) -> None:
        self.config = config

    def _build_domain_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        bound_names = ("R0", "a", "kappa", "delta", "p0", "F_axis", "alpha", "exponent")
        l_bounds = jnp.array([getattr(DomainBounds, name)[0] for name in bound_names])
        u_bounds = jnp.array([getattr(DomainBounds, name)[1] for name in bound_names])
        return l_bounds, u_bounds

    def _get_sobol_sample(
        self,
        n_samples: int,
        seed: int,
        lower_bounds: jnp.ndarray,
        upper_bounds: jnp.ndarray,
    ) -> jnp.ndarray:
        sampler = qmc.Sobol(d=len(lower_bounds), scramble=True, seed=seed)
        sample_unit = jnp.array(sampler.random(n_samples))
        return sample_unit * (upper_bounds - lower_bounds) + lower_bounds

    def sample_flux_input(
        self,
        seed: int,
        n_samples: int,
        plasma_configs: jnp.ndarray,
    ) -> FluxInput:
        """
        Interface for sampling points specifically on the boundary of the plasma.
        Requirements:
        1. Sample 'theta' via Sobol in [0, 2pi].
        2. Use 'plasma_boundary' logic to convert (theta, params) -> (R_wall, Z_wall).
        """
        # Sample theta and rho (scaling factor) via Sobol
        sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
        samples = jnp.array(sampler.random(n_samples))
        theta = samples[:, 0] * 2 * jnp.pi
        rho = samples[:, 1]

        # Extract parameters for batch calculation
        R0 = plasma_configs[:, 0]
        a = plasma_configs[:, 1]
        kappa = plasma_configs[:, 2]
        delta = plasma_configs[:, 3]

        # Calculate R, Z coordinates inside the boundary using vmap
        def compute_interior(
            r0_val: jnp.ndarray,
            a_val: jnp.ndarray,
            k_val: jnp.ndarray,
            d_val: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            geom = PlasmaGeometry(R0=r0_val, a=a_val, kappa=k_val, delta=d_val)
            # vmap over both theta and rho to get interior points
            r_pts, z_pts = jax.vmap(lambda t, s: get_poloidal_points(t, geom, s))(theta, rho)
            return r_pts, z_pts

        R, Z = jax.vmap(compute_interior)(R0, a, kappa, delta)

        return FluxInput(
            R=R,
            Z=Z,
            geometry=PlasmaGeometry(R0=R0, a=a, kappa=kappa, delta=delta),
            state=PlasmaState(
                p0=plasma_configs[:, 4],
                F_axis=plasma_configs[:, 5],
                pressure_alpha=plasma_configs[:, 6],
                field_exponent=plasma_configs[:, 7],
            ),
        )


# --- E. The Trainer ---
class PINNTrainer:
    def __init__(self, config: HyperParams) -> None:
        self.config = config
        self.model = FluxPINN(hidden_dims=config.hidden_dims)
        self.sampler: Sampler = Sampler(config)
        self.state = self._init_state()

        lower_bounds, upper_bounds = self.sampler._build_domain_bounds()
        self.train_set = self.sampler._get_sobol_sample(
            n_samples=config.n_train,
            seed=BASE_SEED,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def _init_state(self) -> train_state.TrainState:
        key = jax.random.PRNGKey(BASE_SEED)
        d_rz = jnp.ones((1, self.config.n_rz_samples))
        d_p = jnp.ones(1)

        dummy_input = FluxInput(
            R=d_rz,
            Z=d_rz,
            geometry=PlasmaGeometry(R0=d_p, a=d_p, kappa=d_p, delta=d_p),
            state=PlasmaState(p0=d_p, F_axis=d_p, pressure_alpha=d_p, field_exponent=d_p),
        )
        params = self.model.init(key, **dummy_input.normalize())

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate_max,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.decay_steps,
            end_value=self.config.learning_rate_min,
        )
        tx = optax.adam(learning_rate=schedule)
        return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    @staticmethod
    @jax.jit
    def train_step(
        state: train_state.TrainState,
        inputs: FluxInput,
    ) -> tuple[train_state.TrainState, jnp.ndarray]:
        def loss_fn(params: dict) -> jnp.ndarray:
            # Single forward pass on the batched tensor_input
            psi_norm = state.apply_fn(params, **inputs.normalize())
            # Map to physical scale for physics loss evaluation
            psi_phys = psi_norm * inputs.get_physical_scale()
            #
            # Placeholder for actual physics loss integration
            return 0.0

        loss, grads = ...  # get loss and gradients
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def train(self, epochs: int) -> None:
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            for i in range(0, len(self.train_set), self.config.batch_size):
                train_batch = self.train_set[i : i + self.config.batch_size]
                # Sample RZ points for each plasma configuration in the batch
                inputs = self.sampler.sample_flux_input(
                    seed=epoch + i, n_samples=self.config.n_rz_samples, plasma_configs=train_batch
                )
                self.state, loss = self.train_step(state=self.state, inputs=inputs)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

    def predict(self, inputs: FluxInput) -> jnp.ndarray:
        psi_norm = self.model.apply(self.state.params, **inputs.normalize())
        return psi_norm * inputs.get_physical_scale()
