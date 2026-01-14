from flax import struct
import jax.numpy as jnp

from src.lib.config import BaseModel
from src.lib.geometry_config import PlasmaConfig

BATCH_SIZE = 128
N_TRAIN = 1024


@struct.dataclass
class HyperParams(BaseModel):
    """Central configuration for the experiment."""

    input_dim: int = 10  # 2 (RZ) + 8 (Params)
    output_dim: int = 1
    hidden_dims: tuple[int, ...] = (128, 128, 128, 128)
    learning_rate_max: float = 2e-3
    learning_rate_min: float = 2e-5
    batch_size: int = BATCH_SIZE
    n_rz_inner_samples: int = 2048
    n_rz_boundary_samples: int = 256
    n_train: int = N_TRAIN
    n_test: int = 32
    n_val: int = 64
    warmup_steps: int = 100 * (N_TRAIN // BATCH_SIZE)
    decay_steps: int = 600 * (N_TRAIN // BATCH_SIZE)


@struct.dataclass
class DomainBounds(BaseModel):
    """Define physical hypercube of domain for the network."""

    R0: tuple[float, float] = (1.0, 8.0)  # Major radius (m)
    a: tuple[float, float] = (0.3, 3.0)  # Minor radius (m)
    kappa: tuple[float, float] = (1.0, 2.0)  # Elongation factor
    delta: tuple[float, float] = (0.0, 0.6)  # Triangularity factor
    p0: tuple[float, float] = (1e4, 1e6)  # Central pressure (Pa)
    F_axis: tuple[float, float] = (1.0, 50.0)  # Toroidal field function at axis (T*m)

    # Exponent from 1.01 for numerical stability
    alpha: tuple[float, float] = (1.01, 2.0)  # Pressure profile shape
    exponent: tuple[float, float] = (1.01, 2.0)  # Current profile shape


def min_max_scale(val: jnp.ndarray, bounds: tuple[float, float]) -> jnp.ndarray:
    """Normalize value to [-1, 1] range based on bounds."""
    min_v, max_v = bounds
    return 2.0 * (val - min_v) / (max_v - min_v) - 1.0


def normalize_plasma_config(config: PlasmaConfig) -> dict[str, jnp.ndarray]:
    """Normalize plasma parameters for network input."""
    return {
        "r0": min_max_scale(config.Geometry.R0, DomainBounds.R0),
        "a": min_max_scale(config.Geometry.a, DomainBounds.a),
        "kappa": min_max_scale(config.Geometry.kappa, DomainBounds.kappa),
        "delta": min_max_scale(config.Geometry.delta, DomainBounds.delta),
        "p0": min_max_scale(config.State.p0, DomainBounds.p0),
        "f_axis": min_max_scale(config.State.F_axis, DomainBounds.F_axis),
        "alpha": min_max_scale(config.State.pressure_alpha, DomainBounds.alpha),
        "exponent": min_max_scale(config.State.field_exponent, DomainBounds.exponent),
    }


# --- B. Physics Containers (JAX Pytrees) ---
@struct.dataclass
class FluxInput(BaseModel):
    """Batch-first Pytree container for physics inputs."""

    R_sample: jnp.ndarray  # Shape (B, N)
    Z_sample: jnp.ndarray  # Shape (B, N)
    config: PlasmaConfig

    def get_norm_params(self) -> dict[str, jnp.ndarray]:
        """Normalize plasma parameters for network input."""
        normed = normalize_plasma_config(self.config)
        return {k: jnp.atleast_1d(v)[:, jnp.newaxis] for k, v in normed.items()}

    def normalize_coords(self, R: jnp.ndarray, Z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Map physical (R, Z) coordinates to normalized (r, z) space."""
        R0_phys = jnp.atleast_1d(self.config.Geometry.R0)[:, jnp.newaxis]
        a_phys = jnp.atleast_1d(self.config.Geometry.a)[:, jnp.newaxis]
        return (R - R0_phys) / a_phys, Z / a_phys

    def get_physical_scale(self) -> jnp.ndarray:
        """Denormalization factor to map network outputs to physical psi units."""
        return (jnp.atleast_1d(self.config.State.F_axis) * jnp.atleast_1d(self.config.Geometry.a))[
            :, jnp.newaxis, jnp.newaxis
        ]
