from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Literal, Protocol

from flax import struct
import jax.numpy as jnp

from src.lib.config import BaseModel
from src.lib.geometry_config import PlasmaConfig


class Architecture(StrEnum):
    """Supported network architectures."""

    mlp = "mlp"
    piratenet = "piratenet"


@struct.dataclass
class HyperParams(BaseModel):
    """Central configuration for the experiment.

    Defaults resemble known best configurations.
    """

    # Optimizer & loss
    learning_rate_max: float = 2e-3
    learning_rate_min: float = 5e-5
    weight_decay: float = 1e-7
    sigma_residual_adaptive_sampling: float = 0.05
    weight_boundary_condition: float | None = 10.0  # soft-BC penalty; None/0.0 = off
    weight_flux_scale: float | None = 10.0  # collapse-guard hinge (pins psi>0); None/0.0 = off
    huber_delta: float | None = None  # None/0.0 = MSE PDE loss
    batch_size: int = 64

    # Training budget
    n_train: int = 1024
    warmup_epochs: int = 100
    decay_epochs: int = 500
    n_rz_inner_samples: int = 512
    n_rz_boundary_samples: int = 128
    lbfgs_steps: int = 0

    arch: Architecture = Architecture.mlp
    hidden_dims: tuple[int, ...] = (200, 200, 200, 200, 200)
    n_fourier_features: int = 0
    fourier_sigma: float = 2.0
    # Random Weight Factorization (Wang et al. arXiv 2210.01274)
    rwf: bool = False
    soft_bc: bool = False

    def to_json(self, path: str) -> None:
        """Write hyperparameters to disk as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str) -> "HyperParams":
        """Load hyperparameters from disk."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@struct.dataclass
class DomainBounds(BaseModel):
    """Define physical hypercube of domain for the network."""

    R0: tuple[float, float] = (5.0, 7.0)  # Major radius (m)
    a: tuple[float, float] = (3.0, 4.0)  # Minor radius (m)
    kappa: tuple[float, float] = (1.0, 2.0)  # Elongation factor
    delta: tuple[float, float] = (0.2, 0.6)  # Triangularity factor
    p0: tuple[float, float] = (1e4, 1e6)  # Central pressure (Pa)
    F_axis: tuple[float, float] = (20.0, 60.0)  # Toroidal field function at axis (T*m)

    # Exponent from 1.01 for numerical stability
    alpha: tuple[float, float] = (1.01, 2.0)  # Pressure profile shape
    exponent: tuple[float, float] = (1.01, 2.0)  # Current profile shape

    @classmethod
    def get_bounds(cls) -> tuple[jnp.ndarray, jnp.ndarray]:
        bound_names = list(DomainBounds.__dataclass_fields__.keys())
        l_bounds = jnp.array([getattr(DomainBounds, name)[0] for name in bound_names])
        u_bounds = jnp.array([getattr(DomainBounds, name)[1] for name in bound_names])
        return l_bounds, u_bounds


def min_max_scale(val: jnp.ndarray, bounds: tuple[float, float]) -> jnp.ndarray:
    """Normalize value to [-1, 1] range based on bounds."""
    min_v, max_v = bounds
    return 2.0 * (val - min_v) / (max_v - min_v) - 1.0


# --- B. Physics Containers (JAX Pytrees) ---
@struct.dataclass
class FluxInput(BaseModel):
    """Batch-first Pytree container for physics inputs."""

    R_sample: jnp.ndarray  # Shape (B, N)
    Z_sample: jnp.ndarray  # Shape (B, N)
    config: PlasmaConfig

    def normalize_plasma_params(self) -> dict[str, jnp.ndarray]:
        """Normalize plasma parameters and expand for broadcasting.

        Returns:
            Dictionary of normalized parameters expanded to (B, 1)
        """
        norm_params = {
            "r0": min_max_scale(self.config.Geometry.R0, DomainBounds.R0),
            "a": min_max_scale(self.config.Geometry.a, DomainBounds.a),
            "kappa": min_max_scale(self.config.Geometry.kappa, DomainBounds.kappa),
            "delta": min_max_scale(self.config.Geometry.delta, DomainBounds.delta),
            "p0": min_max_scale(self.config.State.p0, DomainBounds.p0),
            "f_axis": min_max_scale(self.config.State.F_axis, DomainBounds.F_axis),
            "alpha": min_max_scale(self.config.State.pressure_alpha, DomainBounds.alpha),
            "exponent": min_max_scale(self.config.State.field_exponent, DomainBounds.exponent),
        }

        # Expand each parameter to (B, 1) for broadcasting against (B, N) coordinates
        return {k: jnp.expand_dims(v, -1) for k, v in norm_params.items()}

    def normalize_coordinates(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Normalize R-Z coordinates to dimensionless units.

        Returns:
            Tuple of (normalized_R, normalized_Z)
        """
        R0_phys = jnp.expand_dims(self.config.Geometry.R0, -1)
        a_phys = jnp.expand_dims(self.config.Geometry.a, -1)

        r_n = (self.R_sample - R0_phys) / a_phys
        z_n = self.Z_sample / a_phys

        return r_n, z_n

    def normalize(self) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        """Normalize both plasma parameters and coordinates.

        Returns:
            Tuple of (normalized_params_dict, normalized_R, normalized_Z)
        """
        norm_params = self.normalize_plasma_params()
        r_n, z_n = self.normalize_coordinates()
        return norm_params, r_n, z_n

    def get_physical_scale(self) -> jnp.ndarray:
        """Denormalization factor to map network outputs to physical psi units."""
        return (jnp.atleast_1d(self.config.State.F_axis) * jnp.atleast_1d(self.config.Geometry.a))[
            :, jnp.newaxis, jnp.newaxis
        ]


# --- C. Trainer/NetworkManager seam ---
# Shared result/event types so network.py's Trainer never has to import Rich,
# Plotext, or filesystem code (owned by network_manager.py) to report progress.


@dataclass(frozen=True)
class ValidationMetrics:
    p05: float
    p50: float
    p95: float


@dataclass(frozen=True)
class EpochMetrics:
    """One epoch's training result, emitted to a TrainingObserver.

    ``epoch`` is the 0-indexed loop counter (matches Trainer.train_epoch's
    ``epoch`` argument); ``validation`` is set only on validation rounds.
    ``should_stop`` mirrors the patience decision for this epoch so an
    observer can suppress a final report right before training stops.
    """

    epoch: int
    loss: float
    residual_loss: float
    boundary_loss: float
    gradient_norm: float
    learning_rate: float
    duration_seconds: float
    validation: ValidationMetrics | None = None
    should_stop: bool = False


@dataclass(frozen=True)
class TrainingResult:
    stop_reason: Literal["epoch_budget", "early_stopping", "pruned"]
    trained_epochs: int
    planned_epochs: int
    final_validation: ValidationMetrics
    best_epoch: int | None
    best_smoothed_validation_p50: float | None
    lbfgs_validation: ValidationMetrics | None
    optimizer_updates: int
    duration_seconds: float


class TrainingObserver(Protocol):
    """Return ``True`` to request an early stop for a non-patience reason (e.g. pruning)."""

    def __call__(self, event: EpochMetrics) -> bool | None: ...
