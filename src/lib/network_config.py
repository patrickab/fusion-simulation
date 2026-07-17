import json
from pathlib import Path

from flax import struct
import jax.numpy as jnp

from src.lib.config import BaseModel
from src.lib.geometry_config import PlasmaConfig


@struct.dataclass
class HyperParams(BaseModel):
    """Central configuration for the experiment."""

    hidden_dims: tuple[int, ...] = (128, 128, 128, 128)
    # 0.0 → MSE PDE loss; >0 → optax Huber loss with this delta. Ablation 1
    # (kinked LUT boundary) had Huber winning clearly; grid 2 (smooth Fourier
    # boundary envelope) closed the gap — mse core_med 0.388 vs huber 0.420
    # plain, huber 0.383 vs mse 0.502 with nff=64 — confirming the kinked
    # boundary, not the loss, was ablation 1's real culprit. Kept as a toggle
    # since the two losses are now close enough to depend on architecture.
    huber_delta: float = 1.0
    # random Fourier features on (r, z); 0 = plain MLP. nff=64 gave the best
    # core median in grid 2 (2026-07-11) at the cost of a noisier boundary
    # shell (edge_p95/bnd_leak both ~2-3x higher) — tolerated per the
    # core-first selection rule, and is the CLI default for new training runs
    # (src/engine/network.py --fourier-features). The dataclass default must
    # stay 0 though: any saved checkpoint whose .json predates this field (or
    # any HyperParams() built without one) falls back to this value, and a
    # mismatched nff makes flax reject the checkpoint's params shape outright.
    n_fourier_features: int = 0
    fourier_sigma: float = 2.0
    # L-BFGS polish steps on a fixed batch after AdamW; 0 = off. Grid 2 showed
    # no consistent gain (made plain-huber worse: 0.460 vs 0.420) for
    # 1.5-8min/run extra cost — not worth defaulting on.
    lbfgs_steps: int = 0
    learning_rate_max: float = 2e-3
    learning_rate_min: float = 5e-5
    weight_decay: float = 1e-7
    weight_boundary_condition: float = 10.0
    # Collapse-guard hinge: penalizes interior-mean ψ below 0.05·F_axis·a (also
    # pins the ψ>0-at-axis sign convention); zero loss once above the floor.
    weight_flux_scale: float = 10.0
    # Train like legacy bb503b0: raw ψ output + Dirichlet/Neumann penalties (no envelope).
    soft_bc: bool = False
    # Random Weight Factorization (Wang et al. arXiv 2210.01274): reparametrize each
    # dense kernel as W = diag(exp(s)) · V to improve PINN accuracy. Default must stay
    # False for checkpoint compat — enabling it changes the params tree (same constraint
    # as n_fourier_features: old config.json files without this field deserialize to False,
    # i.e. plain Dense, and saved params remain loadable).
    rwf: bool = False
    # Network architecture: "mlp" = plain MLP (default, existing behaviour), "piratenet" =
    # PirateNet residual blocks (arXiv 2402.00326, eq. 4.1-4.7). Default must stay "mlp"
    # for checkpoint compat — "piratenet" changes the params tree (same constraint as
    # n_fourier_features/rwf: old config.json without this field deserializes to "mlp").
    arch: str = "mlp"
    sigma_residual_adaptive_sampling: float = 0.05
    batch_size: int = 64
    n_rz_inner_samples: int = 512
    n_rz_boundary_samples: int = 128
    n_train: int = 1024
    warmup_epochs: int = 100
    decay_epochs: int = 500

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
