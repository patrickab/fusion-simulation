from collections import deque
from dataclasses import dataclass
import functools
from functools import partial
import time
from typing import Callable, Literal

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from scipy.stats import qmc

from src.engine.physics import pinn_loss_function
from src.engine.plasma import (
    boundary_normalized_radius,
    calculate_poloidal_boundary,
    get_poloidal_points,
)
from src.lib.config import KPI_EVAL_CONFIGS, KPI_POINTS_PER_CONFIG
from src.lib.geometry_config import (
    PlasmaConfig,
    PlasmaGeometry,
    PlasmaState,
)
from src.lib.logger import get_logger
from src.lib.network_config import (
    DomainBounds,
    EpochMetrics,
    FluxInput,
    HyperParams,
    TrainingObserver,
    TrainingResult,
    ValidationMetrics,
)

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


# --- Random Weight Factorization dense layer ---
class RWFDense(nn.Module):
    """Dense layer with Random Weight Factorization (Wang et al. arXiv 2210.01274).

    Reparametrizes the kernel as W = V * exp(s) where s ~ N(1.0, 0.1) and
    V is initialized so that the effective kernel V * exp(s) equals a standard
    glorot_normal draw at init. Both s and V are stored as a single structured
    param ("w_fact") so their initialization is coupled via one PRNG key.

    See also: "An Expert's Guide to Training PINNs" (arXiv 2308.08468) §4.3.
    """

    features: int
    dtype: jnp.dtype = jnp.float32

    _mu: float = 1.0
    _sigma: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]

        def w_fact_init(
            key: jax.Array,
            shape: tuple[int, ...],
        ) -> dict[str, jnp.ndarray]:
            # shape is (in_features, features) — used only to derive the two
            # sub-shapes; both sub-arrays are drawn from splits of `key`.
            in_f, out_f = shape
            key_w, key_s = jax.random.split(key)
            w0 = nn.initializers.glorot_normal()(key_w, (in_f, out_f), self.dtype)
            s = self._mu + self._sigma * jax.random.normal(key_s, (out_f,), self.dtype)
            v = w0 / jnp.exp(s)
            return {"s": s, "v": v}

        w = self.param("w_fact", w_fact_init, (in_features, self.features))
        kernel = w["v"] * jnp.exp(w["s"])  # effective kernel = glorot draw at init
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        return (x @ kernel + bias).astype(self.dtype)


# --- Network (simple MLP) ---
class FluxPINN(nn.Module):
    hidden_dims: tuple[int, ...]
    # Random Fourier features on the spatial (r, z) inputs to counter spectral
    # bias (Wang et al. 2021). 0 disables; the projection matrix is drawn from
    # a fixed PRNG key so train and reload see the identical embedding.
    n_fourier_features: int = 0
    fourier_sigma: float = 2.0
    # Random Weight Factorization (Wang et al. arXiv 2210.01274). Default False
    # for checkpoint compat — enabling changes the params tree.
    rwf: bool = False
    # Network architecture: "mlp" = plain MLP (default), "piratenet" = PirateNet
    # residual blocks (arXiv 2402.00326, eq. 4.1-4.7). Default "mlp" for
    # checkpoint compat — "piratenet" changes the params tree.
    arch: str = "mlp"

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
        dense = (
            (lambda f: RWFDense(features=f, dtype=jnp.float32))
            if self.rwf
            else (lambda f: nn.Dense(features=f, dtype=jnp.float32))
        )

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

        if self.arch == "piratenet":
            # PirateNet (arXiv 2402.00326, eq. 4.1-4.7).
            # Two encoder branches + gated residual blocks with learnable skip weights.
            # All alpha_l initialise to 0 so every block is a pure linear projection of
            # the input embedding at init — the paper's core fix for PINN initialisation.
            width = self.hidden_dims[0]
            n_blocks = len(self.hidden_dims)

            u = nn.swish(dense(width)(x))  # eq. 4.1
            v = nn.swish(dense(width)(x))  # eq. 4.2
            h = dense(width)(x)  # linear projection; no activation (eq. 4.3, dim-match residual)

            for l in range(n_blocks):  # noqa: E741
                f = nn.swish(dense(width)(h))  # eq. 4.4a
                z1 = f * u + (1.0 - f) * v  # eq. 4.4b
                g = nn.swish(dense(width)(z1))  # eq. 4.5a
                z2 = g * u + (1.0 - g) * v  # eq. 4.5b
                q = nn.swish(dense(width)(z2))  # eq. 4.6
                # Scalar skip weight: alpha=0 at init → identity through projected embedding
                alpha = self.param(f"alpha_{l}", nn.initializers.zeros, ())
                h = alpha * q + (1.0 - alpha) * h  # eq. 4.7

            psi_hat = dense(1)(h)
        else:
            for dim in self.hidden_dims:
                x = dense(dim)(x)
                x = nn.swish(x)

            psi_hat = dense(1)(x)

        return psi_hat


# --- Sampler ---
BASE_SEED = 42
RESAMPLING_FREQUENCY = 10  # Resample training set every N epochs
N_VALIDATION_SIZE = KPI_EVAL_CONFIGS  # Number of validation plasma configs
VALIDATION_FREQUENCY = 50  # Evaluate validation set every N epochs
# Early stopping is always on. If it fires prematurely, the run is misconfigured
# (LR band too high/low, oversized budget); fix that or adjust the improvement
# threshold below — do not add a bypass flag.
EARLY_STOPPING_PATIENCE = 6  # Stop after this many non-improving validation rounds
EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT = 0.01
EARLY_STOPPING_ROLLING_WINDOW = 3


class _Patience:
    """Track meaningful improvements in a rolling-average validation metric."""

    def __init__(self, patience: int, min_relative_improvement: float, window: int) -> None:
        if patience < 1 or window < 1 or min_relative_improvement < 0:
            raise ValueError(
                "Early-stopping patience/window must be positive and min delta non-negative"
            )
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.values: deque[float] = deque(maxlen=window)
        self.best_value = float("inf")
        self.best_epoch: int | None = None
        self.rounds_without_improvement = 0

    def update(self, epoch: int, value: float) -> tuple[bool, bool]:
        self.values.append(value)
        if len(self.values) < self.values.maxlen:
            return False, False

        rolling_average = sum(self.values) / len(self.values)
        improved = rolling_average <= self.best_value * (1.0 - self.min_relative_improvement)
        if improved:
            self.best_value = rolling_average
            self.best_epoch = epoch
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
        return self.rounds_without_improvement >= self.patience, improved


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


# ---------------------------------------------------------------------------
# FoundationModel — frozen checkpoint used as a constant prior
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoundationModel:
    """A converged FluxPINN bound to a specific frozen checkpoint.

    Its params are a constant prior — never part of the trainable pytree.
    Used by the multistage corrector: stage-1 is wrapped here so jax.grad
    differentiates only stage-2 params while R/Z derivatives still flow through
    the sum (arXiv 2407.17213 / 2507.16636).
    """

    model: FluxPINN
    params: any

    @property
    def apply_fn(self) -> Callable:
        return self.model.apply


# ---------------------------------------------------------------------------
# _Field — owns the psi-function (single net or composed with a frozen prior)
# ---------------------------------------------------------------------------


class _Field:
    """Builds and owns the psi-function for a Trainer.

    For a plain network: ``psi_fn(params, R, Z, cfg) = apply_psi_fn(head, params, ...)``.
    For a corrector: ``psi_fn(params, R, Z, cfg) = psi_stage1(frozen) + scale *
    psi_stage2(params)``.
    The prior's params are closed over as constants so jax.grad on the corrector
    differentiates only stage-2 params.
    """

    def __init__(
        self,
        head_apply_fn: Callable,
        *,
        soft_bc: bool,
        prior: FoundationModel | None = None,
        scale: float = 1.0,
    ) -> None:
        self._head_apply_fn = head_apply_fn
        self._soft_bc = soft_bc
        self._prior = prior
        self._scale = scale

    @property
    def is_corrector(self) -> bool:
        return self._prior is not None

    def make_psi_fn(
        self,
    ) -> Callable[[any, jnp.ndarray, jnp.ndarray, PlasmaConfig], jnp.ndarray]:
        """Return a ``(params, R, Z, cfg) -> psi`` callable for this field."""
        soft_bc = self._soft_bc
        head_apply_fn = self._head_apply_fn
        if self._prior is None:
            return functools.partial(apply_psi_fn, head_apply_fn, soft_bc=soft_bc)

        prior_apply_fn = self._prior.apply_fn
        prior_params = self._prior.params  # closed over as constant
        scale = self._scale

        def psi_fn(params: any, R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> jnp.ndarray:
            psi1 = apply_psi_fn(prior_apply_fn, prior_params, R, Z, cfg, soft_bc=soft_bc)
            psi2 = apply_psi_fn(head_apply_fn, params, R, Z, cfg, soft_bc=soft_bc)
            return psi1 + scale * psi2

        return psi_fn


# ---------------------------------------------------------------------------
# Trainer — model, sampler, optimizer/JIT state, and the training loop
# ---------------------------------------------------------------------------


class Trainer:
    """Owns the FluxPINN, Sampler, optimizer/JIT state, and the training loop.

    Reports progress through an optional ``TrainingObserver`` callback instead
    of touching Rich, Plotext, or the filesystem — those live in
    ``network_manager.NetworkManager``, which composes a Trainer and subscribes
    to its events.
    """

    def __init__(
        self,
        config: HyperParams,
        seed: int = BASE_SEED,
        n_validation_size: int = N_VALIDATION_SIZE,
        *,
        prior: FoundationModel | None = None,
        scale: float = 1.0,
    ) -> None:
        self.config = config
        self.seed = seed
        self.n_validation_size = n_validation_size
        self.model = FluxPINN(
            hidden_dims=config.hidden_dims,
            n_fourier_features=config.n_fourier_features,
            fourier_sigma=config.fourier_sigma,
            rwf=config.rwf,
            arch=config.arch,
        )
        self.sampler: Sampler = Sampler(config, seed=self.seed)
        self._validation_kpi_configs: list | None = None
        self.state = self._init_state()

        # _Field owns the psi-function; for a corrector it closes over prior.params as a constant.
        self._field = _Field(
            self.model.apply,
            soft_bc=config.soft_bc,
            prior=prior,
            scale=scale,
        )
        self._psi_fn = self._field.make_psi_fn()
        self._psi_fn_jit = jax.jit(self._psi_fn)

        self._prior = prior
        self._scale = scale
        self._lbfgs_params: any = None

        self.train_set = self.sampler._get_sobol_sample(
            n_samples=self.config.n_train,
            lower_bounds=self.sampler._domain_lower_bounds,
            upper_bounds=self.sampler._domain_upper_bounds,
        )

        # Instance-level jitted train_step — closed over self._psi_fn so both the plain
        # and composed cases use one kernel.  soft_bc must stay static (pinn_loss_function
        # declares it static_argnames) so it is declared static here too.
        self._train_step_jit = jax.jit(self._train_step, static_argnames=("soft_bc",))

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

    @property
    def lbfgs_params(self) -> any:
        """Polished params from the last ``lbfgs()`` call, or None if not run."""
        return self._lbfgs_params

    @staticmethod
    def compute_loss(
        params: any,
        psi_fn: Callable,
        inputs: FluxInput,
        weight_boundary_condition: float,
        huber_delta: float,
        weight_flux_scale: float,
        soft_bc: bool,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Pure function to compute physics loss given an explicit psi_fn.

        ``psi_fn`` is ``(params, R, Z, cfg) -> psi``.  For a plain network pass
        ``functools.partial(apply_psi_fn, apply_fn, soft_bc=soft_bc)``; for a
        corrector pass the composed fn from ``_Field.make_psi_fn()``.
        """
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
        psi_fn = functools.partial(apply_psi_fn, state.apply_fn, soft_bc=soft_bc)
        total, (l_res, l_dir, l_per_cfg) = Trainer.compute_loss(
            state.params,
            psi_fn,
            inputs,
            weight_boundary_condition,
            huber_delta,
            weight_flux_scale,
            soft_bc,
        )
        return total, l_res, l_dir, l_per_cfg

    def validation_configs(self) -> list:
        """The trainer's fixed validation configs: the first n_validation_size of the
        shared KPI config stream (kpi_benchmark_configs), so training-time tracking,
        HPO ranking and run.json KPIs all score identical PlasmaConfig objects.

        PlasmaConfig construction bakes the building sampler's boundary-theta draw
        into the Fourier fit, so configs must come from ``self.sampler`` — a
        separately-seeded sampler yields subtly different boundaries for the same
        domain vectors.
        """
        from src.engine.model_evaluation import kpi_benchmark_configs

        if self._validation_kpi_configs is None:
            self._validation_kpi_configs = kpi_benchmark_configs(self, self.n_validation_size)
        return self._validation_kpi_configs

    def _calculate_validation_kpis(self) -> tuple[float, float, float]:
        """p05/p50/p95 |R_GS| over the fixed validation configuration stream."""
        from src.engine.model_evaluation import evaluate_plasma_kpis

        kpis = evaluate_plasma_kpis(
            self, self.validation_configs(), sample_size=KPI_POINTS_PER_CONFIG
        )
        return kpis["loss_p05"], kpis["loss_median"], kpis["loss_p95"]

    def _train_step(
        self,
        state: train_state.TrainState,
        inputs: FluxInput,
        weight_boundary_condition: float,
        huber_delta: float,
        weight_flux_scale: float,
        soft_bc: bool,
    ) -> tuple[
        train_state.TrainState,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        """Single training step; closed over self._psi_fn so plain/corrector share one kernel."""
        psi_fn = self._psi_fn

        @jax.checkpoint
        def loss_wrapper(
            params: any,
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            return Trainer.compute_loss(
                params,
                psi_fn,
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
        return (
            state.apply_gradients(grads=grads),
            loss,
            l_res,
            l_dir,
            l_per_cfg,
            grad_norm,
        )

    def train_epoch(self, epoch: int) -> tuple[float, float, float, float]:
        """Run one training epoch and aggregate every minibatch."""
        self.sampler.precompute_coordinate_samples()

        losses = []
        residuals = []
        boundaries = []
        batch_sizes = []
        per_config_losses = []
        grad_norms = []

        for i in range(0, len(self.train_set), self.config.batch_size):
            train_batch = self.train_set[i : i + self.config.batch_size]
            inputs = self.sampler.sample_flux_input(plasma_configs=train_batch)
            (
                self.state,
                loss,
                l_res,
                l_dir,
                per_config_loss,
                grad_norm,
            ) = self._train_step_jit(
                self.state,
                inputs,
                self.config.weight_boundary_condition,
                self.config.huber_delta,
                self.config.weight_flux_scale,
                self.config.soft_bc,
            )
            losses.append(loss)
            residuals.append(l_res)
            boundaries.append(l_dir)
            batch_sizes.append(len(train_batch))
            per_config_losses.append(per_config_loss)
            grad_norms.append(grad_norm)

        if epoch % RESAMPLING_FREQUENCY == 0 and epoch > 0:
            self.train_set = self.sampler.resample_train_set(
                train_set=self.train_set,
                epoch=epoch,
                per_config_losses=per_config_losses,
            )

        weights = jnp.asarray(batch_sizes)

        def weighted_mean(values: list[jnp.ndarray]) -> float:
            return float(jnp.average(jnp.asarray(values), weights=weights))

        return (
            weighted_mean(losses),
            weighted_mean(residuals),
            weighted_mean(boundaries),
            weighted_mean(grad_norms),
        )

    def train(self, observer: TrainingObserver | None = None) -> TrainingResult:
        """Run the full training loop: epoch budget, validation, early stopping, L-BFGS polish.

        ``observer`` is called once per epoch with an ``EpochMetrics`` event (a
        ``TrainingObserver``); it is the sole progress-reporting seam so this
        method never touches Rich, Plotext, or the filesystem.
        """
        training_started = time.perf_counter()
        self._lbfgs_params = None
        lbfgs_val_kpis = None
        val_kpis = None
        stopper = _Patience(
            EARLY_STOPPING_PATIENCE,
            EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT,
            EARLY_STOPPING_ROLLING_WINDOW,
        )
        best_params = None
        trained_epochs = 0
        stop_reason = "epoch_budget"

        for epoch in range(self.epochs):
            start_time = time.perf_counter()
            loss, residual, boundary, grad_norm = self.train_epoch(epoch)
            epoch_time = time.perf_counter() - start_time

            val_kpis = None
            should_stop = False
            if (epoch + 1) % VALIDATION_FREQUENCY == 0:
                val_kpis = self._calculate_validation_kpis()
                should_stop, improved = stopper.update(epoch + 1, val_kpis[1])
                if improved:
                    best_params = self.state.params

            lr = float(self._lr_schedule(self.state.step))
            trained_epochs = epoch + 1
            if observer is not None:
                observer(
                    EpochMetrics(
                        epoch=epoch,
                        loss=loss,
                        residual_loss=residual,
                        boundary_loss=boundary,
                        gradient_norm=grad_norm,
                        learning_rate=lr,
                        duration_seconds=epoch_time,
                        validation=ValidationMetrics(*val_kpis) if val_kpis is not None else None,
                        should_stop=should_stop,
                    )
                )
            if should_stop:
                stop_reason = "early_stopping"
                if best_params is not None:
                    self.state = self.state.replace(params=best_params)
                logger.info(
                    f"early stopping at epoch {trained_epochs}: no >= "
                    f"{EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT:.1%} rolling-average "
                    f"validation improvement for {EARLY_STOPPING_PATIENCE} rounds"
                )
                break

        if stop_reason != "early_stopping" and self.config.lbfgs_steps > 0:
            adamw_params = self.state.params
            self.lbfgs(self.config.lbfgs_steps)
            self._lbfgs_params = self.state.params
            lbfgs_val_kpis = self._calculate_validation_kpis()
            # Canonical weights stay the AdamW result; polish must never
            # overwrite them, so the polished set is persisted separately.
            self.state = self.state.replace(params=adamw_params)

        if stop_reason == "early_stopping" or val_kpis is None:
            val_kpis = self._calculate_validation_kpis()

        return TrainingResult(
            stop_reason=stop_reason,
            trained_epochs=trained_epochs,
            planned_epochs=self.epochs,
            final_validation=ValidationMetrics(*val_kpis),
            best_epoch=stopper.best_epoch,
            best_smoothed_validation_p50=(
                stopper.best_value if stopper.best_epoch is not None else None
            ),
            lbfgs_validation=(
                ValidationMetrics(*lbfgs_val_kpis) if lbfgs_val_kpis is not None else None
            ),
            optimizer_updates=int(self.state.step),
            duration_seconds=time.perf_counter() - training_started,
        )

    def lbfgs(self, steps: int, learning_rate: float | None = 0.1) -> None:
        """Polish AdamW-trained params with L-BFGS on one fixed batch."""
        self.sampler.precompute_coordinate_samples()
        batch = self.train_set[: self.config.batch_size]
        inputs = self.sampler.sample_flux_input(plasma_configs=batch)
        psi_fn = self._psi_fn

        def loss_fn(params: any) -> jnp.ndarray:
            total, _ = Trainer.compute_loss(
                params,
                psi_fn,
                inputs,
                self.config.weight_boundary_condition,
                self.config.huber_delta,
                self.config.weight_flux_scale,
                self.config.soft_bc,
            )
            return total

        # Damped steps: the polish objective is a single fixed batch, and full
        # linesearch-sized quasi-Newton steps over-specialize to it at the
        # expense of full-distribution KPIs.
        opt = optax.lbfgs(learning_rate=learning_rate)

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
        """Evaluate magnetic flux psi at physical coordinates."""
        return self._psi_fn_jit(self.state.params, R, Z, config)

    def make_psi_fn(self) -> Callable[[any, jnp.ndarray, jnp.ndarray, PlasmaConfig], jnp.ndarray]:
        """Factory returning the psi function for this field (single or composed)."""
        return self._field.make_psi_fn()
