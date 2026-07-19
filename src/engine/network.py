import argparse
from collections import deque
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime
import functools
from functools import partial
import json
import math
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
import plotext
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.measure import Measurement
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
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
from src.lib.run_artifacts import format_duration, gpu_name, kpi_values, load_run, write_json

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
LOG_FREQUENCY = 10  # Refresh the live table and flush metrics every N epochs
N_VALIDATION_SIZE = KPI_EVAL_CONFIGS  # Number of validation plasma configs
VALIDATION_FREQUENCY = 5 * LOG_FREQUENCY  # Evaluate validation set every N epochs
EARLY_STOPPING_PATIENCE = 6  # Stop after this many non-improving validation rounds
EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT = 0.01
EARLY_STOPPING_ROLLING_WINDOW = 3
# Cap rows shown in the live table; unbounded growth desyncs Rich's redraw
# once it overflows the terminal, corrupting scrollback.
LIVE_TABLE_MAX_ROWS = 15
CHART_HEIGHT = 18  # terminal rows for the side-by-side validation / lr+||∇L|| charts


class _PatienceStopper:
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
    """Builds and owns the psi-function for a NetworkManager.

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
# Terminal charts (plotext rendered as Rich text) + _MetricsManager
# ---------------------------------------------------------------------------


class _PlotextChart:
    """A plotext figure as a Rich renderable.

    Rebuilt only when the data length or terminal width changes — Rich Live
    re-renders at 10 Hz, so the built ANSI text is cached between epochs.
    """

    def __init__(self, draw: Callable[[list[dict]], None], rows: list[dict], height: int) -> None:
        self._draw = draw
        self._rows = rows
        self._height = height
        self._cache: tuple[tuple[int, int], Text] | None = None

    def __rich_console__(self, console: Console, options: any) -> any:
        key = (options.max_width, len(self._rows))
        if self._cache is None or self._cache[0] != key:
            try:
                plotext.clf()
                plotext.plotsize(options.max_width - 2, self._height)
                plotext.theme("pro")  # transparent bg, default fg: inherits terminal colors
                self._draw(self._rows)
                built = Text.from_ansi(plotext.build())
            except Exception as exc:  # a chart bug must never kill a training run
                built = Text(f"chart unavailable: {exc!r}", style="dim")
            self._cache = (key, built)
        yield self._cache[1]

    def __rich_measure__(self, console: Console, options: any) -> Measurement:
        # Charts stretch to whatever width they're given (lets a grid split evenly).
        return Measurement(20, options.max_width)


def _sci(x: float) -> str:
    """Compact scientific notation: 2e-3, 5e-5, 1.5e-3 (trailing zeros trimmed)."""
    mantissa, exp = f"{x:.1e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"


def _config_summary(config: "HyperParams") -> Table:
    """Two-row architecture/operating hyperparameter header for the metrics panel."""
    dims = config.hidden_dims
    layers = f"{dims[0]}×{len(dims)}" if len(set(dims)) == 1 else "-".join(map(str, dims))  # noqa: RUF001
    sep = " [dim]·[/] "
    architecture = sep.join(
        (
            f"[dim]arch[/] {config.arch}",
            f"[dim]layers[/] {layers}",
            f"[dim]soft_bc[/] {'on' if config.soft_bc else 'off'}",
            f"[dim]rwf[/] {'on' if config.rwf else 'off'}",
        )
    )
    total = config.warmup_epochs + config.decay_epochs
    operating = sep.join(
        (
            f"[dim]lr[/] {_sci(config.learning_rate_max)}→{_sci(config.learning_rate_min)}",
            f"[dim]epochs[/] {total} ({config.warmup_epochs}w/{config.decay_epochs}d)",
            f"[dim]wd[/] {_sci(config.weight_decay)}",
            f"[dim]batch[/] {config.batch_size}",
            f"[dim]σ_resample[/] {config.sigma_residual_adaptive_sampling:g}",  # noqa: RUF001
        )
    )
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", style="bold")
    grid.add_column()
    grid.add_row("architecture", architecture)
    grid.add_row("operating", operating)
    return grid


def _charts_renderable(rows: list[dict]) -> any:
    """Validation and lr/||∇L|| charts side by side; lr chart alone before first validation."""
    if not rows:
        return None
    lr_chart = _PlotextChart(_draw_lr_grad_chart, rows, CHART_HEIGHT)
    if not any(r["val_kpi_p50"] is not None for r in rows):
        return lr_chart
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(_PlotextChart(_draw_validation_chart, rows, CHART_HEIGHT), lr_chart)
    return grid


def _log_series(x: list, y: list) -> tuple[list, list]:
    """Drop points with y <= 0 — plotext's log scale raises on them (e.g. warmup lr=0)."""
    pairs = [(a, b) for a, b in zip(x, y, strict=False) if b is not None and b > 0]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _log_ticks(values: list[float]) -> tuple[list[float], list[str]]:
    """1-2-5 log-decade ticks covering the data range, labeled in scientific notation."""
    lo, hi = min(values), max(values)
    ticks, labels = [], []
    for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            if lo <= m * 10.0**k <= hi:
                ticks.append(m * 10.0**k)
                labels.append(f"{m}e{k}")
    thin = max(1, math.ceil(len(ticks) / 6))
    return ticks[::thin], labels[::thin]


def _linear_ticks(values: list[float]) -> tuple[list[float], list[str]]:
    """Round-number ticks (1-2-5 steps) for a linear axis."""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = 10.0 ** math.floor(math.log10(span / 5))
    for mult in (1, 2, 5, 10):
        if span / (step * mult) <= 5:
            step *= mult
            break
    ticks = [t * step for t in range(math.ceil(lo / step), math.floor(hi / step) + 1)]
    return ticks, [f"{t:g}" for t in ticks]


def _draw_validation_chart(rows: list[dict]) -> None:
    """Validation |R_GS| p05/p50/p95 vs epoch, log y. Colors match the table's Val KPI column."""
    val_rows = [r for r in rows if r["val_kpi_p50"] is not None]
    all_epochs, all_vals = [], []
    for key, label, color in (
        ("val_kpi_p95", "p95", "gray"),
        ("val_kpi_p50", "p50", "green"),
        ("val_kpi_p05", "p05", "gray"),
    ):
        epochs, series = _log_series([r["epoch"] for r in val_rows], [r.get(key) for r in val_rows])
        if epochs:
            plotext.plot(epochs, series, marker="braille", color=color, label=label)
            all_epochs += epochs
            all_vals += series
    plotext.yscale("log")
    if all_vals:
        plotext.yticks(*_log_ticks(all_vals))
        plotext.xticks(*_linear_ticks(all_epochs))
    plotext.title("validation |R_GS|")
    plotext.xlabel("epoch")


def _draw_lr_grad_chart(rows: list[dict]) -> None:
    """lr (left axis) and ||∇L|| (right axis) vs epoch, both log y."""
    epochs = [r["epoch"] for r in rows]
    lr_x, lr_y = _log_series(epochs, [r["lr"] for r in rows])
    gn_x, gn_y = _log_series(epochs, [r["grad_norm"] for r in rows])
    if lr_x:
        plotext.plot(lr_x, lr_y, marker="braille", color="yellow", label="lr")
        plotext.yticks(*_log_ticks(lr_y))
    if gn_x:
        plotext.plot(gn_x, gn_y, marker="braille", color="magenta", label="||∇L||", yside="right")
        plotext.yticks(*_log_ticks(gn_y), yside="right")
    plotext.yscale("log")
    plotext.yscale("log", yside="right")
    if epochs:
        plotext.xticks(*_linear_ticks(epochs))
    plotext.title("lr (left) · ||∇L|| (right)")
    plotext.xlabel("epoch")


class _MetricsManager:
    """Owns the Rich training table, progress bar, and completed metric windows."""

    def __init__(self, total_epochs: int, config: "HyperParams | None" = None) -> None:
        self._total_epochs = total_epochs
        self._config = config
        self.rows: list[dict] = []
        self._table_rows: deque[tuple[str, ...]] = deque(maxlen=LIVE_TABLE_MAX_ROWS)
        self._table = _new_table(with_title=False)
        self._progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(style="cyan", complete_style="bold cyan"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self._epoch_task = self._progress.add_task("Training", total=total_epochs)
        self._acc_loss = self._acc_res = self._acc_bnd = self._acc_gn = self._acc_t = 0.0
        self._acc_count = 0
        self._live: Live | None = None
        self.metrics_row_sink: Callable[[tuple[str, ...]], None] | None = None

    def renderable(self) -> Panel:
        parts: list[any] = [
            Align.center(Text("Training Metrics", style="italic")),
            Align.center(self._progress),
        ]
        if self._config is not None:
            parts.append(Align.center(_config_summary(self._config)))
        parts.append(Align.center(self._table))
        if (charts := _charts_renderable(self.rows)) is not None:
            parts.append(charts)
        return Panel(Group(*parts), border_style="cyan")

    def log(
        self,
        epoch: int,
        loss: float,
        residual: float,
        boundary: float,
        val_kpis: tuple[float, float, float] | None,
        lr: float,
        grad_norm: float,
        epoch_time: float,
    ) -> dict | None:
        self._acc_loss += loss
        self._acc_res += residual
        self._acc_bnd += boundary
        self._acc_gn += grad_norm
        self._acc_t += epoch_time
        self._acc_count += 1

        if (epoch + 1) % LOG_FREQUENCY == 0 or val_kpis is not None:
            p05, p50, p95 = val_kpis or (None, None, None)
            persisted = {
                "epoch": epoch + 1,
                "lr": lr,
                "loss": self._acc_loss / self._acc_count,
                "residual": self._acc_res / self._acc_count,
                "boundary": self._acc_bnd / self._acc_count,
                "grad_norm": self._acc_gn / self._acc_count,
                "epoch_time_seconds": self._acc_t / self._acc_count,
                "val_kpi_p05": p05,
                "val_kpi_p50": p50,
                "val_kpi_p95": p95,
            }
            self.rows.append(persisted)
            display_row = _metrics_row(
                epoch=epoch + 1,
                total_epochs=self._total_epochs,
                lr=lr,
                grad_norm=persisted["grad_norm"],
                loss=persisted["loss"],
                val_kpis=val_kpis,
                epoch_time=persisted["epoch_time_seconds"],
            )
            self._table_rows.append(display_row)
            if self.metrics_row_sink is not None:
                self.metrics_row_sink(display_row)
            table = _new_table(with_title=False)
            for r in self._table_rows:
                table.add_row(*r)
            self._table = table
            self._acc_loss = self._acc_res = self._acc_bnd = self._acc_gn = self._acc_t = 0.0
            self._acc_count = 0
        else:
            persisted = None

        self._progress.update(self._epoch_task, advance=1)
        if self._live is not None:
            self._live.update(self.renderable())
        return persisted


# ---------------------------------------------------------------------------
# _FileStorageManager — run-dir + artifact I/O
# ---------------------------------------------------------------------------


class _FileStorageManager:
    """Owns the run directory, artifact stem, and all file I/O for one training run."""

    def __init__(
        self,
        name: str,
        output_dir: Path | None,
        *,
        stage1_run_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.output_dir = output_dir
        self._stage1_run_dir = stage1_run_dir
        self.artifact_stem: str | None = None

    def new_artifact_stem(self) -> str:
        if self._stage1_run_dir is not None:
            return f"{self._stage1_run_dir.name}/stage2"
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.output_dir:
            return f"pinn_{timestamp}"
        return f"{timestamp}_{self.name}_{current_commit()}"

    def run_dir(self) -> Path:
        if self.artifact_stem is None:
            self.artifact_stem = self.new_artifact_stem()
        if self._stage1_run_dir is not None:
            return self._stage1_run_dir / "stage2"
        base = self.output_dir or Filepaths.BENCHMARKS
        return base / self.artifact_stem

    def discard_unsaved_run(self) -> None:
        """Delete the run dir unless a checkpoint was saved."""
        if self.artifact_stem is None:
            return
        run_dir = self.run_dir()
        if (run_dir / "network.flax").exists():
            return
        shutil.rmtree(run_dir, ignore_errors=True)
        if self.output_dir:
            with suppress(OSError):
                run_dir.parent.rmdir()

    def write_params(self, run_dir: Path, params: any) -> None:
        path = run_dir / "network.flax"
        temporary = path.with_suffix(".flax.tmp")
        temporary.write_bytes(flax.serialization.to_bytes(params))
        temporary.replace(path)

    def read_params(self, pinn_path: Path, target_params: any) -> any:
        with open(pinn_path, "rb") as f:
            return flax.serialization.from_bytes(target_params, f.read())

    def write_metrics(self, run_dir: Path, rows: list[dict]) -> None:
        fields = (
            "lr",
            "loss",
            "residual",
            "boundary",
            "grad_norm",
            "epoch_time_seconds",
            "val_kpi_p05",
            "val_kpi_p50",
            "val_kpi_p95",
        )
        metrics = {key: [row[key] for row in rows] for key in fields}
        write_json(
            run_dir / "metrics.json",
            {"format_version": 1, "logging_distance": LOG_FREQUENCY, **metrics},
        )

    def write_run(
        self,
        run_dir: Path,
        manager: "NetworkManager",
        duration: str | None,
        result: dict,
    ) -> None:
        write_json(
            run_dir / "run.json",
            {
                "format_version": 1,
                "commit": current_commit(),
                "duration": duration,
                "device": manager.device,
                "seed": manager.seed,
                "config": manager.config.to_dict(),
                "result": result,
            },
        )

    def benchmark(self, manager: "NetworkManager", test_mode: bool) -> dict:
        """Save the residual montage and return post-training KPIs."""
        if test_mode:
            return {}
        from src.engine.model_evaluation import (
            EVAL_RESOLUTION,
            N_PLOTS,
            build_kpi_record,
            evaluate_plasma_grids,
            evaluate_plasma_kpis,
            kpi_benchmark_configs,
            plot_plasma_grid_montage,
        )

        run_dir = self.run_dir()
        configs = kpi_benchmark_configs(manager, KPI_EVAL_CONFIGS)
        kpis = evaluate_plasma_kpis(manager, configs, sample_size=KPI_POINTS_PER_CONFIG)
        grids = evaluate_plasma_grids(
            manager, configs[:N_PLOTS], resolution=EVAL_RESOLUTION, quantities=("residual",)
        )
        plot_plasma_grid_montage(
            grids,
            run_dir / "residual.png",
            quantity="residual",
            title=self.artifact_stem,
            metadata=manager.config.to_dict(),
            display_parameters=(
                "huber_delta",
                "learning_rate_max",
                "n_fourier_features",
                "lbfgs_steps",
            ),
            kpis=kpis,
        )
        record = build_kpi_record(manager, kpis, KPI_EVAL_CONFIGS, KPI_POINTS_PER_CONFIG, 0.85)
        logger.info(f"residual plot saved to {run_dir}")
        return kpi_values(record)

    def save_training_curves(self, run_dir: Path, rows: list[dict], artifact_stem: str) -> None:
        if not rows:
            return
        from src.engine.model_evaluation import plot_training_curves

        plot_training_curves(
            run_dir / "metrics.json", run_dir / "training_curves.png", title=artifact_stem
        )


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
        *,
        prior: FoundationModel | None = None,
        scale: float = 1.0,
        stage1_run_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.n_validation_size = n_validation_size
        self.test_mode = test_mode
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

        if stage1_run_dir is not None and prior is None:
            raise ValueError("stage1_run_dir is only valid for a corrector")
        self._prior = prior
        self._scale = scale
        self._files = _FileStorageManager(
            name=name,
            output_dir=output_dir,
            stage1_run_dir=stage1_run_dir,
        )

        self.train_set = self.sampler._get_sobol_sample(
            n_samples=self.config.n_train,
            lower_bounds=self.sampler._domain_lower_bounds,
            upper_bounds=self.sampler._domain_upper_bounds,
        )
        self.training_log: list[dict] = []
        self.training_summary: dict | None = None
        self.artifact_stem: str | None = None
        self.device = "unknown"
        self.training_duration_seconds: float | None = None

        # Instance-level jitted train_step — closed over self._psi_fn so both the plain
        # and composed cases use one kernel.  soft_bc must stay static (pinn_loss_function
        # declares it static_argnames) so it is declared static here too.
        self._train_step_jit = jax.jit(self._train_step, static_argnames=("soft_bc",))

    @classmethod
    def for_inference(
        cls,
        config: HyperParams,
        params: any,
        *,
        prior: FoundationModel | None = None,
        scale: float = 1.0,
        seed: int = BASE_SEED,
    ) -> "NetworkManager":
        """Construct a minimal manager for inference only (no training artifacts needed).

        Builds the full NetworkManager (sampler is needed by kpi_benchmark_configs /
        build_sample_response) but skips any disk I/O. Params are injected directly.
        """
        mgr = cls(config, seed=seed, prior=prior, scale=scale)
        mgr.state = mgr.state.replace(params=params)
        return mgr

    # ------------------------------------------------------------------
    # Artifact stem / run-dir (delegates to _FileStorageManager)
    # ------------------------------------------------------------------

    def _new_artifact_stem(self) -> str:
        return self._files.new_artifact_stem()

    def run_dir(self) -> Path:
        stem = self._files.artifact_stem or self._files.new_artifact_stem()
        self._files.artifact_stem = stem
        self.artifact_stem = stem
        return self._files.run_dir()

    def discard_unsaved_run(self) -> None:
        """Delete the benchmark run dir unless this run's checkpoint was saved."""
        self._files.discard_unsaved_run()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_disk(self) -> str:
        """Save params, consolidated run data, metrics, and benchmark plots.

        Correctors configured with ``stage1_run_dir`` are nested under
        ``<stage1_run_dir>/stage2/`` and record their scale in ``run.json``.
        """
        if self._files.artifact_stem is None:
            self._files.artifact_stem = self._files.new_artifact_stem()
        self.artifact_stem = self._files.artifact_stem

        run_dir = self._files.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.device == "unknown":
            self.device = gpu_name()

        self._files.write_metrics(run_dir, self.training_log)
        kpis = self._files.benchmark(self, self.test_mode)
        result = {
            "status": "completed",
            **(self.training_summary or {}),
            "optimizer_updates": int(self.state.step),
            "examples_processed": int(self.state.step) * self.config.batch_size,
            "peak_memory_bytes": int(
                (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)
            ),
            "kpis": kpis,
        }
        if self._prior is not None:
            result["stage2_scale"] = self._scale
        self._files.write_run(
            run_dir,
            self,
            format_duration(self.training_duration_seconds)
            if self.training_duration_seconds is not None
            else None,
            result,
        )
        self._files.save_training_curves(run_dir, self.training_log, self.artifact_stem)
        self._files.write_params(run_dir, self.state.params)
        return self.artifact_stem

    def from_disk(self, pinn_path) -> any:  # noqa
        """Load Flax model parameters from disk."""
        return self._files.read_params(pinn_path, self.state.params)

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

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
        total, (l_res, l_dir, l_per_cfg) = NetworkManager.compute_loss(
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
        """The manager's fixed validation configs: the first n_validation_size of the
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
            return NetworkManager.compute_loss(
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

    def training_renderable(self) -> Panel:
        """Current metrics table + progress bar; rendered by Live or the HPO TUI."""
        return self._metrics.renderable()

    def train(
        self,
        save_to_disk: bool = True,
        validation_callback: Callable[[int, float | None], None] | None = None,
        show_progress: bool = True,
    ) -> float:
        self._files.artifact_stem = self._files.new_artifact_stem()
        self.artifact_stem = self._files.artifact_stem

        # Only materialise a benchmark run dir when we intend to keep artifacts.
        # Correctors (save_to_disk=False) train in memory and persist later.
        run_dir = None
        if save_to_disk:
            run_dir = self._files.run_dir()
            run_dir.mkdir(parents=True, exist_ok=True)
            self.device = gpu_name()
            self._files.write_run(run_dir, self, None, {"status": "running"})

        training_started = time.perf_counter()
        try:
            self._metrics = _MetricsManager(total_epochs=self.epochs, config=self.config)
            self._metrics.metrics_row_sink = self.metrics_row_sink
            self._live = None
            if show_progress:
                live = Live(
                    self.training_renderable(),
                    refresh_per_second=10,
                    console=console,
                    vertical_overflow="visible",
                )
            else:
                live = nullcontext()
            val_kpis = None
            stopper = _PatienceStopper(
                EARLY_STOPPING_PATIENCE,
                EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT,
                EARLY_STOPPING_ROLLING_WINDOW,
            )
            best_params = None
            trained_epochs = 0
            stop_reason = "epoch_budget"
            with live as active_live:
                if show_progress:
                    self._live = active_live
                    self._metrics._live = active_live
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
                    persisted = self._metrics.log(
                        epoch,
                        loss,
                        residual,
                        boundary,
                        val_kpis,
                        lr,
                        grad_norm,
                        epoch_time,
                    )
                    trained_epochs = epoch + 1
                    if run_dir is not None and (persisted is not None or val_kpis is not None):
                        self._files.write_metrics(run_dir, self._metrics.rows)
                    if validation_callback is not None and not should_stop:
                        validation_callback(epoch + 1, val_kpis[1] if val_kpis else None)
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
                self.lbfgs(self.config.lbfgs_steps)
                val_kpis = self._calculate_validation_kpis()

            if stop_reason == "early_stopping" or val_kpis is None:
                val_kpis = self._calculate_validation_kpis()

            self.training_log = self._metrics.rows
            self.training_summary = {
                "stop_reason": stop_reason,
                "trained_epochs": trained_epochs,
                "planned_epochs": self.epochs,
                "final_val_kpi_p05": val_kpis[0],
                "final_val_kpi_p50": val_kpis[1],
                "final_val_kpi_p95": val_kpis[2],
                "best_smoothed_val_kpi_p50": (
                    stopper.best_value if stopper.best_epoch is not None else None
                ),
                "best_validation_epoch": stopper.best_epoch,
                "validation_frequency": VALIDATION_FREQUENCY,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "early_stopping_min_relative_improvement": (
                    EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT
                ),
                "early_stopping_rolling_window": EARLY_STOPPING_ROLLING_WINDOW,
            }
            self.training_duration_seconds = time.perf_counter() - training_started

            if save_to_disk:
                self.to_disk()
            return val_kpis[1]
        except BaseException as exc:
            if run_dir is not None:
                self.training_duration_seconds = time.perf_counter() - training_started
                metrics = getattr(self, "_metrics", None)
                if metrics is not None and metrics.rows:
                    self._files.write_metrics(run_dir, metrics.rows)
                self._files.write_run(
                    run_dir,
                    self,
                    format_duration(self.training_duration_seconds),
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
            raise

    def lbfgs(self, steps: int) -> None:
        """Polish AdamW-trained params with L-BFGS on one fixed batch."""
        self.sampler.precompute_coordinate_samples()
        batch = self.train_set[: self.config.batch_size]
        inputs = self.sampler.sample_flux_input(plasma_configs=batch)
        psi_fn = self._psi_fn

        def loss_fn(params: any) -> jnp.ndarray:
            total, _ = NetworkManager.compute_loss(
                params,
                psi_fn,
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
        """Evaluate magnetic flux psi at physical coordinates."""
        return self._psi_fn_jit(self.state.params, R, Z, config)

    @staticmethod
    def _new_table() -> Table:
        """Training Metrics table skeleton; kept as a static method for HPO TUI compat."""
        return _new_table()

    def make_psi_fn(self) -> Callable[[any, jnp.ndarray, jnp.ndarray, PlasmaConfig], jnp.ndarray]:
        """Factory returning the psi function for this field (single or composed)."""
        return self._field.make_psi_fn()


def _new_table(with_title: bool = True) -> Table:
    # The live panel and --show replay render their own centered heading; the HPO
    # TUI's sequential per-trial log still wants the title on the table itself.
    table = Table(
        title="Training Metrics" if with_title else None,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Epoch", justify="right", style="cyan")
    table.add_column("LR", justify="right", style="yellow")
    table.add_column("||∇L||", justify="right", style="magenta")
    table.add_column("Loss", justify="right", style="magenta")
    table.add_column("Val p05", justify="right", style="green")
    table.add_column("Val p50", justify="right", style="green")
    table.add_column("Val p95", justify="right", style="green")
    table.add_column("Time/Ep", justify="right")
    return table


def _metrics_row(
    epoch: int,
    total_epochs: int,
    lr: float,
    grad_norm: float,
    loss: float,
    val_kpis: tuple[float, float, float] | None,
    epoch_time: float,
) -> tuple[str, ...]:
    """One Training Metrics table row shared by live display and replay."""
    p05, p50, p95 = val_kpis or (None, None, None)
    return (
        f"{epoch}/{total_epochs}",
        f"{lr:.2e}",
        f"{grad_norm:.2e}",
        f"{loss:.2e}",
        f"{p05:.2e}" if p05 is not None else "-",
        f"{p50:.2e}" if p50 is not None else "-",
        f"{p95:.2e}" if p95 is not None else "-",
        f"{epoch_time:.2f}s",
    )


def show_run(run: str) -> None:
    """Re-render the Training Metrics table for a stored run from metrics.json."""
    run_dir = Path(run)
    if not run_dir.is_dir():
        candidates = [
            Filepaths.BENCHMARKS / run,
            *Filepaths.BENCHMARKS.glob(f"*/{run}"),
            *(Filepaths.DATA / "hpo").glob(f"*/{run}"),
        ]
        run_dir = next((p for p in candidates if p.is_dir()), run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"no metrics.json found for run '{run}'")

    metrics = json.loads(metrics_path.read_text())
    distance = int(metrics["logging_distance"])
    run_record = load_run(run_dir)
    total_epochs = int(run_record.get("result", {}).get("trained_epochs", 0))
    if not total_epochs:
        total_epochs = len(metrics["lr"]) * distance
    rows = [
        {
            "epoch": (index + 1) * distance,
            "lr": lr,
            "grad_norm": metrics["grad_norm"][index],
            "val_kpi_p05": metrics["val_kpi_p05"][index],
            "val_kpi_p50": metrics["val_kpi_p50"][index],
            "val_kpi_p95": metrics["val_kpi_p95"][index],
        }
        for index, lr in enumerate(metrics["lr"])
    ]
    table = _new_table(with_title=False)
    for index, row in enumerate(rows):
        table.add_row(
            *_metrics_row(
                epoch=row["epoch"],
                total_epochs=total_epochs,
                lr=row["lr"],
                grad_norm=row["grad_norm"],
                loss=metrics["loss"][index],
                val_kpis=(row["val_kpi_p05"], row["val_kpi_p50"], row["val_kpi_p95"])
                if row["val_kpi_p50"] is not None
                else None,
                epoch_time=metrics["epoch_time_seconds"][index],
            )
        )
    parts: list[any] = [
        Align.center(Text("Training Metrics", style="italic")),
        Align.center(table),
    ]
    if (charts := _charts_renderable(rows)) is not None:
        parts.append(charts)
    console.print(Panel(Group(*parts), border_style="cyan"))


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
        "--rwf",
        action="store_true",
        help="Random Weight Factorization (Wang et al. arXiv 2210.01274): reparametrize "
        "each dense kernel as W = V * exp(s) to improve PINN accuracy",
    )
    parser.add_argument(
        "--arch",
        choices=["mlp", "piratenet"],
        default="mlp",
        help="Network architecture: 'mlp' = plain MLP (default), 'piratenet' = PirateNet "
        "residual blocks (arXiv 2402.00326)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default=None,
        help="Comma-separated hidden layer widths, e.g. 128,128,128,128; for piratenet each "
        "entry is one residual block of that width (default: HyperParams default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Plasma configs per train step (default: HyperParams 64); smaller batches "
        "halve activation memory, unlocking wider nets on 12GB",
    )
    parser.add_argument(
        "--n-train", type=int, default=None, help="Training configurations sampled per epoch"
    )
    parser.add_argument(
        "--inner-samples", type=int, default=None, help="Interior collocation points per config"
    )
    parser.add_argument(
        "--boundary-samples", type=int, default=None, help="Boundary points per config"
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
                rwf=args.rwf,
                arch=args.arch,
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
            if args.hidden_dims is not None:
                config = config.replace(
                    hidden_dims=tuple(int(d) for d in args.hidden_dims.split(","))
                )
            if args.batch_size is not None:
                config = config.replace(batch_size=args.batch_size)
            if args.n_train is not None:
                config = config.replace(n_train=args.n_train)
            if args.inner_samples is not None:
                config = config.replace(n_rz_inner_samples=args.inner_samples)
            if args.boundary_samples is not None:
                config = config.replace(n_rz_boundary_samples=args.boundary_samples)
            if (
                min(
                    config.batch_size,
                    config.n_train,
                    config.n_rz_inner_samples,
                    config.n_rz_boundary_samples,
                )
                <= 0
            ):
                parser.error("training and sample budgets must be positive")
            if config.n_train % config.batch_size:
                parser.error("--n-train must be divisible by --batch-size")
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
