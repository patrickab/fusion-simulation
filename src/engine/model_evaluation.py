from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from src.engine.network import BASE_SEED, NetworkManager
from src.engine.physics import estimate_psi_axis, grad_shafranov_residual
from src.engine.plasma import get_poloidal_points
from src.lib.config import current_commit
from src.lib.geometry_config import PlasmaConfig
from src.lib.network_config import HyperParams

GridQuantity = Literal["flux", "residual"]
DEFAULT_KPI_SAMPLE_SIZE = 16_384
EVAL_CONFIG_COUNT = 8
# Polar-mesh residual montages paint each (theta, rho) cell as one flat/gouraud
# patch; cell arc-length grows with rho, so at low theta resolution a single
# narrow high-residual filament near the O-point gets flat-filled across a
# whole wedge, badly overstating its true extent. 600 was verified (visually,
# against the frontend's contourcarpet renderer at its own max resolution of
# 300) to collapse those wedges back down to the filaments' real footprint.
EVAL_RESOLUTION = 600
# Per-config point-eval batch size for grid evaluation (see `fields` in
# evaluate_plasma_grids) — bounds peak memory independent of EVAL_RESOLUTION.
# 20_000 matches the old resolution=200 grid (n_theta*n_rho), a size already
# proven safe mid-training on a 12GB GPU.
GRID_EVAL_CHUNK = 20_000
# Fixed residual colorbar range, shared by the montage plot and the frontend
# (via /api/config) so both render on one comparable scale. [0, 1] was the
# right span for soft-BC checkpoints; the hard-BC fix (commit 8a8b9a4) and its
# corrected-physics collapse-guard pushed core residuals into the 1e-4-1e-2
# band (N3 sweep run: core_loss_median 0.0025 vs bb503b0's soft-BC 0.087), so
# [0, 1] now renders every current checkpoint as a flat black square.
RESIDUAL_COLOR_RANGE: tuple[float, float] = (0.0, 0.01)


def build_kpi_record(
    manager: NetworkManager,
    kpis: Mapping[str, float],
    n_configs: int,
    n_points: int,
    core_rho: float,
    network_name: str | None = None,
) -> dict:
    """Assemble the kpis.json record dict shared by training and CLI eval.

    When network_name is None (training path), derive it from the current
    commit + the manager's artifact_stem. When provided (CLI path), use it
    directly — the manager may not have artifact_stem set if loaded from disk.
    """
    hp = manager.config
    if network_name is None:
        network_name = f"{current_commit()}/{manager.artifact_stem}"
    loss_label = f"huber={hp.huber_delta:g}" if hp.huber_delta > 0 else "mse"
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "network": network_name,
        "loss": loss_label,
        "lr_max": hp.learning_rate_max,
        "n_fourier_features": hp.n_fourier_features,
        "lbfgs_steps": hp.lbfgs_steps,
        "n_configs": n_configs,
        "n_points": n_points,
        "core_rho": core_rho,
        **kpis,
    }


@dataclass(frozen=True)
class PlasmaGridBatch:
    """Display-ready fields on one shared normalized grid per plasma config."""

    theta: jnp.ndarray
    rho: jnp.ndarray
    R: jnp.ndarray
    Z: jnp.ndarray
    boundary_R: jnp.ndarray
    boundary_Z: jnp.ndarray
    values: dict[GridQuantity, jnp.ndarray]


def estimate_axis_for_config(
    manager: NetworkManager,
    config: PlasmaConfig,
    sample_size: int = DEFAULT_KPI_SAMPLE_SIZE,
    seed: int = BASE_SEED + 124,
) -> jnp.ndarray:
    """psi_axis on the fixed area-uniform Sobol sample used for KPI ranking.

    Callers that evaluate on a different, smaller point set (a display grid,
    an arbitrary scatter) must reuse this instead of refitting axis locally —
    otherwise the same checkpoint normalizes its residual differently in the
    KPI table vs. whatever it happens to be rendered against.
    """
    unit_points = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(
        (sample_size - 1).bit_length()
    )[:sample_size]
    theta = jnp.asarray(2.0 * np.pi * unit_points[:, 0], dtype=jnp.float32)
    rho = jnp.asarray(np.sqrt(unit_points[:, 1]), dtype=jnp.float32)
    R, Z = jax.vmap(lambda t, r: get_poloidal_points(t, config.Geometry, r))(theta, rho)
    psi_fn = manager.make_psi_fn()
    psi = jax.vmap(lambda r, z: psi_fn(manager.state.params, r, z, config))(R, Z)
    return estimate_psi_axis(psi)


def compute_gs_residual_on_points(
    manager: NetworkManager,
    config: PlasmaConfig,
    R_pts: jnp.ndarray,
    Z_pts: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate normalised Grad-Shafranov residual at supplied ``(R, Z)`` points."""
    psi_fn = manager.make_psi_fn()
    params = manager.state.params

    psi_axis = estimate_axis_for_config(manager, config)

    residual_fn = jax.vmap(
        lambda r, z: grad_shafranov_residual(psi_fn, params, r, z, psi_axis, config)
    )
    return residual_fn(R_pts, Z_pts)


def evaluate_plasma_kpis(
    manager: NetworkManager,
    configs: Sequence[PlasmaConfig],
    sample_size: int = DEFAULT_KPI_SAMPLE_SIZE,
    core_rho: float = 0.85,
    seed: int = BASE_SEED + 124,
) -> dict[str, float]:
    """Evaluate ``|R_GS|`` KPIs on a deterministic area-uniform Sobol sample."""
    if not configs:
        raise ValueError("At least one plasma config is required")
    if sample_size < 16:
        raise ValueError("KPI sample_size must be at least 16")
    if not 0.0 < core_rho < 1.0:
        raise ValueError("core_rho must be between 0 and 1")

    # Generate a power-of-two Sobol block (its balance guarantee), then truncate
    # so callers can still request any exact sample size without SciPy warnings.
    unit_points = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(
        (sample_size - 1).bit_length()
    )[:sample_size]
    theta = jnp.asarray(2.0 * np.pi * unit_points[:, 0], dtype=jnp.float32)
    # sqrt maps uniform unit-square samples to uniform area in the normalized disk.
    rho = jnp.asarray(np.sqrt(unit_points[:, 1]), dtype=jnp.float32)
    core = np.asarray(rho) < core_rho
    if not core.any() or core.all():
        raise ValueError("KPI sample does not cover both core and edge regions")

    psi_fn = manager.make_psi_fn()
    params = manager.state.params
    boundary_theta = jnp.linspace(0.0, 2.0 * jnp.pi, 257)[:-1]
    losses = []
    boundary_leaks = []
    for config in configs:
        R, Z = jax.vmap(lambda t, r, cfg=config: get_poloidal_points(t, cfg.Geometry, r))(
            theta, rho
        )
        psi = jax.vmap(lambda r, z, cfg=config: psi_fn(params, r, z, cfg))(R, Z)
        psi_axis = estimate_axis_for_config(manager, config, sample_size, seed)
        residual = jax.vmap(
            lambda r, z, axis=psi_axis, cfg=config: grad_shafranov_residual(
                psi_fn, params, r, z, axis, cfg
            )
        )(R, Z)
        losses.append(np.abs(np.asarray(residual)))

        R_boundary, Z_boundary = get_poloidal_points(boundary_theta, config.Geometry, 1.0)
        psi_boundary = jax.vmap(lambda r, z, cfg=config: psi_fn(params, r, z, cfg))(
            R_boundary, Z_boundary
        )
        # Use the strongest 5% by magnitude so leakage remains meaningful for
        # checkpoints that converged to either flux sign.
        n_axis = max(1, sample_size // 20)
        flux_depth = jnp.mean(jnp.sort(jnp.abs(psi))[-n_axis:])
        boundary_leaks.append(float(jnp.max(jnp.abs(psi_boundary)) / (flux_depth + 1e-12)))

    loss = np.stack(losses)

    def summary(values: np.ndarray, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}loss_median": float(np.median(values)),
            f"{prefix}loss_mean": float(np.mean(values)),
            f"{prefix}loss_p95": float(np.percentile(values, 95)),
            f"{prefix}loss_p05": float(np.percentile(values, 5)),
        }

    return {
        **summary(loss.ravel(), ""),
        **summary(loss[:, core].ravel(), "core_"),
        "edge_loss_p95": float(np.percentile(loss[:, ~core], 95)),
        "boundary_leak_max": max(boundary_leaks),
    }


def evaluate_plasma_grids(
    manager: NetworkManager,
    configs: Sequence[PlasmaConfig],
    resolution: int,
    quantities: Sequence[GridQuantity] = ("flux", "residual"),
) -> PlasmaGridBatch:
    """Evaluate display fields on identical plasma-aligned grids.

    ``resolution`` is the poloidal resolution; radial resolution is half as large.
    The linear-rho grid is intended for structured plots. Area statistics should
    continue to use the sqrt-rho Sobol samples used by ``Sampler``.
    """
    if resolution < 4:
        raise ValueError("resolution must be at least 4")
    requested = tuple(dict.fromkeys(quantities))
    if not requested:
        raise ValueError("At least one grid quantity is required")
    unsupported = set(requested) - {"flux", "residual"}
    if unsupported:
        raise ValueError(f"Unsupported grid quantities: {sorted(unsupported)}")
    if not configs:
        raise ValueError("At least one plasma config is required")

    n_theta = resolution
    n_rho = max(16, resolution // 2)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta)
    # Avoid the degenerate rho=0 row, where every theta maps to the same point.
    rho = jnp.linspace(0.03, 1.0, n_rho)
    theta_grid, rho_grid = jnp.meshgrid(theta, rho)
    theta_flat, rho_flat = theta_grid.ravel(), rho_grid.ravel()

    batched_config = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *configs)

    def coordinates(cfg: PlasmaConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.vmap(lambda t, r: get_poloidal_points(t, cfg.Geometry, r))(theta_flat, rho_flat)

    R_flat, Z_flat = jax.vmap(coordinates)(batched_config)
    psi_fn = manager.make_psi_fn()
    params = manager.state.params

    # Axis fixed per config on the same Sobol sample used for KPI ranking —
    # not refit from this grid's own (much smaller, differently-shaped) points,
    # so the rendered residual always matches the reported KPI's normalization.
    axis_per_config = jnp.stack([estimate_axis_for_config(manager, cfg) for cfg in configs])

    def fields(
        R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig, psi_axis: jnp.ndarray
    ) -> tuple[jnp.ndarray, ...]:
        # Chunked rather than a flat vmap over all n_theta*n_rho points: this is
        # already vmapped over n_configs outside, so an unchunked inner vmap
        # scales peak memory with resolution^2 * n_configs at once — that's what
        # OOM'd a GPU mid-run when EVAL_RESOLUTION went up (2026-07-13). Capping
        # the inner batch keeps peak memory flat regardless of resolution.
        psi = jax.lax.map(
            lambda rz: psi_fn(params, rz[0], rz[1], cfg), (R, Z), batch_size=GRID_EVAL_CHUNK
        )
        output = []
        if "flux" in requested:
            output.append(psi)
        if "residual" in requested:
            residual = jax.lax.map(
                lambda rz: grad_shafranov_residual(psi_fn, params, rz[0], rz[1], psi_axis, cfg),
                (R, Z),
                batch_size=GRID_EVAL_CHUNK,
            )
            # Linear |R_GS|: post-fix residuals are O(0.1-1), where a log scale
            # only spreads the noise floor; display range is fixed [0, 1].
            output.append(jnp.abs(residual))
        return tuple(output)

    evaluated = jax.vmap(fields)(R_flat, Z_flat, batched_config, axis_per_config)
    values = {
        quantity: field.reshape(len(configs), n_rho, n_theta)
        for quantity, field in zip(requested, evaluated, strict=True)
    }

    boundary_theta = jnp.linspace(0.0, 2.0 * jnp.pi, 257)[:-1]

    def boundary(cfg: PlasmaConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
        return get_poloidal_points(boundary_theta, cfg.Geometry, 1.0)

    boundary_R, boundary_Z = jax.vmap(boundary)(batched_config)
    return PlasmaGridBatch(
        theta=theta,
        rho=rho,
        R=R_flat.reshape(len(configs), n_rho, n_theta),
        Z=Z_flat.reshape(len(configs), n_rho, n_theta),
        boundary_R=boundary_R,
        boundary_Z=boundary_Z,
        values=values,
    )


def plot_plasma_grid_montage(
    grids: PlasmaGridBatch,
    output_path: str | Path,
    quantity: GridQuantity = "residual",
    title: str | None = None,
    metadata: Mapping[str, object] | None = None,
    display_parameters: Sequence[str] = (),
    kpis: Mapping[str, float] | None = None,
) -> None:
    """Save one coherent Matplotlib figure from shared evaluated grid data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if quantity not in grids.values:
        raise ValueError(f"Quantity {quantity!r} was not evaluated")
    metadata = metadata or {}
    missing = set(display_parameters) - metadata.keys()
    if missing:
        raise ValueError(f"Unknown display parameters: {sorted(missing)}")

    n_configs = grids.R.shape[0]
    n_cols = min(4, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    # Extra height for the KPI table below the montage
    kpi_table_height = 1.0 if kpis else 0.0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows + kpi_table_height),
        squeeze=False,
        layout="constrained",
    )
    values = np.asarray(grids.values[quantity])
    if quantity == "flux":
        # Data-driven: a fixed range hides sign-flipped or collapsed fields;
        # the colorbar ticks expose a flat ψ≈0 run immediately.
        vmin, vmax = float(values.min()), float(values.max())
        if vmax - vmin < 1e-12:
            vmax = vmin + 1e-6
        color = {"cmap": "viridis", "vmin": vmin, "vmax": vmax, "label": r"$\psi$"}
    else:
        # Fixed linear scale (RESIDUAL_COLOR_RANGE): residual montages from
        # different checkpoints must be directly comparable by eye.
        vmin, vmax = RESIDUAL_COLOR_RANGE
        color = {"cmap": "magma", "vmin": vmin, "vmax": vmax, "label": r"$|R_{GS}|$"}
    image = None
    for i, ax in enumerate(axes.ravel()):
        if i >= n_configs:
            ax.set_visible(False)
            continue
        image = ax.pcolormesh(
            np.asarray(grids.R[i]),
            np.asarray(grids.Z[i]),
            values[i],
            cmap=color["cmap"],
            vmin=color["vmin"],
            vmax=color["vmax"],
            shading="gouraud",
        )
        ax.plot(
            np.asarray(grids.boundary_R[i]),
            np.asarray(grids.boundary_Z[i]),
            color="0.65",
            lw=0.8,
        )
        ax.set_aspect("equal")
        ax.set_title(f"cfg {i}", fontsize=9)
        ax.tick_params(labelsize=7)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label=color["label"])

    parameter_text = ", ".join(f"{key}={metadata[key]}" for key in display_parameters)
    heading = "\n".join(part for part in (title, parameter_text) if part)
    if heading:
        fig.suptitle(heading, fontsize=10)

    if kpis:
        _add_kpi_table(fig, kpis)

    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def plot_training_curves(
    csv_path: str | Path, output_path: str | Path, title: str | None = None
) -> None:
    """Two-panel training overview from training.csv: loss/val-loss on top,
    LR + gradient norm (twinned y-axes) below — one glance to see whether a
    run is still descending or has actually annealed to its LR floor."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs: list[int] = []
    lr: list[float] = []
    loss: list[float] = []
    grad_norm: list[float] = []
    val_epochs: list[int] = []
    val_loss: list[float] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            lr.append(float(row["lr"]))
            loss.append(float(row["moving_avg_loss"]))
            grad_norm.append(float(row["grad_norm"]))
            if row["val_loss"]:
                val_epochs.append(int(row["epoch"]))
                val_loss.append(float(row["val_loss"]))

    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")

    ax_loss.plot(epochs, loss, color="#6cce5a", lw=1.2, label="train loss")
    if val_loss:
        ax_loss.plot(
            val_epochs, val_loss, color="#fde725", marker="o", markersize=3, lw=0, label="val loss"
        )
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.set_title(title or Path(csv_path).parent.name, fontsize=10)

    ax_lr.plot(epochs, lr, color="#26828e", lw=1.2)
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("learning rate", color="#26828e")
    ax_lr.tick_params(axis="y", labelcolor="#26828e")
    ax_lr.set_xlabel("epoch")

    ax_gn = ax_lr.twinx()
    ax_gn.plot(epochs, grad_norm, color="#d4d4d8", lw=1.0, alpha=0.7)
    ax_gn.set_yscale("log")
    ax_gn.set_ylabel("||∇L||", color="#d4d4d8")
    ax_gn.tick_params(axis="y", labelcolor="#d4d4d8")

    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def _add_kpi_table(fig: object, kpis: Mapping[str, float]) -> None:
    """Render KPIs as a 2-row (All/Core) x 4-col (mean/median/p95/p05) table."""
    stats = ["mean", "median", "p95", "p05"]
    rows = ["All", "Core"]
    cell_text: list[list[str]] = []
    for row in rows:
        prefix = "core_" if row == "Core" else ""
        cell_text.append([f"{kpis.get(f'{prefix}loss_{s}', float('nan')):.3e}" for s in stats])

    ax = fig.add_axes([0.12, -0.02, 0.76, 0.10])
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=stats,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # Style header row
    for j in range(len(stats)):
        cell = table[0, j]
        cell.set_facecolor("#444")
        cell.set_text_props(color="white", weight="bold")
    # Style data cells
    for i in range(1, len(rows) + 1):
        for j in range(len(stats)):
            table[i, j].set_facecolor("#222" if i % 2 else "#333")
            table[i, j].set_text_props(color="white", family="monospace")


if __name__ == "__main__":
    # Region-split |GS residual| report: selection metric is the plasma core
    # (rho < --core-rho); the edge shell is reported but tolerated by design.
    # Each run dir (data/benchmarks/<commit>/<run>/) already holds network.flax,
    # config.json and training.csv from training; this CLI adds kpis.json and the
    # montage PNG (fixed linear color scale for comparability) into the same dir.
    import argparse
    from datetime import datetime

    from src.lib.network_config import DomainBounds, HyperParams
    from src.streamlit.network_utils import get_available_networks, resolve_run_directory

    parser = argparse.ArgumentParser(description="Region-split |GS residual| KPIs per checkpoint")
    parser.add_argument(
        "networks",
        nargs="*",
        help="Network names as commit/run (default: all in data/benchmarks/)",
    )
    parser.add_argument("--n-configs", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=DEFAULT_KPI_SAMPLE_SIZE)
    parser.add_argument("--core-rho", type=float, default=0.85)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-resolution", type=int, default=96)
    parser.add_argument("--plot-title")
    parser.add_argument("--plot-quantity", choices=("flux", "residual"), default="residual")
    parser.add_argument(
        "--plot-parameters",
        nargs="*",
        default=("huber_delta", "learning_rate_max", "n_fourier_features", "lbfgs_steps"),
        help="Hyperparameter names rendered below the montage title",
    )
    args = parser.parse_args()

    names = args.networks or get_available_networks(view_mode="All")

    lower, upper = DomainBounds.get_bounds()
    sobol = qmc.Sobol(d=len(lower), scramble=True, seed=BASE_SEED + 123)
    val_configs = jnp.array(
        qmc.scale(sobol.random(args.n_configs), np.asarray(lower), np.asarray(upper)),
        dtype=jnp.float32,
    )
    print(
        f"{'network':<50} {'loss':>10} {'lr_max':>8} {'nff':>4} "
        f"{'lbfgs':>6} {'median':>9} {'mean':>9} {'p95':>9} {'p05':>9} "
        f"{'core_med':>9} {'core_avg':>9} {'core_p95':>9} {'core_p05':>9} "
        f"{'edge_p95':>9} {'bnd_leak':>9}"
    )
    for name in names:
        try:
            run_dir = resolve_run_directory(name)
        except FileNotFoundError:
            print(f"  skip {name}: run dir not found")
            continue
        config_path = run_dir / "config.json"
        flax_path = run_dir / "network.flax"
        if not config_path.exists() or not flax_path.exists():
            print(f"  skip {name}: missing config or network.flax")
            continue
        hp = HyperParams.from_json(str(config_path))
        manager = NetworkManager(hp)
        loaded = manager.from_disk(pinn_path=flax_path)
        manager.state = manager.state.replace(params=loaded)
        inputs = manager.sampler.sample_flux_input(plasma_configs=val_configs)
        configs = [inputs.config[i] for i in range(args.n_configs)]
        kpis = evaluate_plasma_kpis(
            manager, configs, sample_size=args.n_points, core_rho=args.core_rho
        )
        loss_label = f"huber={hp.huber_delta:g}" if hp.huber_delta > 0 else "mse"
        print(
            f"{name:<50} {loss_label:>10} {hp.learning_rate_max:>8.1e} "
            f"{hp.n_fourier_features:>4d} {hp.lbfgs_steps:>6d} "
            + " ".join(f"{value:>9.4f}" for value in kpis.values())
        )

        # kpis.json: valid JSON, one key per line (indent=2) for terminal readability.
        # Overwritten each run (latest eval wins); the eval params are included
        # so the file is self-describing without needing config.json alongside.
        record = build_kpi_record(
            manager, kpis, args.n_configs, args.n_points, args.core_rho, network_name=name
        )
        (run_dir / "kpis.json").write_text(json.dumps(record, indent=2) + "\n")

        if args.no_plots:
            continue
        csv_path = run_dir / "training.csv"
        if csv_path.exists():
            plot_training_curves(csv_path, run_dir / "training_curves.png", title=name)
        grids = evaluate_plasma_grids(
            manager,
            configs,
            resolution=args.plot_resolution,
            quantities=(args.plot_quantity,),
        )
        title = name
        if args.plot_title:
            title = f"{args.plot_title}: {title}"
        plot_plasma_grid_montage(
            grids,
            run_dir / f"{args.plot_quantity}.png",
            quantity=args.plot_quantity,
            title=title,
            metadata=hp.to_dict(),
            display_parameters=args.plot_parameters,
            kpis=kpis,
        )
