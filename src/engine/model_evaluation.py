from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from src.engine.network import BASE_SEED, NetworkManager
from src.engine.physics import grad_shafranov_residual
from src.engine.plasma import get_poloidal_points
from src.lib.geometry_config import PlasmaConfig

GridQuantity = Literal["flux", "residual"]
DEFAULT_KPI_SAMPLE_SIZE = 16_384


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


def compute_gs_residual_on_points(
    manager: NetworkManager,
    config: PlasmaConfig,
    R_pts: jnp.ndarray,
    Z_pts: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate normalised Grad-Shafranov residual at supplied ``(R, Z)`` points."""
    psi_fn = manager.make_psi_fn()
    params = manager.state.params

    psi_vals = jax.vmap(lambda r, z: psi_fn(params, r, z, config))(R_pts, Z_pts)
    n_axis = max(1, psi_vals.shape[0] // 20)
    psi_axis = jax.lax.stop_gradient(jnp.mean(jnp.sort(psi_vals)[:n_axis]))

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
        n_axis = max(1, sample_size // 20)
        psi_axis = jax.lax.stop_gradient(jnp.mean(jnp.sort(psi)[:n_axis]))
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

    def fields(R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> tuple[jnp.ndarray, ...]:
        psi = jax.vmap(lambda r, z: psi_fn(params, r, z, cfg))(R, Z)
        output = []
        if "flux" in requested:
            output.append(psi)
        if "residual" in requested:
            n_axis = max(1, R.shape[0] // 20)
            psi_axis = jax.lax.stop_gradient(jnp.mean(jnp.sort(psi)[:n_axis]))
            residual = jax.vmap(
                lambda r, z: grad_shafranov_residual(psi_fn, params, r, z, psi_axis, cfg)
            )(R, Z)
            output.append(jnp.log10(jnp.abs(residual) + 1e-6))
        return tuple(output)

    evaluated = jax.vmap(fields)(R_flat, Z_flat, batched_config)
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
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    values = np.asarray(grids.values[quantity])
    color = {
        "flux": {"cmap": "viridis", "vmin": 0.0, "vmax": 90.0, "label": r"$\psi$"},
        "residual": {
            "cmap": "magma",
            "vmin": -2.0,
            "vmax": 1.0,
            "label": r"$\log_{10}|R_{GS}|$",
        },
    }[quantity]
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
    kpi_items = [f"{key}={value:.3e}" for key, value in (kpis or {}).items()]
    kpi_text = "\n".join(", ".join(kpi_items[i : i + 5]) for i in range(0, len(kpi_items), 5))
    heading = "\n".join(part for part in (title, parameter_text, kpi_text) if part)
    if heading:
        fig.suptitle(heading, fontsize=10)
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # Region-split |GS residual| report: selection metric is the plasma core
    # (rho < --core-rho); the edge shell is reported but tolerated by design.
    # Outputs are grouped per commit: <outdir>/<commit>/kpis.csv (appended) plus
    # one <outdir>/<commit>/<run>/ dir per checkpoint holding config.json,
    # training.csv and the montage PNG (fixed log color scale for comparability).
    import argparse
    import csv
    from datetime import datetime
    import shutil

    from src.lib.config import Filepaths
    from src.lib.network_config import DomainBounds, HyperParams
    from src.streamlit.network_utils import extract_commit

    parser = argparse.ArgumentParser(description="Region-split |GS residual| KPIs per checkpoint")
    parser.add_argument("networks", nargs="*", help="Checkpoint names (default: all *.flax)")
    parser.add_argument("--n-configs", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=DEFAULT_KPI_SAMPLE_SIZE)
    parser.add_argument("--core-rho", type=float, default=0.85)
    parser.add_argument("--outdir", type=str, default=str(Filepaths.BENCHMARKS))
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = args.networks or sorted(p.name for p in Filepaths.NETWORKS.glob("*.flax"))

    lower, upper = DomainBounds.get_bounds()
    sobol = qmc.Sobol(d=len(lower), scramble=True, seed=BASE_SEED + 123)
    val_configs = jnp.array(
        qmc.scale(sobol.random(args.n_configs), np.asarray(lower), np.asarray(upper)),
        dtype=jnp.float32,
    )
    print(
        f"{'network':<42} {'loss':>10} {'lr_max':>8} {'nff':>4} "
        f"{'lbfgs':>6} {'median':>9} {'mean':>9} {'p95':>9} {'p05':>9} "
        f"{'core_med':>9} {'core_avg':>9} {'core_p95':>9} {'core_p05':>9} "
        f"{'edge_p95':>9} {'bnd_leak':>9}"
    )
    for name in names:
        run_dir = outdir / (extract_commit(name) or "no_git") / Path(name).stem
        run_dir.mkdir(parents=True, exist_ok=True)
        hp = HyperParams.from_json(str((Filepaths.NETWORKS / name).with_suffix(".json")))
        shutil.copyfile((Filepaths.NETWORKS / name).with_suffix(".json"), run_dir / "config.json")
        training_csv = (Filepaths.NETWORKS / name).with_suffix(".csv")
        if training_csv.exists():
            shutil.copyfile(training_csv, run_dir / "training.csv")
        manager = NetworkManager(hp)
        loaded = manager.from_disk(pinn_path=Filepaths.NETWORKS / name)
        manager.state = manager.state.replace(params=loaded)
        inputs = manager.sampler.sample_flux_input(plasma_configs=val_configs)
        configs = [inputs.config[i] for i in range(args.n_configs)]
        kpis = evaluate_plasma_kpis(
            manager, configs, sample_size=args.n_points, core_rho=args.core_rho
        )
        loss_label = f"huber={hp.huber_delta:g}" if hp.huber_delta > 0 else "mse"
        print(
            f"{name:<42} {loss_label:>10} {hp.learning_rate_max:>8.1e} "
            f"{hp.n_fourier_features:>4d} {hp.lbfgs_steps:>6d} "
            + " ".join(f"{value:>9.4f}" for value in kpis.values())
        )

        csv_path = run_dir.parent / "kpis.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "date",
                        "network",
                        "loss",
                        "lr_max",
                        "n_fourier_features",
                        "lbfgs_steps",
                        "n_configs",
                        "n_points",
                        "core_rho",
                        *kpis.keys(),
                    ]
                )
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    name,
                    loss_label,
                    hp.learning_rate_max,
                    hp.n_fourier_features,
                    hp.lbfgs_steps,
                    args.n_configs,
                    args.n_points,
                    args.core_rho,
                    *(f"{v:.5f}" for v in kpis.values()),
                ]
            )

        if args.no_plots:
            continue
        grids = evaluate_plasma_grids(
            manager,
            configs,
            resolution=args.plot_resolution,
            quantities=(args.plot_quantity,),
        )
        title = Path(name).stem
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
