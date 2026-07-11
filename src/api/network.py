"""Checkpoint sampling, flux/residual grids, and B-field grids for the Network view."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from src.engine.model_evaluation import compute_gs_residual_on_points
from src.engine.network import NetworkManager, Sampler
from src.engine.physics import get_b_field_cartesian
from src.engine.plasma import boundary_normalized_radius, is_point_in_plasma
from src.lib.geometry_config import CylindricalCoordinates, PlasmaGeometry, PlasmaState
from src.lib.network_config import FluxInput
from src.streamlit.network_utils import to_plasma_config


def _seeded_samples(
    manager: NetworkManager, seed: int, sample_size: int
) -> tuple[list[dict], PlasmaGeometry, PlasmaState]:
    """Re-derive the same deterministic seeded samples as the Streamlit `reseed_*` flow."""
    sampler = Sampler(manager.config, seed=seed)

    seeded_geometry_configs = sampler._get_sobol_sample(
        n_samples=sample_size,
        lower_bounds=sampler._domain_lower_bounds,
        upper_bounds=sampler._domain_upper_bounds,
        sobol_sampler="domain",
    )
    seeded_train_set = sampler._get_sobol_sample(
        n_samples=manager.config.n_train,
        lower_bounds=sampler._domain_lower_bounds,
        upper_bounds=sampler._domain_upper_bounds,
        sobol_sampler="domain",
    )
    sample_3d = seeded_train_set[seed % manager.config.n_train]

    geom_3d = PlasmaGeometry(
        R0=float(sample_3d[0]),
        a=float(sample_3d[1]),
        kappa=float(sample_3d[2]),
        delta=float(sample_3d[3]),
    )
    state_3d = PlasmaState(
        p0=float(sample_3d[4]),
        F_axis=float(sample_3d[5]),
        pressure_alpha=float(sample_3d[6]),
        field_exponent=float(sample_3d[7]),
    )

    flux_input = sampler.sample_flux_input(plasma_configs=seeded_geometry_configs)
    data = [
        {
            "geom": flux_input.config[i].Geometry,
            "state": flux_input.config[i].State,
            "bR": flux_input.config[i].Boundary.R,
            "bZ": flux_input.config[i].Boundary.Z,
            "iR": flux_input.R_sample[i],
            "iZ": flux_input.Z_sample[i],
        }
        for i in range(sample_size)
    ]
    return data, geom_3d, state_3d


def build_sample_response(manager: NetworkManager, seed: int, sample_size: int) -> dict:
    data, geom_3d, state_3d = _seeded_samples(manager, seed, sample_size)

    configs = [to_plasma_config(d["geom"], d["state"]) for d in data]

    samples = [
        {
            "R0": float(d["geom"].R0),
            "a": float(d["geom"].a),
            "kappa": float(d["geom"].kappa),
            "delta": float(d["geom"].delta),
            "p0": float(d["state"].p0),
            "F_axis": float(d["state"].F_axis),
            "pressure_alpha": float(d["state"].pressure_alpha),
            "field_exponent": float(d["state"].field_exponent),
            "boundary_R": np.asarray(d["bR"]).tolist(),
            "boundary_Z": np.asarray(d["bZ"]).tolist(),
            "interior_R": np.asarray(d["iR"]).tolist(),
            "interior_Z": np.asarray(d["iZ"]).tolist(),
        }
        for d in data
    ]

    metrics = _compute_metrics(manager, data, configs)

    return {
        "samples": samples,
        "metrics": metrics,
        "geom3d": {
            "R0": geom_3d.R0,
            "a": geom_3d.a,
            "kappa": geom_3d.kappa,
            "delta": geom_3d.delta,
        },
        "state3d": {
            "p0": state_3d.p0,
            "F_axis": state_3d.F_axis,
            "pressure_alpha": state_3d.pressure_alpha,
            "field_exponent": state_3d.field_exponent,
        },
    }


def _compute_metrics(manager: NetworkManager, data: list[dict], configs: list) -> dict:
    geometry = jnp_tree_stack([c.Geometry for c in configs])
    boundary = jnp_tree_stack([c.Boundary for c in configs])
    state = jnp_tree_stack([c.State for c in configs])
    batched_config = configs[0].__class__(Geometry=geometry, Boundary=boundary, State=state)

    flux_input = FluxInput(
        R_sample=jnp.stack([d["iR"] for d in data]),
        Z_sample=jnp.stack([d["iZ"] for d in data]),
        config=batched_config,
    )

    total, l_res, l_dir, l_per_cfg = manager.eval_step(
        manager.state, flux_input, manager.config.weight_boundary_condition
    )

    max_res = 0.0
    for i, cfg in enumerate(configs):
        res_vals = compute_gs_residual_on_points(
            manager, cfg, flux_input.R_sample[i], flux_input.Z_sample[i]
        )
        max_res = max(max_res, float(jnp.max(jnp.abs(res_vals))))

    return {
        "avg_loss": float(total),
        "interior_loss": float(l_res),
        "boundary_loss": float(l_dir),
        "max_loss": float(jnp.max(l_per_cfg)),
        "max_residual": max_res,
    }


def jnp_tree_stack(items: list) -> object:
    """Stack a list of identically-shaped flax dataclasses along a new batch axis 0."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *items)


def _grid_bounds(geoms: list[PlasmaGeometry]) -> tuple[list[float], list[float]]:
    r_min = min(g.R0 - g.a * 1.2 for g in geoms)
    r_max = max(g.R0 + g.a * 1.2 for g in geoms)
    z_max = max(g.kappa * g.a * 1.2 for g in geoms)
    r_mid, extent = (r_min + r_max) / 2, max(r_max - r_min, 2 * z_max)
    return [r_mid - extent / 2, r_mid + extent / 2], [-extent / 2, extent / 2]


def build_flux_grids(
    manager: NetworkManager, seed: int, sample_size: int, resolution: int
) -> list[dict]:
    data, _, _ = _seeded_samples(manager, seed, sample_size)
    configs = [to_plasma_config(d["geom"], d["state"]) for d in data]

    r_lims, z_lims = _grid_bounds([c.Geometry for c in configs])
    R = jnp.linspace(*r_lims, resolution)
    Z = jnp.linspace(*z_lims, resolution)
    R_grid, Z_grid = jnp.meshgrid(R, Z)
    coords_flat = CylindricalCoordinates(
        R=R_grid.flatten(), Z=Z_grid.flatten(), phi=jnp.zeros_like(R_grid.flatten())
    )

    grids = []
    for cfg in configs:
        mask = is_point_in_plasma(coords_flat, cfg.Boundary)
        psi = jnp.full(mask.shape, jnp.nan)
        if mask.any():
            val = manager.get_psi(coords_flat.R[mask], coords_flat.Z[mask], cfg)
            psi = psi.at[mask].set(val.flatten())

        grids.append(
            {
                "R": np.asarray(R).tolist(),
                "Z": np.asarray(Z).tolist(),
                "values": _nan_to_none_grid(np.asarray(psi).reshape(resolution, resolution)),
                "boundary_R": np.asarray(cfg.Boundary.R).tolist(),
                "boundary_Z": np.asarray(cfg.Boundary.Z).tolist(),
            }
        )
    return grids


def build_residual_grids(
    manager: NetworkManager, seed: int, sample_size: int, resolution: int
) -> list[dict]:
    data, _, _ = _seeded_samples(manager, seed, sample_size)
    configs = [to_plasma_config(d["geom"], d["state"]) for d in data]

    r_lims, z_lims = _grid_bounds([c.Geometry for c in configs])
    R = jnp.linspace(*r_lims, resolution)
    Z = jnp.linspace(*z_lims, resolution)
    R_grid, Z_grid = jnp.meshgrid(R, Z)
    coords_flat = CylindricalCoordinates(
        R=R_grid.flatten(), Z=Z_grid.flatten(), phi=jnp.zeros_like(R_grid.flatten())
    )

    grids = []
    for cfg in configs:
        mask = is_point_in_plasma(coords_flat, cfg.Boundary)
        residual = jnp.full(mask.shape, jnp.nan)
        if mask.any():
            R_masked = coords_flat.R[mask]
            Z_masked = coords_flat.Z[mask]
            res_vals = compute_gs_residual_on_points(manager, cfg, R_masked, Z_masked)
            residual = residual.at[mask].set(res_vals.flatten())

        grids.append(
            {
                "R": np.asarray(R).tolist(),
                "Z": np.asarray(Z).tolist(),
                "values": _nan_to_none_grid(np.asarray(residual).reshape(resolution, resolution)),
                "boundary_R": np.asarray(cfg.Boundary.R).tolist(),
                "boundary_Z": np.asarray(cfg.Boundary.Z).tolist(),
            }
        )
    return grids


def _nan_to_none_grid(grid: np.ndarray) -> list[list[float | None]]:
    return [[None if math.isnan(v) else float(v) for v in row] for row in grid]


def build_bfield_grid(
    manager: NetworkManager, seed: int, sample_size: int, n_lines: int = 24
) -> dict:
    _, geom_3d, state_3d = _seeded_samples(manager, seed, sample_size)
    config = to_plasma_config(geom_3d, state_3d)

    R0, a, kappa = float(geom_3d.R0), float(geom_3d.a), float(geom_3d.kappa)
    # Tight box: field lines live inside the plasma, so padding is wasted resolution.
    extent = R0 + a + 0.5
    z_extent = (a * kappa) + 0.5

    # 48³ keeps the tracer's psi-drift (trilinear interpolation error, O(h²))
    # at the few-percent level; 30³ drifted ~10% of the axis flux per line.
    nx = ny = nz = 48
    xs = np.linspace(-extent, extent, nx)
    ys = np.linspace(-extent, extent, ny)
    zs = np.linspace(-z_extent, z_extent, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    vectors = get_b_field_cartesian(
        manager.make_psi_fn(),
        manager.state.params,
        jnp.array(X.flatten()),
        jnp.array(Y.flatten()),
        jnp.array(Z.flatten()),
        config,
    )
    vectors = np.array(vectors)  # copy: asarray on a jax array is read-only

    # Outside the boundary psi is meaningless extrapolation, and the hard-BC
    # envelope makes |B| spike ~7x there, bleeding into edge cells through the
    # tracer's trilinear interpolation. Zeroing it keeps edge cells clean and
    # terminates streamlines at the plasma edge (|B| < threshold stops the JS
    # integrator). 1.05 margin so boundary-straddling cells keep their inside nodes.
    rho = boundary_normalized_radius(
        jnp.array(np.sqrt(X.flatten() ** 2 + Y.flatten() ** 2)),
        jnp.array(Z.flatten()),
        config.Boundary,
    )
    vectors[np.asarray(rho) > 1.05] = 0.0

    seed_points = np.stack(
        [np.linspace(R0 - a * 0.9, R0 + a * 0.9, n_lines), np.zeros(n_lines), np.zeros(n_lines)],
        axis=-1,
    )

    return {
        "grid": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "origin": [-extent, -extent, -z_extent],
            "spacing": [2 * extent / (nx - 1), 2 * extent / (ny - 1), 2 * z_extent / (nz - 1)],
        },
        "vectors": vectors.flatten().tolist(),
        "seed_points": seed_points.tolist(),
    }
