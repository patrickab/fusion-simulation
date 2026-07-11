"""Checkpoint sampling, flux/residual grids, and B-field grids for the Network view."""

import jax.numpy as jnp
import numpy as np

from src.engine.model_evaluation import (
    DEFAULT_KPI_SAMPLE_SIZE,
    GridQuantity,
    PlasmaGridBatch,
    evaluate_plasma_grids,
    evaluate_plasma_kpis,
)
from src.engine.network import NetworkManager, Sampler
from src.engine.physics import get_b_field_cartesian
from src.engine.plasma import boundary_normalized_radius
from src.lib.geometry_config import PlasmaGeometry, PlasmaState
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


def build_sample_response(
    manager: NetworkManager,
    seed: int,
    sample_size: int,
    kpi_sample_size: int = DEFAULT_KPI_SAMPLE_SIZE,
) -> dict:
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

    metrics = evaluate_plasma_kpis(manager, configs, sample_size=kpi_sample_size)

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


def build_kpis(
    manager: NetworkManager,
    seed: int,
    sample_size: int,
    kpi_sample_size: int = DEFAULT_KPI_SAMPLE_SIZE,
) -> dict[str, float]:
    data, _, _ = _seeded_samples(manager, seed, sample_size)
    configs = [to_plasma_config(d["geom"], d["state"]) for d in data]
    return evaluate_plasma_kpis(manager, configs, sample_size=kpi_sample_size)


def _serialize_grid_batch(grids: PlasmaGridBatch, quantity: GridQuantity) -> list[dict]:
    """Convert shared JAX grid data into the frontend's JSON contract."""
    return [
        {
            "theta": np.asarray(grids.theta).tolist(),
            "rho": np.asarray(grids.rho).tolist(),
            "R": np.asarray(grids.R[i]).tolist(),
            "Z": np.asarray(grids.Z[i]).tolist(),
            "values": np.asarray(grids.values[quantity][i]).tolist(),
            "boundary_R": np.asarray(grids.boundary_R[i]).tolist(),
            "boundary_Z": np.asarray(grids.boundary_Z[i]).tolist(),
        }
        for i in range(grids.R.shape[0])
    ]


def build_plasma_grids(
    manager: NetworkManager,
    seed: int,
    sample_size: int,
    resolution: int,
    quantities: tuple[GridQuantity, ...],
) -> dict[GridQuantity, list[dict]]:
    """Evaluate requested quantities once and serialize them for API consumers."""
    data, _, _ = _seeded_samples(manager, seed, sample_size)
    configs = [to_plasma_config(d["geom"], d["state"]) for d in data]
    grids = evaluate_plasma_grids(manager, configs, resolution, quantities)
    return {quantity: _serialize_grid_batch(grids, quantity) for quantity in quantities}


def build_flux_grids(
    manager: NetworkManager, seed: int, sample_size: int, resolution: int
) -> list[dict]:
    return build_plasma_grids(manager, seed, sample_size, resolution, ("flux",))["flux"]


def build_residual_grids(
    manager: NetworkManager, seed: int, sample_size: int, resolution: int
) -> list[dict]:
    return build_plasma_grids(manager, seed, sample_size, resolution, ("residual",))["residual"]


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
