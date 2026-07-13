"""Checkpoint sampling, flux/residual grids, and B-field grids for the Network view."""

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from scipy.stats import qmc

from src.engine.model_evaluation import (
    GridQuantity,
    PlasmaGridBatch,
    evaluate_plasma_grids,
)
from src.engine.network import NetworkManager, Sampler
from src.engine.physics import get_b_field_cartesian
from src.engine.plasma import boundary_normalized_radius
from src.lib.geometry_config import PlasmaGeometry, PlasmaState
from src.lib.network_config import DomainBounds
from src.streamlit.network_utils import to_plasma_config


def _split_row(row: jnp.ndarray) -> tuple[PlasmaGeometry, PlasmaState]:
    """Domain Sobol row -> (geometry, state); columns as in Sampler._compute_single_config."""
    geom = PlasmaGeometry(
        R0=float(row[0]), a=float(row[1]), kappa=float(row[2]), delta=float(row[3])
    )
    state = PlasmaState(
        p0=float(row[4]),
        F_axis=float(row[5]),
        pressure_alpha=float(row[6]),
        field_exponent=float(row[7]),
    )
    return geom, state


def _seeded_domain_draw(
    manager: NetworkManager, seed: int, sample_size: int
) -> tuple[jnp.ndarray, PlasmaGeometry, PlasmaState]:
    """Single owner of the seeded domain Sobol contract: engine construction
    (identical to Sampler._sobol_domain), scaling, and the 3D-config pick —
    row (seed % n_train) of the train-set draw that historically followed,
    reached via fast_forward instead of generating the other n_train-1 rows."""
    lower, upper = DomainBounds.get_bounds()
    sobol = qmc.Sobol(d=len(lower), scramble=True, seed=seed)

    def scale(unit: np.ndarray) -> jnp.ndarray:
        return jnp.array(unit, dtype=jnp.float32) * (upper - lower) + lower

    rows = scale(sobol.random(sample_size))
    if seed % manager.config.n_train:
        sobol.fast_forward(seed % manager.config.n_train)
    geom_3d, state_3d = _split_row(scale(sobol.random(1))[0])
    return rows, geom_3d, state_3d


def _seeded_configs(
    manager: NetworkManager, seed: int, sample_size: int
) -> tuple[list[tuple[PlasmaGeometry, PlasmaState]], PlasmaGeometry, PlasmaState]:
    """Seeded config params only — no Sampler (whose init precomputes thousands
    of collocation coordinates that the grid/fieldlines endpoints never use)."""
    rows, geom_3d, state_3d = _seeded_domain_draw(manager, seed, sample_size)
    return [_split_row(row) for row in rows], geom_3d, state_3d


def _seeded_samples(
    manager: NetworkManager, seed: int, sample_size: int
) -> tuple[list[dict], PlasmaGeometry, PlasmaState]:
    """Re-derive the same deterministic seeded samples as the Streamlit `reseed_*` flow.

    Adds interior collocation points on top of the shared domain draw — the one
    endpoint that justifies constructing a Sampler (its own domain engine goes
    unused; seed+1/+2 engines supply the collocation coordinates).
    """
    rows, geom_3d, state_3d = _seeded_domain_draw(manager, seed, sample_size)

    sampler = Sampler(manager.config, seed=seed)
    flux_input = sampler.sample_flux_input(plasma_configs=rows)
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


def _to_list(arr: object) -> list:
    return np.asarray(arr, dtype=np.float64).tolist()


def build_sample_response(manager: NetworkManager, seed: int, sample_size: int) -> dict:
    data, geom_3d, state_3d = _seeded_samples(manager, seed, sample_size)

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
            "boundary_R": _to_list(d["bR"]),
            "boundary_Z": _to_list(d["bZ"]),
            "interior_R": _to_list(d["iR"]),
            "interior_Z": _to_list(d["iZ"]),
        }
        for d in data
    ]

    return {
        "samples": samples,
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


def _serialize_grid_batch(grids: PlasmaGridBatch, quantity: GridQuantity) -> list[dict]:
    """Convert shared JAX grid data into the frontend's JSON contract."""
    return [
        {
            "theta": _to_list(grids.theta),
            "rho": _to_list(grids.rho),
            "R": _to_list(grids.R[i]),
            "Z": _to_list(grids.Z[i]),
            "values": _to_list(grids.values[quantity][i]),
            "boundary_R": _to_list(grids.boundary_R[i]),
            "boundary_Z": _to_list(grids.boundary_Z[i]),
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
    pairs, _, _ = _seeded_configs(manager, seed, sample_size)
    configs = [to_plasma_config(geom, state) for geom, state in pairs]
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


def build_field_lines(
    manager: NetworkManager, seed: int, sample_size: int, n_lines: int = 24
) -> dict:
    """Trace field lines server-side (VTK RK45, the legacy PyVista path) and
    ship finished polylines — the client only fills a BufferGeometry."""
    _, geom_3d, state_3d = _seeded_configs(manager, seed, sample_size)
    config = to_plasma_config(geom_3d, state_3d)

    R0, a, kappa = float(geom_3d.R0), float(geom_3d.a), float(geom_3d.kappa)
    # Tight box: field lines live inside the plasma, so padding is wasted resolution.
    extent = R0 + a + 0.5
    z_extent = (a * kappa) + 0.5

    # 48³ keeps the tracer's psi-drift (trilinear interpolation error, O(h²))
    # at the few-percent level; 30³ drifted ~10% of the axis flux per line.
    n = 48
    grid = pv.ImageData(
        dimensions=(n, n, n),
        spacing=(2 * extent / (n - 1), 2 * extent / (n - 1), 2 * z_extent / (n - 1)),
        origin=(-extent, -extent, -z_extent),
    )
    pts = grid.points

    vectors = get_b_field_cartesian(
        manager.make_psi_fn(),
        manager.state.params,
        jnp.array(pts[:, 0]),
        jnp.array(pts[:, 1]),
        jnp.array(pts[:, 2]),
        config,
    )
    vectors = np.array(vectors)  # copy: asarray on a jax array is read-only

    # Outside the boundary psi is meaningless extrapolation, and the hard-BC
    # envelope makes |B| spike ~7x there, bleeding into edge cells through
    # trilinear interpolation. Zeroing it keeps edge cells clean and terminates
    # streamlines at the plasma edge (terminal_speed stops the integrator).
    # 1.05 margin so boundary-straddling cells keep their inside nodes.
    rho = boundary_normalized_radius(
        jnp.array(np.hypot(pts[:, 0], pts[:, 1])), jnp.array(pts[:, 2]), config.Boundary
    )
    vectors[np.asarray(rho) > 1.05] = 0.0
    grid["B"] = vectors
    grid.set_active_vectors("B")

    # Midplane seed line, same params as the legacy Streamlit render
    # (src/lib/visualization.py); resolution=k yields k+1 seed points.
    seed_line = pv.Line(
        pointa=(R0 - a * 0.9, 0, 0), pointb=(R0 + a * 0.9, 0, 0), resolution=n_lines - 1
    )
    streamlines = grid.streamlines_from_source(
        seed_line,
        integration_direction="both",
        max_length=1000.0,
        max_steps=2000,
        integrator_type=45,
        compute_vorticity=False,
    )

    line_points = np.asarray(streamlines.points)
    speeds_all = (
        np.linalg.norm(np.asarray(streamlines["B"]), axis=1)
        if streamlines.n_points
        else np.empty(0)
    )

    # VTK cell array: [n, id0..id{n-1}, m, ...] — gather per line, drop stubs.
    # Vertices are decimated for transport only (integration stays exact):
    # RK45 steps ~0.05 m here, 4x that is still finer than the retired JS
    # tracer's 0.8 m max step, and ~4x less JSON.
    stride = 4
    conn = np.asarray(streamlines.lines)
    chunks: list[np.ndarray] = []
    lengths: list[int] = []
    i = 0
    while i < len(conn):
        n_pts = int(conn[i])
        ids = conn[i + 1 : i + 1 + n_pts]
        i += 1 + n_pts
        if n_pts < 8:  # stubs from seeds in near-zero field
            continue
        keep = ids[::stride]
        if (n_pts - 1) % stride:  # always keep the true endpoint
            keep = np.append(keep, ids[-1])
        chunks.append(keep)
        lengths.append(len(keep))

    if chunks:
        ids = np.concatenate(chunks)
        points = line_points[ids]
        speeds = speeds_all[ids]
        b_range = [float(speeds.min()), float(speeds.max())]
    else:
        points = np.empty((0, 3))
        speeds = np.empty(0)
        b_range = [0.0, 1.0]

    return {
        "points": _to_list(points.flatten()),
        "speeds": _to_list(speeds),
        "line_lengths": lengths,
        "b_range": b_range,
    }
