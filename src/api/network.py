"""Checkpoint sampling, flux/residual grids, and B-field grids for the Network view."""

import base64

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from src.engine.model_evaluation import (
    GridQuantity,
    PlasmaGridBatch,
    evaluate_plasma_grids,
    shared_psi_fn,
)
from src.engine.network import Sampler
from src.engine.network_manager import NetworkManager
from src.engine.physics import get_b_field, trace_field_lines
from src.engine.plasma import get_poloidal_points
from src.lib.geometry_config import PlasmaGeometry, PlasmaState, RotationalAngles
from src.lib.network_config import DomainBounds
from src.streamlit.network_utils import to_plasma_config

# Field-line tracing. Seeds pad to MAX_FIELD_LINES so the tracer compiles once
# per process; lines run FIELD_LINE_TURNS toroidal transits per direction.
MAX_FIELD_LINES = 64
FIELD_GRID_NODES = 64  # (R, Z) B-grid nodes per axis
FIELD_LINE_TURNS = 12
VERTICES_PER_TURN = 48

_b_field = jax.jit(get_b_field, static_argnums=(0,))


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


def _f32_base64(arr: np.ndarray) -> str:
    """Little-endian float32 bytes as base64: decodes straight into a Float32Array."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def build_field_lines(
    manager: NetworkManager, seed: int, sample_size: int, n_lines: int = 24
) -> dict:
    """Trace field lines server-side and ship finished polylines plus the
    poloidal boundary; the client only fills buffers and revolves the boundary
    into the plasma wireframe."""
    _, geom_3d, state_3d = _seeded_configs(manager, seed, sample_size)
    config = to_plasma_config(geom_3d, state_3d)
    R0, a, kappa = float(geom_3d.R0), float(geom_3d.a), float(geom_3d.kappa)

    # Axisymmetric field: B depends on (R, Z) only, so a 2D grid over the
    # plasma's bounding box (margin keeps boundary cells' outer nodes) is exact
    # in phi. 64^2 network evaluations instead of a 48^3 Cartesian grid.
    margin = 0.05 * a
    origin = np.array([R0 - a - margin, -(kappa * a + margin)])
    spacing = 2 * np.array([a + margin, kappa * a + margin]) / (FIELD_GRID_NODES - 1)
    R_nodes = origin[0] + spacing[0] * np.arange(FIELD_GRID_NODES)
    Z_nodes = origin[1] + spacing[1] * np.arange(FIELD_GRID_NODES)
    R_mesh, Z_mesh = np.meshgrid(R_nodes, Z_nodes, indexing="ij")
    B = _b_field(
        shared_psi_fn(manager),
        manager.state.params,
        jnp.asarray(R_mesh.ravel(), dtype=jnp.float32),
        jnp.asarray(Z_mesh.ravel(), dtype=jnp.float32),
        config,
    )
    B_grid = B.T.reshape(3, FIELD_GRID_NODES, FIELD_GRID_NODES)

    # Midplane seeds across the plasma, as in the legacy Streamlit render.
    R_seed = np.linspace(R0 - 0.9 * a, R0 + 0.9 * a, n_lines)
    R_seed = np.pad(R_seed, (0, MAX_FIELD_LINES - n_lines), mode="edge")
    # Arc length per vertex = one RK4 step; drift is set by the B grid, not the step.
    vertex_spacing = 2 * np.pi * R0 / VERTICES_PER_TURN
    xyz, b_mag, n_valid = trace_field_lines(
        B_grid,
        jnp.asarray(origin, dtype=jnp.float32),
        jnp.asarray(spacing, dtype=jnp.float32),
        jnp.asarray(R_seed, dtype=jnp.float32),
        jnp.zeros(MAX_FIELD_LINES, dtype=jnp.float32),
        config.Boundary,
        vertex_spacing,
        n_vertices=FIELD_LINE_TURNS * VERTICES_PER_TURN + 1,
    )
    xyz, b_mag, n_valid = np.asarray(xyz), np.asarray(b_mag), np.asarray(n_valid)

    # One polyline per seed: backward branch reversed, then forward (seed shared).
    point_chunks: list[np.ndarray] = []
    speed_chunks: list[np.ndarray] = []
    lengths: list[int] = []
    for i in range(n_lines):
        n_fwd, n_bwd = int(n_valid[0, i]), int(n_valid[1, i])
        if n_fwd + n_bwd < 3:  # seed outside the plasma, or both branches exit at once
            continue
        point_chunks.append(np.concatenate([xyz[1, i, :n_bwd][::-1], xyz[0, i, 1:n_fwd]]))
        speed_chunks.append(np.concatenate([b_mag[1, i, :n_bwd][::-1], b_mag[0, i, 1:n_fwd]]))
        lengths.append(n_fwd + n_bwd - 1)

    points = np.concatenate(point_chunks) if point_chunks else np.empty((0, 3))
    speeds = np.concatenate(speed_chunks) if speed_chunks else np.empty(0)
    b_range = [float(speeds.min()), float(speeds.max())] if speeds.size else [0.0, 1.0]
    boundary_R, boundary_Z = get_poloidal_points(RotationalAngles.THETA, geom_3d)

    return {
        "points": _f32_base64(points),
        "speeds": _f32_base64(speeds),
        "line_lengths": lengths,
        "b_range": b_range,
        "boundary": {"R": _to_list(boundary_R), "Z": _to_list(boundary_Z)},
    }
