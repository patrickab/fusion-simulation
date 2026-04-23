import numpy as np
import pyvista as pv

from src.engine.network import NetworkManager, Sampler
from src.lib.geometry_config import PlasmaGeometry, ToroidalCoilConfig
from src.lib.geometry_config import PlasmaState
import streamlit as st

default_coil_config = ToroidalCoilConfig(
    distance_from_plasma=1.5,
    radial_thickness=0.8,
    vertical_thickness=0.2,
    angular_span=6,
    n_field_coils=8,
)


def reactor_config_sidebar() -> tuple[PlasmaGeometry, ToroidalCoilConfig]:
    """Create a sidebar form for reactor geometry configuration.

    Returns:
        tuple: (PlasmaGeometry, ToroidalCoilConfig)
    """
    with st.sidebar, st.form("geometry_coil_form"):
        st.header("Plasma Geometry")
        R0 = st.slider("Major Radius (R0) [m]", 3.0, 10.0, 6.2, 0.1)
        a = st.slider("Minor Radius (a) [m]", 1.0, 5.0, 3.2, 0.1)
        kappa = st.slider("Elongation (kappa)", 1.0, 3.0, 1.7, 0.1)
        delta = st.slider("Triangularity (delta)", 0.0, 1.0, 0.33, 0.01)

        # st.header("Coil Configuration")
        # dist = st.slider("Distance from Plasma [m]", 0.5, 3.0, 1.5, 0.1)
        # r_thick = st.slider("Radial Thickness [m]", 0.1, 2.0, 0.8, 0.1)
        # v_thick = st.slider("Vertical Thickness [m]", 0.1, 2.0, 0.2, 0.1)
        # span = st.slider("Angular Span [deg]", 1, 20, 6, 1)
        # n_coils = st.slider("Number of Coils", 4, 24, 8, 1)

        st.form_submit_button("Apply")

    plasma_geometry = PlasmaGeometry(R0=R0, a=a, kappa=kappa, delta=delta)
    return plasma_geometry, default_coil_config


def generate_field_lines(
    plasma_mesh: pv.PolyData, n_lines: int, B_field: np.ndarray | None = None, q_pitch: float = 2.5
) -> pv.PolyData:
    """Generate 3D magnetic field lines on a plasma surface.

    Args:
        plasma_mesh: Surface mesh to trace lines on
        n_lines: Number of field lines to generate
        B_field: Pre-calculated magnetic field vectors (N, 3). If None, uses geometric pitch.
        q_pitch: Safety factor (pitch) of field lines (only used if B_field is None)

    Returns:
        pv.PolyData: The generated streamlines
    """
    if B_field is not None:
        plasma_mesh["B"] = B_field
    else:
        # Fallback to geometric pitch if no physical B field provided
        # Ensure point-aligned normals for vector operations at vertices
        plasma_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

        pts = plasma_mesh.points
        ns = plasma_mesh.point_data["Normals"]
        Rs = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)

        # Unit toroidal vector e_phi
        e_phi = np.column_stack([-pts[:, 1] / Rs, pts[:, 0] / Rs, np.zeros_like(pts[:, 0])])

        # Poloidal tangent B_pol = e_phi x n
        B_pol = np.cross(e_phi, ns)

        # Total magnetic field vector field
        B_total = B_pol + q_pitch * e_phi
        plasma_mesh["B"] = B_total

    # Seed Field Lines choose seed points on the surface
    seed_indices = np.linspace(0, plasma_mesh.n_points - 1, n_lines, dtype=int)
    seeds = plasma_mesh.points[seed_indices]

    return plasma_mesh.streamlines_from_source(
        pv.PolyData(seeds),
        vectors="B",
        integration_direction="both",
        max_time=100.0,
        initial_step_length=0.05,
    )


def reseed_network_visualisation() -> None:
    """Reseed deterministic samples used by the network visualisation page."""
    manager: NetworkManager = st.session_state.manager
    sampler = Sampler(manager.config, seed=int(st.session_state.seed))

    seeded_geometry_configs = sampler._get_sobol_sample(
        n_samples=4,
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
    sample_3d = seeded_train_set[st.session_state.seed]

    geom_3d = PlasmaGeometry(
        R0=sample_3d[0],
        a=sample_3d[1],
        kappa=sample_3d[2],
        delta=sample_3d[3],
    )
    state_3d = PlasmaState(
        p0=sample_3d[4],
        F_axis=sample_3d[5],
        pressure_alpha=sample_3d[6],
        field_exponent=sample_3d[7],
    )

    flux_input = sampler.sample_flux_input(plasma_configs=seeded_geometry_configs)
    seeded_geometry_data = [
        {
            "geom": flux_input.config[i].Geometry,
            "state": flux_input.config[i].State,
            "bR": flux_input.config[i].Boundary.R,
            "bZ": flux_input.config[i].Boundary.Z,
            "iR": flux_input.R_sample[i],
            "iZ": flux_input.Z_sample[i],
        }
        for i in range(len(seeded_geometry_configs))
    ]

    st.session_state.seeded_flux_input = flux_input
    st.session_state.seeded_geometry_data = seeded_geometry_data
    st.session_state.seeded_3d_geom = geom_3d
    st.session_state.seeded_3d_state = state_3d
