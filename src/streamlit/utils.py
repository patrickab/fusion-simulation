import numpy as np
import pyvista as pv

from src.lib.geometry_config import PlasmaGeometry, ToroidalCoilConfig
import streamlit as st


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

        st.header("Coil Configuration")
        dist = st.slider("Distance from Plasma [m]", 0.5, 3.0, 1.5, 0.1)
        r_thick = st.slider("Radial Thickness [m]", 0.1, 2.0, 0.8, 0.1)
        v_thick = st.slider("Vertical Thickness [m]", 0.1, 2.0, 0.2, 0.1)
        span = st.slider("Angular Span [deg]", 1, 20, 6, 1)
        n_coils = st.slider("Number of Coils", 4, 24, 8, 1)

        st.form_submit_button("Apply")

    plasma_geometry = PlasmaGeometry(R0=R0, a=a, kappa=kappa, delta=delta)
    toroid_coil_config = ToroidalCoilConfig(
        distance_from_plasma=dist,
        radial_thickness=r_thick,
        vertical_thickness=v_thick,
        angular_span=span,
        n_field_coils=n_coils,
    )
    return plasma_geometry, toroid_coil_config


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
