from stpyvista import stpyvista

from src.engine.plasma import calculate_fusion_plasma, calculate_poloidal_boundary
from src.lib.geometry_config import PlasmaGeometry, RotationalAngles, ToroidalCoilConfig
from src.lib.visualization import (
    calculate_2d_geometry,
    initialize_plotter,
    render_fusion_plasma,
    render_plasma_boundary,
)
from src.toroidal_geometry import generate_toroidal_coils_3d
import streamlit as st

st.title("Reactor Geometry")

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

    submitted = st.form_submit_button("Apply")

# Construct configuration objects
plasma_geometry = PlasmaGeometry(R0=R0, a=a, kappa=kappa, delta=delta)
toroid_coil_config = ToroidalCoilConfig(
    distance_from_plasma=dist,
    radial_thickness=r_thick,
    vertical_thickness=v_thick,
    angular_span=span,
    n_field_coils=n_coils,
)

# Generate and display the plot
with st.spinner("Calculating geometry..."):
    # Calculate 2D geometry
    theta = RotationalAngles.THETA
    plasma_boundary = calculate_poloidal_boundary(theta=theta, plasma_geometry=plasma_geometry)
    toroidal_coil_2d = calculate_2d_geometry(
        plasma_geometry=plasma_geometry, toroid_coil_config=toroid_coil_config
    )[1]  # Get only the coil 2D

    # Calculate 3D geometry
    fusion_plasma = calculate_fusion_plasma(plasma_boundary=plasma_boundary)
    toroidal_coils_3d = generate_toroidal_coils_3d(
        toroidal_coil_2d=toroidal_coil_2d, toroid_coil_config=toroid_coil_config
    )

    # Create plotter with vertical layout (2 rows, 1 column)
    plotter = initialize_plotter(shape=(1, 2))
    plotter.window_size = (1000, 1000)

    # First row: 2D boundary
    plotter.subplot(0, 0)
    render_plasma_boundary(
        plotter=plotter, plasma_boundary=plasma_boundary, toroidal_coil_2d=toroidal_coil_2d
    )

    # Second row: 3D geometry - use existing function
    plotter.subplot(0, 1)
    render_fusion_plasma(
        plotter=plotter,
        fusion_plasma=fusion_plasma,
        toroidal_coils=toroidal_coils_3d,
        show_cylindrical_angles=True,
        show_wireframe=True,
    )
    plotter.view_isometric()

    # Display in Streamlit
    stpyvista(plotter=plotter, key="fusion_plot")
