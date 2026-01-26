from stpyvista import stpyvista

from src.engine.plasma import calculate_fusion_plasma
from src.lib.geometry_config import PlasmaGeometry, ToroidalCoilConfig
from src.lib.visualization import (
    calculate_2d_geometry,
    initialize_plotter,
    render_fusion_plasma,
    render_plasma_boundary,
)
from src.toroidal_geometry import generate_toroidal_coils_3d
from src.streamlit.utils import reactor_config_sidebar
import streamlit as st

st.title("Reactor Geometry")

# Get geometry from sidebar
plasma_geometry, toroid_coil_config = reactor_config_sidebar()

# Layout for view controls
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    view_option = st.selectbox(
        "View Mode", ["2D Geometry", "3D Geometry", "Both"], index=2
    )
with col2:
    show_coils = st.checkbox("Show Coils", value=True)

# Extract parameters for the key
R0, a, kappa, delta = (
    plasma_geometry.R0,
    plasma_geometry.a,
    plasma_geometry.kappa,
    plasma_geometry.delta,
)
dist = toroid_coil_config.distance_from_plasma
r_thick = toroid_coil_config.radial_thickness
v_thick = toroid_coil_config.vertical_thickness
span = toroid_coil_config.angular_span
n_coils = toroid_coil_config.n_field_coils

# Generate and display the plot
with st.spinner("Calculating geometry..."):
    # Calculate geometry
    plasma_boundary, toroidal_coil_2d = calculate_2d_geometry(
        plasma_geometry=plasma_geometry, toroid_coil_config=toroid_coil_config
    )

    # Calculate 3D geometry
    fusion_plasma = calculate_fusion_plasma(plasma_boundary=plasma_boundary)
    toroidal_coils_3d = generate_toroidal_coils_3d(
        toroidal_coil_2d=toroidal_coil_2d, toroid_coil_config=toroid_coil_config
    )

    # Create plotter based on view option
    if view_option == "Both":
        plotter = initialize_plotter(shape=(1, 2))
        plotter.window_size = (1000, 500)
    else:
        plotter = initialize_plotter(shape=(1, 1))
        plotter.window_size = (600, 600)

    # Render based on selection
    if view_option in ["2D Geometry", "Both"]:
        if view_option == "Both":
            plotter.subplot(0, 0)
        render_plasma_boundary(
            plotter=plotter,
            plasma_boundary=plasma_boundary,
            toroidal_coil_2d=toroidal_coil_2d if show_coils else None,
        )

    if view_option in ["3D Geometry", "Both"]:
        if view_option == "Both":
            plotter.subplot(0, 1)
        render_fusion_plasma(
            plotter=plotter,
            fusion_plasma=fusion_plasma,
            toroidal_coils=toroidal_coils_3d if show_coils else [],
            show_cylindrical_angles=True,
            show_wireframe=True,
        )
        plotter.view_isometric()

    # Display in Streamlit
    # Create a container for the plot
    plot_container = st.empty()

    # Clear and render the plot in the container
    with plot_container:
        stpyvista(
            plotter=plotter,
            key=f"fusion_plot_{R0}_{a}_{kappa}_{delta}_{dist}_{r_thick}_{v_thick}_{span}_{n_coils}_{view_option}_{show_coils}",
        )
