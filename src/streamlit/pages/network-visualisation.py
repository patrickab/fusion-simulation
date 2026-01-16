from typing import Dict, List, Union

import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stpyvista import stpyvista

from src.engine.network import NetworkManager, Sampler
from src.engine.plasma import (
    calculate_fusion_plasma,
    calculate_poloidal_boundary,
    is_point_in_plasma,
)
from src.lib.geometry_config import (
    CylindricalCoordinates,
    PlasmaConfig,
    PlasmaGeometry,
    PlasmaState,
)
from src.lib.network_config import FluxInput, HyperParams
from src.lib.utils import _plasma_to_polydata
from src.lib.visualization import (
    initialize_plotter,
    render_fusion_plasma,
)
from src.streamlit.utils import generate_field_lines, reactor_config_sidebar
from src.toroidal_geometry import (
    calculate_toroidal_coil_boundary,
    generate_toroidal_coils_3d,
)
import streamlit as st

st.set_page_config(layout="wide", page_title="Plasma Geometry Lab")

# Get custom geometry from sidebar
custom_geom, custom_coil_config = reactor_config_sidebar()

# Define a type for the geometry data
GeometryData = Dict[str, Union[PlasmaGeometry, PlasmaState, jnp.ndarray]]


# --- Data Sampling Logic ---
@st.cache_data
def get_data(seed: int) -> List[GeometryData]:
    """Generate sampled geometry data for visualization.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries containing geometry data for each configuration
    """
    # Create sampler instance with default hyperparameters
    config = HyperParams()  # Use default hyperparameters
    sampler = Sampler(config)

    # Always generate 4 geometries
    n_geoms = 4

    # Use default values from config for interior and boundary points
    n_int = config.n_rz_inner_samples
    n_bound = config.n_rz_boundary_samples

    # Get domain bounds and sample plasma configurations
    lower_bounds, upper_bounds = sampler._build_domain_bounds()
    plasma_configs = sampler._get_sobol_sample(
        n_samples=n_geoms,
        seed=seed,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    # Sample interior and boundary points for all configurations
    flux_input = sampler.sample_flux_input(
        seed=seed,
        n_samples=n_int,
        n_boundary_samples=n_bound,
        plasma_configs=plasma_configs,
    )

    # Unpack FluxInput and format results
    results: List[GeometryData] = []
    # When n_geoms > 1, flux_input.config is a batched PlasmaConfig
    # When n_geoms == 1, it's a single PlasmaConfig
    if n_geoms > 1:
        for i in range(n_geoms):
            config_item = flux_input.config[i]
            results.append(
                {
                    "geom": config_item.Geometry,
                    "state": config_item.State,
                    "bR": config_item.Boundary.R,
                    "bZ": config_item.Boundary.Z,
                    "iR": flux_input.R_sample[i],
                    "iZ": flux_input.Z_sample[i],
                }
            )
    else:
        # Single configuration case
        results.append(
            {
                "geom": flux_input.config.Geometry,
                "state": flux_input.config.State,
                "bR": flux_input.config.Boundary.R,
                "bZ": flux_input.config.Boundary.Z,
                "iR": flux_input.R_sample[0],
                "iZ": flux_input.Z_sample[0],
            }
        )

    return results


# --- UI Layout ---
st.title("Network Visualization")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Geometry Sampling", "Model Training", "Predictions", "3D Visualization"]
)

with tab1:
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        view_option = st.selectbox(
            "Select Geometry View", ["Show All"] + [f"Geometry {i + 1}" for i in range(4)]
        )
    with col2:
        seed = st.number_input("Random Seed", 0, 9999, 42, key="seed_input")

    data = get_data(seed)

    indices = range(4) if view_option == "Show All" else [int(view_option.split()[-1]) - 1]

    # Plotting
    n_plots = len(indices)
    cols = n_plots
    rows = 1

    # Adjust row height: Single column views need more height to match the wider container
    # while maintaining the 1:1 aspect ratio.
    if view_option == "Show All":
        row_height = 600  # Increased for stretched view
    else:
        row_height = 1200 if cols == 1 else 450

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Geometry {i + 1}" for i in indices],
        # Reduce internal spacing to maximize data-ink ratio
        vertical_spacing=0.1 if rows > 1 else 0,
    )

    for i, idx in enumerate(indices):
        d = data[idx]
        r, c = (i // cols) + 1, (i % cols) + 1
        fig.add_trace(
            go.Scatter(
                x=d["iR"],
                y=d["iZ"],
                mode="markers",
                marker=dict(size=2, color="purple"),  # noqa
                name="Interior",
            ),
            r,
            c,
        )
        fig.add_trace(
            go.Scatter(
                x=d["bR"],
                y=d["bZ"],
                mode="markers",
                marker=dict(size=4, color="red"),  # noqa
                name="Boundary",
            ),
            r,
            c,
        )

        # Ensure 1:1 Aspect Ratio (Physical dimensions)
        # Note: scaleanchor must point to the x-axis of the *same* subplot
        xaxis_name = f"x{i + 1 if i > 0 else ''}"
        fig.update_xaxes(title_text="R (m)", row=r, col=c)
        fig.update_yaxes(
            title_text="Z (m)",
            scaleanchor=xaxis_name,
            scaleratio=1,
            row=r,
            col=c,
        )

    fig.update_layout(
        height=row_height * rows,
        showlegend=False,
        margin={"t": 40, "b": 40, "l": 10, "r": 10},  # Reduce margins
    )
    if view_option == "Show All":
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, width="content")

with tab2:
    st.info("Model Training metrics and loss curves will appear here.")

with tab3:
    st.header("Flux Prediction")

    # Load Model
    config = HyperParams()
    manager = NetworkManager(config)
    try:
        params = NetworkManager.from_disk(manager.state.params)
        manager.state = manager.state.replace(params=params)
    except FileNotFoundError:
        st.warning("Model not found. Please train the model first.")
        st.stop()

    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        idx = st.slider("Select Training Sample", 0, len(manager.train_set) - 1, 0)
    with col2:
        res = st.slider("Grid Resolution", 20, 200, 50)

    # Reconstruct Plasma Config
    p = manager.train_set[idx]
    geom = PlasmaGeometry(R0=p[0], a=p[1], kappa=p[2], delta=p[3])
    state = PlasmaState(p0=p[4], F_axis=p[5], pressure_alpha=p[6], field_exponent=p[7])
    boundary = calculate_poloidal_boundary(jnp.linspace(0, 2 * jnp.pi, 128), geom)

    # Generate Grid
    R = jnp.linspace(geom.R0 - geom.a * 1.2, geom.R0 + geom.a * 1.2, res)
    Z = jnp.linspace(-geom.kappa * geom.a * 1.2, geom.kappa * geom.a * 1.2, res)
    R_grid, Z_grid = jnp.meshgrid(R, Z)

    # Filter Points
    coords = CylindricalCoordinates(R=R_grid.flatten(), Z=Z_grid.flatten(), phi=jnp.zeros(res**2))
    mask = is_point_in_plasma(coords, boundary)

    if mask.any():
        # Predict
        inputs = FluxInput(
            R_sample=coords.R[mask],
            Z_sample=coords.Z[mask],
            config=PlasmaConfig(Geometry=geom, Boundary=boundary, State=state),
        )
        psi = manager.predict(inputs)

        # Visualize
        psi_grid = jnp.full(mask.shape, jnp.nan).at[mask].set(psi.flatten()).reshape(res, res)

        fig = go.Figure(
            go.Heatmap(x=R, y=Z, z=psi_grid, colorscale="Viridis", colorbar_title="Psi")
        )
        fig.add_trace(
            go.Scatter(
                x=boundary.R,
                y=boundary.Z,
                mode="lines",
                line_color="white",
                name="Boundary",
            )
        )
        fig.update_layout(yaxis_scaleanchor="x", height=600)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("No points found inside the plasma boundary.")

with tab4:
    st.header("3D Magnetic Field Visualization (PyVista)")

    # Load Model
    config_params = HyperParams()
    manager = NetworkManager(config_params)
    try:
        params = NetworkManager.from_disk(manager.state.params)
        manager.state = manager.state.replace(params=params)
    except FileNotFoundError:
        st.warning("Model not found. Please train the model first.")
        st.stop()

    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        use_custom = st.checkbox("Use Custom Geometry from Sidebar", value=False)
        idx_pv = st.slider(
            "Select Sample",
            0,
            len(manager.train_set) - 1,
            0,
            key="idx_pv_slider",
            disabled=use_custom,
        )
    with col2:
        n_lines_pv = st.slider("Number of Field Lines", 1, 30, 10, key="n_lines_pv_slider")

    # Reconstruct Plasma Config
    if use_custom:
        geom_pv = custom_geom
        state_pv = PlasmaState(p0=1e5, F_axis=5.0, pressure_alpha=2.0, field_exponent=1.0)
        coil_config = custom_coil_config
    else:
        p_pv = manager.train_set[idx_pv]
        geom_pv = PlasmaGeometry(R0=p_pv[0], a=p_pv[1], kappa=p_pv[2], delta=p_pv[3])
        state_pv = PlasmaState(
            p0=p_pv[4],
            F_axis=p_pv[5],
            pressure_alpha=p_pv[6],
            field_exponent=p_pv[7],
        )
        coil_config = custom_coil_config

    boundary_pv = calculate_poloidal_boundary(jnp.linspace(0, 2 * jnp.pi, 128), geom_pv)
    plasma_config_pv = PlasmaConfig(Geometry=geom_pv, Boundary=boundary_pv, State=state_pv)

    # 1. Create Fusion Plasma Surface
    fusion_plasma = calculate_fusion_plasma(boundary_pv)
    plasma_mesh = _plasma_to_polydata(fusion_plasma)

    # 2. Generate Field Lines from PINN - Use Cartesian method directly
    pts = plasma_mesh.points  # Already in Cartesian (X, Y, Z)
    X_mesh = jnp.array(pts[:, 0])
    Y_mesh = jnp.array(pts[:, 1])
    Z_mesh = jnp.array(pts[:, 2])

    # Get B-field directly in Cartesian coordinates
    B_cartesian = manager.get_b_field_cartesian(
        X=X_mesh, Y=Y_mesh, Z=Z_mesh, config=plasma_config_pv
    )

    streamlines = generate_field_lines(plasma_mesh, n_lines_pv, B_field=B_cartesian)

    # 3. Generate Toroidal Coils
    coil_2d = calculate_toroidal_coil_boundary(
        jnp.linspace(0, 2 * jnp.pi, 64), geom_pv, coil_config
    )
    coils_3d = generate_toroidal_coils_3d(coil_2d, coil_config)

    # 4. Plotting with PyVista
    plotter = initialize_plotter()

    render_fusion_plasma(
        plotter=plotter,
        fusion_plasma=fusion_plasma,
        toroidal_coils=coils_3d,
        show_wireframe=False,
    )

    # Add field lines
    plotter.add_mesh(streamlines, color="white", line_width=2, name="field_lines")
    plotter.view_isometric()

    # Render in Streamlit
    stpyvista(plotter, key=f"pv_reactor_{idx_pv}_{n_lines_pv}_{use_custom}")
