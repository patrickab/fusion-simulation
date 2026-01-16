from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv
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
    RotationalAngles,
)
from src.lib.network_config import FluxInput, HyperParams
from src.lib.visualization import (
    initialize_plotter,
    render_fusion_plasma,
    render_magnetic_field_lines,
)
from src.streamlit.utils import reactor_config_sidebar
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
    """Generate sampled geometry data for visualization."""
    config = HyperParams()
    sampler = Sampler(config)
    n_geoms = 4

    bounds = sampler._build_domain_bounds()
    plasma_configs = sampler._get_sobol_sample(n_geoms, seed, *bounds)

    flux_input = sampler.sample_flux_input(
        seed, config.n_rz_inner_samples, config.n_rz_boundary_samples, plasma_configs
    )

    return [
        {
            "geom": flux_input.config[i].Geometry,
            "state": flux_input.config[i].State,
            "bR": flux_input.config[i].Boundary.R,
            "bZ": flux_input.config[i].Boundary.Z,
            "iR": flux_input.R_sample[i],
            "iZ": flux_input.Z_sample[i],
        }
        for i in range(n_geoms)
    ]


@st.cache_resource
def load_network_manager() -> NetworkManager | None:
    """Initialize manager and load parameters from disk."""
    mgr = NetworkManager(HyperParams())
    try:
        params = NetworkManager.from_disk(mgr.state.params)
        mgr.state = mgr.state.replace(params=params)
        return mgr
    except FileNotFoundError:
        return None


def setup_physics_subplots(
    geoms: List[PlasmaGeometry], titles: List[str]
) -> tuple[go.Figure, List[float], List[float]]:
    """Initialize subplots with consistent global scaling and 1:1 aspect ratio."""
    r_min = min(g.R0 - g.a * 1.2 for g in geoms)
    r_max = max(g.R0 + g.a * 1.2 for g in geoms)
    z_max = max(g.kappa * g.a * 1.2 for g in geoms)

    r_mid, extent = (r_min + r_max) / 2, max(r_max - r_min, 2 * z_max)
    r_lims, z_lims = [r_mid - extent / 2, r_mid + extent / 2], [-extent / 2, extent / 2]

    fig = make_subplots(
        rows=1, cols=len(geoms), subplot_titles=titles, horizontal_spacing=0.05, shared_yaxes=True
    )
    for i in range(len(geoms)):
        col = i + 1
        fig.update_xaxes(title_text="R (m)", range=r_lims, row=1, col=col)
        fig.update_yaxes(
            range=z_lims, scaleanchor=f"x{col if col > 1 else ''}", scaleratio=1, row=1, col=col
        )
        if i == 0:
            fig.update_yaxes(title_text="Z (m)", row=1, col=col)

    fig.update_layout(margin={"t": 40, "b": 40, "l": 40, "r": 10}, showlegend=False)
    return fig, r_lims, z_lims


def plot_flux_heatmap(
    manager: NetworkManager,
    configs: List[PlasmaConfig],
    titles: List[str],
    resolution: int = 50,
) -> go.Figure:
    """Generate a Plotly figure with magnetic flux heatmaps for multiple configurations."""
    fig, r_lims, z_lims = setup_physics_subplots([c.Geometry for c in configs], titles)

    res = resolution
    R, Z = jnp.linspace(*r_lims, res), jnp.linspace(*z_lims, res)
    R_grid, Z_grid = jnp.meshgrid(R, Z)
    coords_flat = CylindricalCoordinates(
        R=R_grid.flatten(), Z=Z_grid.flatten(), phi=jnp.zeros(res**2)
    )

    all_psi = []
    plot_data = []

    for cfg in configs:
        mask = is_point_in_plasma(coords_flat, cfg.Boundary)
        psi_grid = jnp.full(mask.shape, jnp.nan)
        if mask.any():
            psi = manager.predict(
                FluxInput(
                    R_sample=coords_flat.R[mask],
                    Z_sample=coords_flat.Z[mask],
                    config=cfg,
                )
            )
            all_psi.append(psi.flatten())
            psi_grid = psi_grid.at[mask].set(psi.flatten())

        plot_data.append(
            {
                "psi": psi_grid.reshape(res, res),
                "boundary": cfg.Boundary,
                "active": mask.any(),
            }
        )

    if not all_psi:
        return fig

    psi_min, psi_max = (
        float(jnp.nanmin(jnp.concatenate(all_psi))),
        float(jnp.nanmax(jnp.concatenate(all_psi))),
    )

    for i, p in enumerate(plot_data):
        col = i + 1
        if p["active"]:
            fig.add_trace(
                go.Heatmap(
                    x=R,
                    y=Z,
                    z=p["psi"],
                    colorscale="Viridis",
                    zmin=psi_min,
                    zmax=psi_max,
                    showscale=(i == len(plot_data) - 1),
                    colorbar={"title": "Psi", "len": 0.8} if i == len(plot_data) - 1 else None,
                    name=f"Psi {i + 1}",
                ),
                1,
                col,
            )
        fig.add_trace(
            go.Scatter(
                x=p["boundary"].R,
                y=p["boundary"].Z,
                mode="lines",
                line={"color": "white", "width": 2},
                showlegend=False,
                name=f"Boundary {i + 1}",
            ),
            1,
            col,
        )

    return fig


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

    # Initialize subplots
    geoms = [data[idx]["geom"] for idx in indices]
    titles = [f"Geometry {idx + 1}" for idx in indices]
    fig, _, _ = setup_physics_subplots(geoms, titles)
    fig.update_layout(height=600 if view_option == "Show All" else 800)

    for i, idx in enumerate(indices):
        d = data[idx]
        col = i + 1
        fig.add_trace(
            go.Scatter(
                x=d["iR"],
                y=d["iZ"],
                mode="markers",
                marker={"size": 2, "color": "purple"},
                name="Interior",
            ),
            1,
            col,
        )
        fig.add_trace(
            go.Scatter(
                x=d["bR"],
                y=d["bZ"],
                mode="markers",
                marker={"size": 4, "color": "red"},
                name="Boundary",
            ),
            1,
            col,
        )

    if view_option == "Show All":
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, width="content")

with tab2:
    st.info("Model Training metrics and loss curves will appear here.")
    manager = load_network_manager()
    if manager:
        st.subheader("Current Model Flux Preview")
        p = manager.train_set[0]
        geom = PlasmaGeometry(R0=p[0], a=p[1], kappa=p[2], delta=p[3])
        state = PlasmaState(p0=p[4], F_axis=p[5], pressure_alpha=p[6], field_exponent=p[7])
        boundary = calculate_poloidal_boundary(
            jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta), geom
        )
        config = PlasmaConfig(Geometry=geom, Boundary=boundary, State=state)
        fig = plot_flux_heatmap(manager, [config], ["Training Sample 1"], resolution=50)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Flux Prediction")
    manager = load_network_manager()
    if not manager:
        st.warning("Model not found. Please train the model first.")
        st.stop()

    col1, col2 = st.columns(2)
    seed = col1.number_input("Random Seed", 0, 9999, 42, key="prediction_seed")
    res = col2.slider("Grid Resolution", 20, 200, 50, key="prediction_res")

    data = get_data(seed)
    configs = []
    for d in data:
        boundary = calculate_poloidal_boundary(
            jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta), d["geom"]
        )
        configs.append(PlasmaConfig(Geometry=d["geom"], Boundary=boundary, State=d["state"]))

    fig = plot_flux_heatmap(manager, configs, [f"Geom {i + 1}" for i in range(4)], resolution=res)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("3D Magnetic Field Visualization (PyVista)")
    manager = load_network_manager()
    if not manager:
        st.warning("Model not found. Please train the model first.")
        st.stop()

    st.write("Starting 3D visualization...")

    col1, col2 = st.columns(2)
    use_custom = col1.checkbox("Use Custom Geometry from Sidebar", value=False)
    idx_pv = col1.slider(
        "Select Sample",
        0,
        len(manager.train_set) - 1,
        0,
        key="idx_pv_slider",
        disabled=use_custom,
    )
    n_lines_pv = col2.slider("Number of Field Lines", 1, 50, 20, key="n_lines_pv_slider")

    if use_custom:
        geom_pv = custom_geom
        state_pv = PlasmaState(p0=1e5, F_axis=40.0, pressure_alpha=2.0, field_exponent=1.0)
    else:
        p = manager.train_set[idx_pv]
        geom_pv = PlasmaGeometry(R0=p[0], a=p[1], kappa=p[2], delta=p[3])
        state_pv = PlasmaState(p0=p[4], F_axis=p[5], pressure_alpha=p[6], field_exponent=p[7])

    boundary_pv = calculate_poloidal_boundary(
        jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta), geom_pv
    )
    plasma_config = PlasmaConfig(Geometry=geom_pv, Boundary=boundary_pv, State=state_pv)
    fusion_plasma = calculate_fusion_plasma(boundary_pv)

    # 1:3 Layout for 2D Heatmap and 3D Visualization
    c1, c2 = st.columns([1, 3])

    with c1:
        st.subheader("Poloidal Flux")
        fig_2d = plot_flux_heatmap(manager, [plasma_config], ["Selected Geometry"], resolution=50)
        fig_2d.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_2d, use_container_width=True)

    with c2:
        # Initialize plotter
        plotter = initialize_plotter()

        # Render Plasma and Coils
        render_fusion_plasma(
            plotter=plotter,
            fusion_plasma=fusion_plasma,
            toroidal_coils=generate_toroidal_coils_3d(
                calculate_toroidal_coil_boundary(
                    jnp.linspace(0, 2 * jnp.pi, 64), geom_pv, custom_coil_config
                ),
                custom_coil_config,
            ),
            show_wireframe=True,
        )

        # Add Magnetic Field Lines
        with st.spinner("Tracing field lines..."):
            render_magnetic_field_lines(
                plotter=plotter,
                network_manager=manager,
                config=plasma_config,
                n_lines=n_lines_pv,
            )

        plotter.view_isometric()
        stpyvista(plotter, key=f"pv_reactor_{idx_pv}_{n_lines_pv}_{use_custom}")
