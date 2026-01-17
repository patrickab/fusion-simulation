from typing import Dict, List, Union

import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stpyvista import stpyvista

from src.engine.network import NetworkManager, Sampler
from src.engine.plasma import (
    calculate_fusion_plasma,
    calculate_poloidal_boundary,
)
from src.lib.geometry_config import (
    PlasmaConfig,
    PlasmaGeometry,
    PlasmaState,
    RotationalAngles,
)
from src.lib.network_config import HyperParams
from src.lib.visualization import (
    initialize_plotter,
    plot_flux_heatmap,
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


manager = load_network_manager()


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


# --- UI Components ---
def render_geometry_sampling_tab(seed: int):
    """Render the geometry sampling visualization tab."""
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        view_option = st.selectbox(
            "Select Geometry View",
            ["Show All"] + [f"Geometry {i + 1}" for i in range(4)],
            key="geom_view",
        )
    with col2:
        tab_seed = st.number_input("Random Seed", 0, 9999, seed, key="sampling_seed_input")

    data = get_data(tab_seed)
    indices = range(4) if view_option == "Show All" else [int(view_option.split()[-1]) - 1]

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

    st.plotly_chart(fig, use_container_width=True)


def render_flux_predictions_tab(manager: NetworkManager, seed: int):
    """Render the magnetic flux prediction heatmap tab."""
    col1, col2, _ = st.columns([1, 1, 4])
    res = col1.slider("Grid Resolution", 20, 200, 50, key="prediction_res")
    tab_seed = col2.number_input("Random Seed", 0, 9999, seed, key="prediction_seed")

    data = get_data(tab_seed)
    configs = []
    for d in data:
        boundary = calculate_poloidal_boundary(
            jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta), d["geom"]
        )
        configs.append(PlasmaConfig(Geometry=d["geom"], Boundary=boundary, State=d["state"]))

    if manager:
        fig = plot_flux_heatmap(manager, configs, backend="plotly", resolution=res)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Network parameters not found on disk. Please train the network first.")


def render_3d_topology_tab(manager: NetworkManager, custom_coil_config):
    """Render the 3D topology and field line tracing tab."""
    col1, col2, _ = st.columns([1, 1, 4])
    idx_pv = col1.slider("Select Sample", 0, len(manager.train_set) - 1, 0, key="idx_pv_slider")
    n_lines_pv = col2.slider("Number of Field Lines", 1, 50, 20, key="n_lines_pv_slider")

    p = manager.train_set[idx_pv]
    geom_pv = PlasmaGeometry(R0=p[0], a=p[1], kappa=p[2], delta=p[3])
    state_pv = PlasmaState(p0=p[4], F_axis=p[5], pressure_alpha=p[6], field_exponent=p[7])

    boundary_pv = calculate_poloidal_boundary(
        jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta), geom_pv
    )
    plasma_config = PlasmaConfig(Geometry=geom_pv, Boundary=boundary_pv, State=state_pv)
    fusion_plasma = calculate_fusion_plasma(boundary_pv)

    st.subheader("3D Topology")
    plotter = initialize_plotter(window_size=(800, 1000))

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

    with st.spinner("Tracing field lines..."):
        render_magnetic_field_lines(
            plotter=plotter,
            network_manager=manager,
            config=plasma_config,
            n_lines=n_lines_pv,
        )
        plotter.view_isometric()
        stpyvista(plotter, key=f"pv_reactor_{idx_pv}_{n_lines_pv}")


# --- Main Application ---
tab1, tab2, tab3 = st.tabs(["Geometry Sampling", "Flux Predictions", "3D Magnetic Field Lines"])

with tab1:
    render_geometry_sampling_tab(seed=42)

with tab2:
    render_flux_predictions_tab(manager, seed=42)

with tab3:
    if manager:
        render_3d_topology_tab(manager, custom_coil_config)
    else:
        st.error("Network manager not initialized.")
