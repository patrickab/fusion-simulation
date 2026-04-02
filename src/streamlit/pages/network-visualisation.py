import jax.numpy as jnp
from stpyvista import stpyvista

from src.engine.network import NetworkManager
from src.engine.plasma import (
    calculate_fusion_plasma,
    calculate_poloidal_boundary,
)
from src.lib.config import Filepaths
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
    plot_gs_residual_heatmap,
    plot_rz_samples,
    render_fusion_plasma,
    render_magnetic_field_lines,
)
from src.streamlit.utils import reseed_network_visualisation
import streamlit as st

st.set_page_config(layout="wide", page_title="Fusion Simulation Lab")


def sync_selected_network() -> None:
    """Load selected checkpoint once and reseed shared samples."""
    selected_network = st.session_state.selected_pinn
    if st.session_state.get("loaded_pinn") == selected_network:
        return

    pinn_path = Filepaths.NETWORKS / selected_network
    params = st.session_state.manager.from_disk(pinn_path=pinn_path)
    st.session_state.manager.state = st.session_state.manager.state.replace(params=params)
    st.session_state.loaded_pinn = selected_network
    reseed_network_visualisation()


def to_plasma_config(geom: PlasmaGeometry, state: PlasmaState) -> PlasmaConfig:
    boundary = calculate_poloidal_boundary(
        jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta),
        geom,
    )
    return PlasmaConfig(Geometry=geom, Boundary=boundary, State=state)


# --- UI Components ---
def render_geometry_sampling_tab():  # noqa
    """Render the geometry sampling visualization tab."""
    col1, _, _ = st.columns([1, 1, 4])
    with col1:
        view_option = st.selectbox(
            "Select Geometry View",
            ["Show All"] + [f"Geometry {i + 1}" for i in range(4)],
            key="geom_view",
        )

    data = st.session_state.seeded_geometry_data
    indices = range(4) if view_option == "Show All" else [int(view_option.split()[-1]) - 1]

    selected_samples = [data[idx] for idx in indices]
    geometries = [sample["geom"] for sample in selected_samples]
    fig = plot_rz_samples(geometries, selected_samples)
    st.plotly_chart(fig, use_container_width=True)


def render_flux_predictions_tab():  # noqa
    """Render the magnetic flux prediction heatmap tab, with optional GS residual."""
    col1, col2, _ = st.columns([1, 1, 4])
    res = col1.slider("Grid Resolution", 20, 200, 100, key="prediction_res")
    mode = col2.radio(" ", options=["Flux Prediction", "GS Residual"], horizontal=True)

    data = st.session_state.seeded_geometry_data
    configs = [to_plasma_config(d["geom"], d["state"]) for d in data]

    if mode == "Flux Prediction":
        fig_flux = plot_flux_heatmap(
            st.session_state.manager,
            configs,
            backend="plotly",
            resolution=res,
        )
        fig_flux.update_layout(height=500)
        st.subheader("Magnetic Flux ψ(R, Z)")
        st.plotly_chart(fig_flux, use_container_width=True)
        return

    with st.spinner("Computing Grad-Shafranov residual…"):
        fig_res = plot_gs_residual_heatmap(st.session_state.manager, configs, resolution=res)
        fig_res.update_layout(height=500, margin={"l": 20, "r": 20, "t": 40, "b": 20})

    st.subheader("Grad-Shafranov Residual")
    st.plotly_chart(fig_res, use_container_width=True)


def render_3d_topology_tab():  # noqa
    """Render the 3D topology and field line tracing tab."""
    col1, _ = st.columns([1, 5])
    n_lines_pv = col1.slider("Number of Field Lines", 1, 50, 20, key="n_lines_pv_slider")

    geom = st.session_state.seeded_3d_geom
    state = st.session_state.seeded_3d_state
    plasma_config = to_plasma_config(geom, state)
    fusion_plasma = calculate_fusion_plasma(plasma_config.Boundary)

    st.subheader("3D Magnetic Field Lines")
    plotter = initialize_plotter(window_size=(1000, 900))

    render_fusion_plasma(
        plotter=plotter,
        fusion_plasma=fusion_plasma,
        toroidal_coils=[],
        show_wireframe=True,
    )

    with st.spinner("Tracing field lines..."):
        render_magnetic_field_lines(
            plotter=plotter,
            network_manager=st.session_state.manager,
            config=plasma_config,
            n_lines=n_lines_pv,
        )
        plotter.view_isometric()
        stpyvista(plotter)


def render_sidebar() -> None:
    with st.sidebar:
        st.selectbox(
            "Select Network",
            options=st.session_state.available_networks,
            key="selected_pinn",
            on_change=sync_selected_network,
        )
        st.slider(
            "Select Sample",
            min_value=0,
            max_value=1000,
            key="seed",
            on_change=reseed_network_visualisation,
        )


def main() -> None:
    if "manager" not in st.session_state:
        st.session_state.manager = NetworkManager(HyperParams())
        st.session_state.available_networks = sorted(
            p.name for p in Filepaths.NETWORKS.iterdir() if p.is_file()
        )
        st.session_state.selected_pinn = st.session_state.available_networks[0]
        st.session_state.seed = 0
        reseed_network_visualisation()

    sync_selected_network()
    render_sidebar()

    tab1, tab2, tab3 = st.tabs(["Geometry Sampling", "Flux Predictions", "3D Magnetic Field Lines"])

    with tab1:
        render_geometry_sampling_tab()

    with tab2:
        render_flux_predictions_tab()

    with tab3:
        render_3d_topology_tab()


if __name__ == "__main__":
    main()
