import json

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


def extract_commit(filename: str) -> str | None:
    stem = filename.replace(".flax", "")
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[-1]
    return None


def get_available_commits(networks: list[str]) -> list[str]:
    commits = set()
    for network in networks:
        commit = extract_commit(network)
        if commit:
            commits.add(commit)
    return sorted(commits)


def filter_networks_by_commit(networks: list[str], commit: str | None) -> list[str]:
    if not commit:
        return networks
    return [n for n in networks if extract_commit(n) == commit]


def sync_selected_network() -> None:
    """Load selected checkpoint once and reseed shared samples."""
    pinn_path = Filepaths.NETWORKS / st.session_state.selected_pinn
    st.session_state.manager = NetworkManager(HyperParams.from_json(pinn_path.with_suffix(".json")))
    params = st.session_state.manager.from_disk(pinn_path=pinn_path)
    st.session_state.manager.state = st.session_state.manager.state.replace(params=params)
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
        top_container = st.container()

        if st.session_state.get("selected_pinn"):
            col1, col2 = st.columns(2)

            # Rename Network
            rename_clicked = col1.button("Rename", use_container_width=True)
            if rename_clicked:
                st.session_state.rename_mode = not st.session_state.get("rename_mode", False)

            if st.session_state.get("rename_mode", False):
                new_name = st.text_input(
                    "New Name", value=st.session_state.selected_pinn.replace(".flax", "")
                )
                if st.button("Save Name") and new_name:
                    flax_name = new_name if new_name.endswith(".flax") else f"{new_name}.flax"
                    old_path = Filepaths.NETWORKS / st.session_state.selected_pinn
                    new_path = Filepaths.NETWORKS / flax_name

                    if old_path.exists():
                        old_path.rename(new_path)
                    if old_path.with_suffix(".json").exists():
                        old_path.with_suffix(".json").rename(new_path.with_suffix(".json"))

                    # Update state
                    st.session_state.available_networks = sorted(
                        p.name for p in Filepaths.NETWORKS.glob("*.flax") if p.is_file()
                    )
                    st.session_state.selected_pinn = flax_name
                    st.session_state.rename_mode = False
                    sync_selected_network()
                    st.rerun()

            # Delete Network
            if col2.button("Delete", use_container_width=True, type="primary"):
                target_path = Filepaths.NETWORKS / st.session_state.selected_pinn
                if target_path.exists():
                    target_path.unlink()
                if target_path.with_suffix(".json").exists():
                    target_path.with_suffix(".json").unlink()

                st.session_state.available_networks = sorted(
                    p.name for p in Filepaths.NETWORKS.glob("*.flax") if p.is_file()
                )
                if st.session_state.available_networks:
                    st.session_state.selected_pinn = st.session_state.available_networks[-1]
                    sync_selected_network()
                else:
                    st.session_state.selected_pinn = None
                st.session_state.rename_mode = False
                st.rerun()

        with top_container:
            commit_filter = st.session_state.get("filter_commit", "All")
            commit_filter = commit_filter if commit_filter != "All" else None
            filtered_networks = filter_networks_by_commit(
                st.session_state.available_networks, commit_filter
            )
            if st.session_state.selected_pinn not in filtered_networks and filtered_networks:
                st.session_state.selected_pinn = filtered_networks[0]
            st.selectbox(
                "Select Network",
                options=filtered_networks,
                key="selected_pinn",
                on_change=sync_selected_network,
            )
            st.selectbox(
                "Filter by Commit",
                options=["All"] + get_available_commits(st.session_state.available_networks),
                key="filter_commit",
            )

        st.slider(
            "Select Sample",
            min_value=0,
            max_value=1000,
            key="seed",
            on_change=reseed_network_visualisation,
        )
        pinn_path = Filepaths.NETWORKS / st.session_state.selected_pinn
        if pinn_path.with_suffix(".json").exists():
            with open(pinn_path.with_suffix(".json")) as f:
                st.json(json.load(f))


def main() -> None:
    if "manager" not in st.session_state:
        st.session_state.manager = NetworkManager(HyperParams())
        st.session_state.available_networks = sorted(
            p.name for p in Filepaths.NETWORKS.glob("*.flax") if p.is_file()
        )
        st.session_state.selected_pinn = st.session_state.available_networks[0]
        st.session_state.filter_commit = "All"
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
