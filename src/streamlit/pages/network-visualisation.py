import json
from pathlib import Path

import jax.numpy as jnp
from stpyvista import stpyvista

from src.engine.model_evaluation import compute_gs_residual_on_points
from src.engine.network import NetworkManager
from src.engine.plasma import calculate_fusion_plasma
from src.lib.config import Filepaths
from src.lib.network_config import HyperParams
from src.lib.visualization import (
    initialize_plotter,
    plot_flux_heatmap,
    plot_gs_residual_heatmap,
    plot_rz_samples,
    render_fusion_plasma,
    render_magnetic_field_lines,
)
from src.streamlit.network_utils import (
    apply_grid_layout,
    filter_networks_by_commit,
    get_available_commits,
    get_available_networks,
    move_network_files,
    to_plasma_config,
)
from src.streamlit.utils import reseed_network_visualisation
import streamlit as st

st.set_page_config(layout="wide", page_title="Fusion Simulation Lab")

PLOT_GRID_RESOLUTION = 100


def _get_networks() -> list[str]:
    return get_available_networks(st.session_state.get("network_view_mode", "New Benchmarks"))


def sync_selected_network() -> None:
    """Load selected checkpoint once and reseed shared samples."""
    pinn_path = Filepaths.NETWORKS / st.session_state.selected_pinn
    new_config = HyperParams.from_json(pinn_path.with_suffix(".json"))

    # Re-instantiate only if the JIT-sensitive architecture changes
    if (
        "manager" not in st.session_state
        or st.session_state.manager.config.hidden_dims != new_config.hidden_dims
    ):
        st.session_state.manager = NetworkManager(new_config)
    else:
        # Same architecture: safely reuse XLA executables and update configs in-place
        st.session_state.manager.config = new_config
        st.session_state.manager.sampler.config = new_config

    params = st.session_state.manager.from_disk(pinn_path=pinn_path)
    st.session_state.manager.state = st.session_state.manager.state.replace(params=params)
    reseed_network_visualisation()


# --- UI Components ---
def render_geometry_sampling_tab():  # noqa
    """Render the geometry sampling visualization tab."""
    n = st.session_state.get("sample_size", 4)
    col1, _, _ = st.columns([1, 1, 4])
    with col1:
        view_option = st.selectbox(
            "Select Geometry View",
            ["Show All"] + [f"Geometry {i + 1}" for i in range(n)],
            key="geom_view",
        )

    data = st.session_state.seeded_geometry_data
    indices = range(n) if view_option == "Show All" else [int(view_option.split()[-1]) - 1]

    selected_samples = [data[idx] for idx in indices]
    geometries = [sample["geom"] for sample in selected_samples]
    fig = plot_rz_samples(geometries, selected_samples)
    st.plotly_chart(fig, width="stretch")


def render_flux_predictions_tab() -> None:
    """Render the magnetic flux prediction heatmap tab, with optional GS residual."""
    metrics_col, _, options_col, sample_col = st.columns([3, 2, 2, 1])

    with metrics_col:
        render_metrics()

    with options_col:
        st.radio(
            "Plot Options",
            options=["Flux Prediction", "GS Residual", "Both"],
            horizontal=True,
            key="prediction_mode",
        )

    with sample_col:
        st.selectbox(
            "Sample Size",
            options=[4, 8, 12, 16, 20, 24],
            key="sample_size",
            on_change=reseed_network_visualisation,
        )

    mode = st.session_state.get("prediction_mode", "Flux Prediction")

    data = st.session_state.seeded_geometry_data
    configs = [to_plasma_config(d["geom"], d["state"]) for d in data]

    if mode in ["Flux Prediction", "Both"]:
        st.subheader("Magnetic Flux ψ(R, Z)")
        fig_flux = plot_flux_heatmap(
            st.session_state.manager, configs, backend="plotly", resolution=PLOT_GRID_RESOLUTION
        )
        apply_grid_layout(fig_flux, len(configs), height_per_row=500)
        st.plotly_chart(fig_flux, width="stretch", key="heatmap_chart_flux")

    if mode in ["GS Residual", "Both"]:
        st.subheader("Grad-Shafranov Residual")
        fig_res = plot_gs_residual_heatmap(
            st.session_state.manager, configs, resolution=PLOT_GRID_RESOLUTION
        )
        apply_grid_layout(fig_res, len(configs), height_per_row=500)
        st.plotly_chart(fig_res, width="stretch", key="heatmap_chart_res")


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


def handle_archive() -> None:
    Filepaths.NETWORK_ARCHIVE.mkdir(parents=True, exist_ok=True)
    new_path_stem = Filepaths.NETWORK_ARCHIVE / Path(st.session_state.selected_pinn).stem
    move_network_files(st.session_state.selected_pinn, new_path_stem)
    _update_networks_after_action()


def handle_rename(new_name: str) -> None:
    if not new_name:
        return

    old_path = Filepaths.NETWORKS / st.session_state.selected_pinn
    new_path_stem = old_path.parent / new_name

    move_network_files(st.session_state.selected_pinn, new_path_stem)

    st.session_state.available_networks = _get_networks()
    st.session_state._next_pinn = str(
        new_path_stem.with_suffix(".flax").relative_to(Filepaths.NETWORKS)
    )
    st.session_state.rename_mode = False
    st.rerun()


def handle_delete() -> None:
    target_path = Filepaths.NETWORKS / st.session_state.selected_pinn
    if target_path.exists():
        target_path.unlink()
    if target_path.with_suffix(".json").exists():
        target_path.with_suffix(".json").unlink()

    st.session_state.rename_mode = False
    _update_networks_after_action()


def _update_networks_after_action() -> None:
    st.session_state.available_networks = _get_networks()
    if st.session_state.available_networks:
        commit_filter = st.session_state.get("filter_commit", "All")
        filtered_networks = filter_networks_by_commit(
            st.session_state.available_networks,
            commit_filter if commit_filter != "All" else None,
        )
        st.session_state._next_pinn = (
            filtered_networks[-1] if filtered_networks else st.session_state.available_networks[-1]
        )
    else:
        st.session_state._next_pinn = None
    st.rerun()


def render_network_actions() -> None:
    if not st.session_state.get("selected_pinn"):
        return

    col1, col2, col3 = st.columns(3)

    if col1.button("Archive", width="stretch"):
        handle_archive()

    if col2.button("Rename", width="stretch"):
        st.session_state.rename_mode = not st.session_state.get("rename_mode", False)

    if st.session_state.get("rename_mode", False):
        new_name = st.text_input("New Name", value=Path(st.session_state.selected_pinn).stem)
        if st.button("Save Name"):
            handle_rename(new_name)

    if col3.button("Delete", width="stretch", type="primary"):
        handle_delete()


def render_sidebar() -> None:
    with st.sidebar:
        # Apply pending network selection before widget instantiation
        if st.session_state.get("_next_pinn"):
            st.session_state.selected_pinn = st.session_state._next_pinn
            del st.session_state._next_pinn

        st.radio(
            "View",
            options=["New Benchmarks", "Archive", "All"],
            horizontal=True,
            key="network_view_mode",
        )

        st.session_state.available_networks = _get_networks()
        commit_filter = st.session_state.get("filter_commit", "All")
        commit_filter = commit_filter if commit_filter != "All" else None

        filtered_networks = filter_networks_by_commit(
            st.session_state.available_networks, commit_filter
        )

        was_selected = st.session_state.get("selected_pinn")
        if was_selected not in filtered_networks and filtered_networks:
            st.session_state.selected_pinn = filtered_networks[0]
            sync_selected_network()

        st.selectbox(
            "Select Network",
            options=filtered_networks,
            key="selected_pinn",
            on_change=sync_selected_network,
        )

        commits = ["All", *get_available_commits(st.session_state.available_networks)]
        st.selectbox("Filter by Commit", options=commits, key="filter_commit")

        render_network_actions()

        st.divider()

        st.slider(
            "Select Sample",
            min_value=0,
            max_value=1000,
            key="seed",
            on_change=reseed_network_visualisation,
        )

        st.divider()

        pinn_path = Filepaths.NETWORKS / st.session_state.selected_pinn
        if pinn_path.with_suffix(".json").exists():
            with open(pinn_path.with_suffix(".json")) as f:
                st.json(json.load(f))


def render_metrics() -> None:
    """Render high-level metrics for the currently sampled geometries."""
    manager: NetworkManager = st.session_state.manager
    flux_input = st.session_state.seeded_flux_input

    # Compute evaluation metrics
    total, l_res, l_dir, l_per_cfg = manager.eval_step(
        manager.state, flux_input, manager.config.weight_boundary_condition
    )

    # Compute max pointwise residual across all sampled configurations
    max_res = 0.0
    for i, cfg in enumerate(flux_input.config):
        res_vals = compute_gs_residual_on_points(
            manager, cfg, flux_input.R_sample[i], flux_input.Z_sample[i]
        )
        max_res = max(max_res, float(jnp.max(jnp.abs(res_vals))))

    cols = st.columns(5)
    cols[3].metric("Max Loss", f"{float(jnp.max(l_per_cfg)):.2f}")
    cols[0].metric("Avg Loss", f"{float(total):.2f}")
    cols[1].metric("Interior Loss", f"{float(l_res):.2f}")
    cols[2].metric("Boundary Loss (raw)", f"{float(l_dir):.2f}")
    cols[4].metric("Max Residual", f"{max_res:.2f}")


def init_session_state() -> None:
    if "selected_pinn" not in st.session_state:
        st.session_state.seed = 0
        st.session_state.sample_size = 4
        st.session_state.filter_commit = "All"
        st.session_state.network_view_mode = "New Benchmarks"

        networks = _get_networks()
        st.session_state.available_networks = networks
        if networks:
            st.session_state.selected_pinn = networks[0]
            sync_selected_network()
        else:
            # Fallback if no networks exist
            st.session_state.manager = NetworkManager(HyperParams())
            reseed_network_visualisation()


def main() -> None:
    init_session_state()

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
