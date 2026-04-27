from datetime import datetime
import json
import math
from pathlib import Path

import jax.numpy as jnp
import plotly.graph_objects as go

from src.engine.network import NetworkManager
from src.engine.plasma import calculate_poloidal_boundary
from src.lib.config import Filepaths
from src.lib.geometry_config import (
    PlasmaConfig,
    PlasmaGeometry,
    PlasmaState,
    RotationalAngles,
)
from src.lib.network_config import HyperParams
from src.lib.visualization import (
    plot_flux_heatmap,
    plot_gs_residual_heatmap,
)
from src.streamlit.utils import reseed_network_visualisation
import streamlit as st

st.set_page_config(layout="wide", page_title="Benchmark Visualizer")

PLOT_GRID_RESOLUTION = 80


def extract_commit(filename: str) -> str | None:
    parts = Path(filename).stem.split("_")
    return parts[-1] if len(parts) >= 2 else None


def get_available_networks() -> list[str]:
    paths = list(Filepaths.NETWORKS.glob("*.flax"))
    if Filepaths.NETWORK_ARCHIVE.exists():
        paths.extend(Filepaths.NETWORK_ARCHIVE.glob("*.flax"))
    return sorted(str(p.relative_to(Filepaths.NETWORKS)) for p in paths if p.is_file())


def get_available_commits(networks: list[str]) -> list[str]:
    return sorted({c for n in networks if (c := extract_commit(n))})


def filter_networks_by_commit(networks: list[str], commit: str | None) -> list[str]:
    if not commit or commit == "All":
        return networks
    return [n for n in networks if extract_commit(n) == commit]


def to_plasma_config(geom: PlasmaGeometry, state: PlasmaState) -> PlasmaConfig:
    boundary = calculate_poloidal_boundary(
        jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta), geom
    )
    return PlasmaConfig(Geometry=geom, Boundary=boundary, State=state)


def apply_grid_layout(fig: go.Figure, n_items: int) -> None:
    n_cols = min(n_items, 4)
    n_rows = math.ceil(n_items / n_cols)
    fig.update_layout(height=400 * n_rows, margin={"l": 10, "r": 10, "t": 30, "b": 10})


def render_benchmark_row(network_name: str, configs: list[PlasmaConfig], mode: str) -> None:
    pinn_path = Filepaths.NETWORKS / network_name
    config_path = pinn_path.with_suffix(".json")

    if not config_path.exists():
        st.error(f"Missing config for {network_name}")
        return

    network_config = HyperParams.from_json(config_path)
    with open(config_path) as f:
        config_dict = json.load(f)

    mtime = datetime.fromtimestamp(pinn_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    st.subheader(f"{mtime} | {network_name}")

    col_meta, col_plots = st.columns([1, 4])

    with col_meta:
        st.json(config_dict)

    with col_plots:
        manager = NetworkManager(network_config)
        params = manager.from_disk(pinn_path=pinn_path)
        manager.state = manager.state.replace(params=params)

        if mode in ["Flux Prediction", "Both"]:
            st.write("**Magnetic Flux ψ(R, Z)**")
            fig_flux = plot_flux_heatmap(
                manager, configs, backend="plotly", resolution=PLOT_GRID_RESOLUTION
            )
            apply_grid_layout(fig_flux, len(configs))
            st.plotly_chart(fig_flux, use_container_width=True, key=f"flux_{network_name}")

        if mode in ["GS Residual", "Both"]:
            st.write("**Grad-Shafranov Residual**")
            fig_res = plot_gs_residual_heatmap(manager, configs, resolution=PLOT_GRID_RESOLUTION)
            apply_grid_layout(fig_res, len(configs))
            st.plotly_chart(fig_res, use_container_width=True, key=f"res_{network_name}")

    st.divider()


def init_session_state(networks: list[str]) -> None:
    if "manager" not in st.session_state:
        config = HyperParams()
        if networks:
            config_path = (Filepaths.NETWORKS / networks[0]).with_suffix(".json")
            if config_path.exists():
                config = HyperParams.from_json(config_path)
        st.session_state.manager = NetworkManager(config)

    st.session_state.setdefault("seed", 0)
    st.session_state.setdefault("sample_size", 4)


def main() -> None:
    networks = get_available_networks()
    init_session_state(networks)

    with st.sidebar:
        st.header("Benchmark Controls")
        commits = get_available_commits(networks)
        selected_commit = st.selectbox("Select Commit", ["All", *commits], index=0)

        st.selectbox(
            "Sample Size",
            [1, 2, 4, 8, 12, 16],
            index=3,
            key="sample_size_bench",
            on_change=lambda: st.session_state.update(
                sample_size=st.session_state.sample_size_bench
            ),
        )

        mode = st.radio("Visualization Mode", ["Flux Prediction", "GS Residual", "Both"], index=2)

        st.divider()
        st.slider("Reseed Samples", 0, 1000, key="seed", on_change=reseed_network_visualisation)
        run_bench = st.button("Run Benchmark", use_container_width=True, type="primary")

    if "seeded_geometry_data" not in st.session_state:
        reseed_network_visualisation()

    if run_bench:
        st.session_state.sample_size = st.session_state.sample_size_bench
        reseed_network_visualisation()

        filtered_networks = filter_networks_by_commit(networks, selected_commit)
        if not filtered_networks:
            st.warning("No networks found for the selected commit.")
            return

        data = st.session_state.seeded_geometry_data
        configs = [to_plasma_config(d["geom"], d["state"]) for d in data]

        for network_name in reversed(filtered_networks):
            render_benchmark_row(network_name, configs, mode)


if __name__ == "__main__":
    main()
