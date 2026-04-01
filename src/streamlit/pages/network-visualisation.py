import os
from typing import Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stpyvista import stpyvista

from src.engine.network import NetworkManager, Sampler
from src.engine.physics import grad_shafranov_residual
from src.engine.plasma import (
    calculate_fusion_plasma,
    calculate_poloidal_boundary,
    is_point_in_plasma,
)
from src.lib.config import Filepaths
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
    plot_flux_heatmap,
    render_fusion_plasma,
    render_magnetic_field_lines,
)
import streamlit as st

st.set_page_config(layout="wide", page_title="Fusion Simulation Lab")

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


@st.cache_resource(max_entries=1)
def load_network_manager(pinn_path: str) -> NetworkManager:
    """Initialize manager and load parameters from disk."""

    params = st.session_state.manager.from_disk(pinn_path=pinn_path)
    st.session_state.manager.state = st.session_state.manager.state.replace(params=params)
    return st.session_state.manager


def swap_network_button(key: str) -> NetworkManager:
    networks = os.listdir(Filepaths.NETWORKS)
    key = "network_btn" + key
    selected_network = st.selectbox(label="Select Network", options=networks, key=key)
    st.session_state.manager = load_network_manager(Filepaths.NETWORKS / selected_network)


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
def render_geometry_sampling_tab(seed: int):  # noqa
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


def _build_psi_fn(manager: NetworkManager):  # noqa
    """Build a psi_fn(params, R, Z, config) closure matching the training-time signature.

    This mirrors the closure inside ``NetworkManager.train_step`` so that
    ``grad_shafranov_residual`` (which differentiates through *psi_fn*)
    produces results fully consistent with the training loss.
    """
    apply_fn = manager.model.apply

    def psi_fn(params: any, R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> jnp.ndarray:
        inp = FluxInput(R_sample=R, Z_sample=Z, config=cfg)
        p_n, r_n, z_n = inp.normalize()
        psi_n = apply_fn(params, r=r_n, z=z_n, **p_n)
        return (psi_n * cfg.State.F_axis * cfg.Geometry.a).squeeze()

    return psi_fn


def _compute_gs_residual_on_points(
    manager: NetworkManager,
    config: PlasmaConfig,
    R_pts: jnp.ndarray,
    Z_pts: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the normalised Grad-Shafranov residual at given (R, Z) points.

    Uses ``grad_shafranov_residual`` from *physics.py* — the same function
    that the PINN training loss evaluates — so the heatmap is directly
    comparable to the optimisation objective.
    """
    psi_fn = _build_psi_fn(manager)
    params = manager.state.params

    # Estimate psi_axis from the supplied points (mirrors training strategy)
    psi_vals = jax.vmap(lambda r, z: psi_fn(params, r, z, config))(R_pts, Z_pts)
    psi_axis = jax.lax.stop_gradient(jnp.min(psi_vals))

    # Vectorised residual — identical to the inner vmap in pinn_loss_function
    residual_fn = jax.vmap(
        lambda r, z: grad_shafranov_residual(psi_fn, params, r, z, psi_axis, config)
    )
    return residual_fn(R_pts, Z_pts)


def render_flux_predictions_tab(seed: int):  # noqa
    """Render the magnetic flux prediction heatmap tab, with optional GS residual."""
    col1, col2, col3, col4, _ = st.columns([1, 1, 1, 1, 3])
    with col1:
        swap_network_button(key="flux_pred_tab")

    res = col2.slider("Grid Resolution", 20, 200, 100, key="prediction_res")
    tab_seed = col3.number_input("Random Seed", 0, 9999, seed, key="prediction_seed")
    mode = col4.radio(" ", options=["Flux Prediction", "GS Residual"], horizontal=True)

    data = get_data(tab_seed)
    configs: List[PlasmaConfig] = []
    for d in data:
        boundary = calculate_poloidal_boundary(
            jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta), d["geom"]
        )
        configs.append(PlasmaConfig(Geometry=d["geom"], Boundary=boundary, State=d["state"]))

    if mode == "Flux Prediction":
        fig_flux = plot_flux_heatmap(
            st.session_state.manager, configs, backend="plotly", resolution=res
        )
        fig_flux.update_layout(height=500)
        st.subheader("Magnetic Flux ψ(R, Z)")
        st.plotly_chart(fig_flux, use_container_width=True)

    if mode == "GS Residual":
        with st.spinner("Computing Grad-Shafranov residual…"):
            # Build a common grid matching the flux-heatmap approach in visualization.py
            geoms = [c.Geometry for c in configs]
            r_min = min(float(g.R0 - g.a * 1.2) for g in geoms)
            r_max = max(float(g.R0 + g.a * 1.2) for g in geoms)
            z_max = max(float(g.kappa * g.a * 1.2) for g in geoms)
            r_mid = (r_min + r_max) / 2
            extent = max(r_max - r_min, 2 * z_max)
            r_lims = [r_mid - extent / 2, r_mid + extent / 2]
            z_lims = [-extent / 2, extent / 2]

            R_lin = jnp.linspace(float(r_lims[0]), float(r_lims[1]), res)
            Z_lin = jnp.linspace(float(z_lims[0]), float(z_lims[1]), res)
            R_grid, Z_grid = jnp.meshgrid(R_lin, Z_lin)
            R_flat = R_grid.flatten()
            Z_flat = Z_grid.flatten()

            coords_flat = CylindricalCoordinates(R=R_flat, Z=Z_flat, phi=jnp.zeros_like(R_flat))

            n_cols = len(configs)
            fig_res = make_subplots(
                rows=1,
                cols=n_cols,
                subplot_titles=[f"Config {i + 1}" for i in range(n_cols)],
                horizontal_spacing=0.05,
                shared_yaxes=True,
            )

            for idx, cfg in enumerate(configs):
                mask = is_point_in_plasma(coords_flat, cfg.Boundary)

                # Fill with NaN; only compute inside the plasma boundary
                residual_full = jnp.full(R_flat.shape, jnp.nan)
                if jnp.any(mask):
                    res_vals = _compute_gs_residual_on_points(
                        st.session_state.manager, cfg, R_flat[mask], Z_flat[mask]
                    )
                    residual_full = residual_full.at[mask].set(res_vals)

                residual_2d = np.array(residual_full).reshape(res, res)
                col = idx + 1

                fig_res.add_trace(
                    go.Heatmap(
                        x=np.array(R_lin),
                        y=np.array(Z_lin),
                        z=residual_2d,
                        colorscale="RdBu_r",
                        zmid=0,
                        zmin=-0.5,
                        zmax=0.5,
                        showscale=(col == n_cols),
                        colorbar={"title": "Residual"} if col == n_cols else {},
                    ),
                    1,
                    col,
                )
                # Overlay plasma boundary
                fig_res.add_trace(
                    go.Scatter(
                        x=np.array(cfg.Boundary.R),
                        y=np.array(cfg.Boundary.Z),
                        mode="lines",
                        line={"color": "white", "width": 2},
                        showlegend=False,
                    ),
                    1,
                    col,
                )

            # Axis formatting to match the flux heatmap
            for i in range(n_cols):
                col = i + 1
                fig_res.update_xaxes(title_text="R (m)", range=r_lims, row=1, col=col)
                fig_res.update_yaxes(
                    range=z_lims,
                    scaleanchor=f"x{col if col > 1 else ''}",
                    scaleratio=1,
                    row=1,
                    col=col,
                )
                if i == 0:
                    fig_res.update_yaxes(title_text="Z (m)", row=1, col=col)

            fig_res.update_layout(
                height=500,
                template="plotly_dark",
                margin={"l": 20, "r": 20, "t": 40, "b": 20},
                showlegend=False,
            )

        st.subheader("Grad-Shafranov Residual")
        st.plotly_chart(fig_res, use_container_width=True)
        st.caption(
            "The residual shows where the PINN prediction deviates from the "
            "Grad-Shafranov equation (delta*psi + mu_0 R^2 dp/dpsi + F dF/dpsi = 0). "
            "Values near zero (white) indicate good physics compliance."
        )


def render_3d_topology_tab():  # noqa
    """Render the 3D topology and field line tracing tab."""
    col1, col2, col3, _ = st.columns([1, 1, 1, 4])

    with col1:
        swap_network_button(key="3d_tab")

    idx_pv = col2.slider(
        "Select Sample", 0, len(st.session_state.manager.train_set) - 1, 0, key="idx_pv_slider"
    )
    n_lines_pv = col3.slider("Number of Field Lines", 1, 50, 20, key="n_lines_pv_slider")

    p = st.session_state.manager.train_set[idx_pv]
    geom_pv = PlasmaGeometry(R0=p[0], a=p[1], kappa=p[2], delta=p[3])
    state_pv = PlasmaState(p0=p[4], F_axis=p[5], pressure_alpha=p[6], field_exponent=p[7])

    boundary_pv = calculate_poloidal_boundary(
        jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta), geom_pv
    )
    plasma_config = PlasmaConfig(Geometry=geom_pv, Boundary=boundary_pv, State=state_pv)
    fusion_plasma = calculate_fusion_plasma(boundary_pv)

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
        stpyvista(plotter, key=f"pv_reactor_{idx_pv}_{n_lines_pv}")


# --- Main Application ---
tab1, tab2, tab3 = st.tabs(["Geometry Sampling", "Flux Predictions", "3D Magnetic Field Lines"])
if "manager" not in st.session_state:
    st.session_state.manager = NetworkManager(HyperParams())

with tab1:
    render_geometry_sampling_tab(seed=42)

with tab2:
    render_flux_predictions_tab(seed=42)

with tab3:
    render_3d_topology_tab()
