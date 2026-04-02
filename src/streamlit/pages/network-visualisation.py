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

GeometryData = Dict[str, Union[PlasmaGeometry, PlasmaState, jnp.ndarray]]


def get_available_networks() -> List[str]:
    return sorted(p.name for p in Filepaths.NETWORKS.iterdir() if p.is_file())


def reseed() -> None:
    manager: NetworkManager = st.session_state.manager
    sampler = Sampler(manager.config, seed=int(st.session_state.seed))
    bounds = sampler._build_domain_bounds()

    seeded_geometry_configs = sampler._get_sobol_sample(
        n_samples=4,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
    )
    seeded_train_set = sampler._get_sobol_sample(
        n_samples=st.session_state.manager.config.n_train,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
    )
    sample_3d = seeded_train_set[st.session_state.seed]

    geom_3d = PlasmaGeometry(
        R0=sample_3d[0],
        a=sample_3d[1],
        kappa=sample_3d[2],
        delta=sample_3d[3],
    )
    state_3d = PlasmaState(
        p0=sample_3d[4],
        F_axis=sample_3d[5],
        pressure_alpha=sample_3d[6],
        field_exponent=sample_3d[7],
    )

    flux_input = sampler.sample_flux_input(plasma_configs=seeded_geometry_configs)
    seeded_geometry_data = [
        {
            "geom": flux_input.config[i].Geometry,
            "state": flux_input.config[i].State,
            "bR": flux_input.config[i].Boundary.R,
            "bZ": flux_input.config[i].Boundary.Z,
            "iR": flux_input.R_sample[i],
            "iZ": flux_input.Z_sample[i],
        }
        for i in range(len(seeded_geometry_configs))
    ]

    st.session_state.seeded_sampler = sampler
    st.session_state.seeded_geometry_configs = seeded_geometry_configs
    st.session_state.seeded_train_set = seeded_train_set
    st.session_state.seeded_geometry_data = seeded_geometry_data
    st.session_state.seeded_3d_geom = geom_3d
    st.session_state.seeded_3d_state = state_3d


def get_data() -> List[GeometryData]:
    return st.session_state.seeded_geometry_data


def sync_selected_network() -> None:
    """Load selected checkpoint once and reseed shared samples."""
    selected_network = st.session_state.selected_pinn
    if st.session_state.get("loaded_pinn") == selected_network:
        return

    pinn_path = Filepaths.NETWORKS / selected_network
    params = st.session_state.manager.from_disk(pinn_path=pinn_path)
    st.session_state.manager.state = st.session_state.manager.state.replace(params=params)
    st.session_state.loaded_pinn = selected_network
    reseed()


def to_plasma_config(geom: PlasmaGeometry, state: PlasmaState) -> PlasmaConfig:
    boundary = calculate_poloidal_boundary(
        jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta),
        geom,
    )
    return PlasmaConfig(Geometry=geom, Boundary=boundary, State=state)


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
def render_geometry_sampling_tab():  # noqa
    """Render the geometry sampling visualization tab."""
    col1, _, _ = st.columns([1, 1, 4])
    with col1:
        view_option = st.selectbox(
            "Select Geometry View",
            ["Show All"] + [f"Geometry {i + 1}" for i in range(4)],
            key="geom_view",
        )

    data = get_data()
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


def render_flux_predictions_tab():  # noqa
    """Render the magnetic flux prediction heatmap tab, with optional GS residual."""
    col1, col2, _ = st.columns([1, 1, 4])
    res = col1.slider("Grid Resolution", 20, 200, 100, key="prediction_res")
    mode = col2.radio(" ", options=["Flux Prediction", "GS Residual"], horizontal=True)

    data = get_data()
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
        geoms = [cfg.Geometry for cfg in configs]
        fig_res, r_lims, z_lims = setup_physics_subplots(
            geoms, [f"Config {i + 1}" for i in range(len(configs))]
        )

        R_lin = jnp.linspace(float(r_lims[0]), float(r_lims[1]), res)
        Z_lin = jnp.linspace(float(z_lims[0]), float(z_lims[1]), res)
        R_grid, Z_grid = jnp.meshgrid(R_lin, Z_lin)
        R_flat, Z_flat = R_grid.ravel(), Z_grid.ravel()

        coords_flat = CylindricalCoordinates(R=R_flat, Z=Z_flat, phi=jnp.zeros_like(R_flat))

        for col, cfg in enumerate(configs, start=1):
            mask = is_point_in_plasma(coords_flat, cfg.Boundary)
            residual_full = jnp.full(R_flat.shape, jnp.nan)

            if jnp.any(mask):
                residual_vals = _compute_gs_residual_on_points(
                    st.session_state.manager, cfg, R_flat[mask], Z_flat[mask]
                )
                residual_full = residual_full.at[mask].set(residual_vals)

            fig_res.add_trace(
                go.Heatmap(
                    x=np.array(R_lin),
                    y=np.array(Z_lin),
                    z=np.array(residual_full).reshape(res, res),
                    colorscale="RdBu_r",
                    zmid=0,
                    zmin=-0.5,
                    zmax=0.5,
                    showscale=(col == len(configs)),
                    colorbar={"title": "Residual"} if col == len(configs) else {},
                ),
                row=1,
                col=col,
            )
            fig_res.add_trace(
                go.Scatter(
                    x=np.array(cfg.Boundary.R),
                    y=np.array(cfg.Boundary.Z),
                    mode="lines",
                    line={"color": "white", "width": 2},
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

        fig_res.update_layout(
            height=500,
            template="plotly_dark",
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            showlegend=False,
        )

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
            on_change=reseed,
        )


def main() -> None:
    if "manager" not in st.session_state:
        st.session_state.manager = NetworkManager(HyperParams())
        st.session_state.available_networks = get_available_networks()
        st.session_state.selected_pinn = st.session_state.available_networks[0]
        st.session_state.seed = 0
        reseed()

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
