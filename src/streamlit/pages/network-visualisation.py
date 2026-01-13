from typing import Dict, List, Union

import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.engine.network import Sampler
from src.lib.geometry_config import PlasmaGeometry, PlasmaState
from src.lib.network_config import HyperParams
import streamlit as st

st.set_page_config(layout="wide", page_title="Plasma Geometry Lab")

# Define a type for the geometry data
GeometryData = Dict[str, Union[PlasmaGeometry, PlasmaState, jnp.ndarray]]

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    with st.form("sim_params"):
        n_geoms = st.slider("Number of geometries", 1, 8, 4)
        n_int = st.number_input("Interior Points", 100, 5000, 1024)
        n_bound = st.number_input("Boundary Points", 50, 1000, 256)
        seed = st.number_input("Random Seed", 0, 9999, 42)
        st.form_submit_button("Generate Geometries")


# --- Data Sampling Logic ---
@st.cache_data
def get_data(n_geoms: int, n_int: int, n_bound: int, seed: int) -> List[GeometryData]:
    """Generate sampled geometry data for visualization.

    Args:
        n_geoms: Number of plasma geometries to generate
        n_int: Number of interior sampling points per geometry
        n_bound: Number of boundary sampling points per geometry
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries containing geometry data for each configuration
    """
    # Create sampler instance
    config = HyperParams()  # Use default hyperparameters
    sampler = Sampler(config)

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


data = get_data(n_geoms, n_int, n_bound, seed)

# --- UI Layout ---
st.title("Network Visualization")
tab1, tab2 = st.tabs(["Geometry Sampling", "Model Training"])

with tab1:
    view_option = st.selectbox(
        "Select Geometry View", ["Show All"] + [f"Geometry {i + 1}" for i in range(n_geoms)]
    )
    indices = range(n_geoms) if view_option == "Show All" else [int(view_option.split()[-1]) - 1]

    # Display Metrics for the first selected geometry
    m_cols = st.columns(6)
    target = data[indices[0]]
    metrics = [
        ("R0", target["geom"].R0),
        ("a", target["geom"].a),
        ("κ", target["geom"].kappa),
        ("δ", target["geom"].delta),
        ("p0", target["state"].p0),
        ("F_axis", target["state"].F_axis),
    ]
    for col, (label, val) in zip(m_cols, metrics, strict=True):
        col.metric(label, f"{val:.2e}" if val > 100 else f"{val:.2f}")

    # Plotting
    n_plots = len(indices)
    cols = min(n_plots, 2)
    rows = (n_plots + 1) // 2

    # Adjust row height: Single column views need more height to match the wider container
    # while maintaining the 1:1 aspect ratio.
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
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.info("Model Training metrics and loss curves will appear here.")
