import math
from pathlib import Path

import jax.numpy as jnp
import plotly.graph_objects as go

from src.engine.plasma import calculate_poloidal_boundary
from src.lib.config import Filepaths
from src.lib.geometry_config import PlasmaConfig, PlasmaGeometry, PlasmaState, RotationalAngles


def extract_commit(filename: str) -> str | None:
    parts = Path(filename).stem.split("_")
    return parts[-1] if len(parts) >= 2 else None


def get_available_networks(view_mode: str = "All") -> list[str]:
    paths = []
    if view_mode in ["New Benchmarks", "All"]:
        paths.extend(p for p in Filepaths.NETWORKS.glob("*.flax") if p.is_file())
    if view_mode in ["Archive", "All"] and Filepaths.NETWORK_ARCHIVE.exists():
        paths.extend(p for p in Filepaths.NETWORK_ARCHIVE.glob("*.flax") if p.is_file())
    return sorted(str(p.relative_to(Filepaths.NETWORKS)) for p in paths)


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


def apply_grid_layout(fig: go.Figure, n_items: int, height_per_row: int = 400) -> None:
    n_cols = min(n_items, 4)
    n_rows = math.ceil(n_items / n_cols)
    fig.update_layout(height=height_per_row * n_rows, margin={"l": 10, "r": 10, "t": 30, "b": 10})


def move_network_files(old_name: str, new_path_stem: Path) -> None:
    """Moves both the .flax and .json files to a new path/name."""
    old_path = Filepaths.NETWORKS / old_name

    if old_path.exists():
        old_path.rename(new_path_stem.with_suffix(".flax"))
    if old_path.with_suffix(".json").exists():
        old_path.with_suffix(".json").rename(new_path_stem.with_suffix(".json"))
