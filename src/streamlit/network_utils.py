import contextlib
import math
from pathlib import Path
import shutil

import jax.numpy as jnp
import plotly.graph_objects as go

from src.engine.plasma import calculate_poloidal_boundary
from src.lib.config import Filepaths
from src.lib.geometry_config import PlasmaConfig, PlasmaGeometry, PlasmaState, RotationalAngles

# Roots searched for run dirs, in priority order. Add a new root here and
# both resolve_run_directory() and get_available_networks() pick it up.
RUN_ROOTS: list[Path] = [Filepaths.BENCHMARKS, Filepaths.BENCHMARK_ARCHIVE]


def _scan_run_dirs(base: Path) -> list[Path]:
    """Return run dirs (containing network.flax) under <base>/<commit>/<run>/."""
    if not base.exists():
        return []
    runs: list[Path] = []
    for commit_dir in base.iterdir():
        if not commit_dir.is_dir():
            continue
        for run_dir in commit_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "network.flax").exists():
                runs.append(run_dir)
    return runs


def network_name(run_dir: Path) -> str:
    """Canonical network name: commit/run (relative to the benchmarks root)."""
    return f"{run_dir.parent.name}/{run_dir.name}"


def get_available_networks(view_mode: str = "All") -> list[str]:
    """List networks as 'commit/run' paths from the benchmark tree."""
    names: list[str] = []
    if view_mode in ["New Benchmarks", "All"]:
        for run_dir in _scan_run_dirs(Filepaths.BENCHMARKS):
            if run_dir.parent.name == "_archive":
                continue
            names.append(network_name(run_dir))
    if view_mode in ["Archive", "All"]:
        for run_dir in _scan_run_dirs(Filepaths.BENCHMARK_ARCHIVE):
            names.append(network_name(run_dir))
    return sorted(names)


def get_available_commits(networks: list[str]) -> list[str]:
    return sorted({n.split("/")[0] for n in networks if "/" in n})


def filter_networks_by_commit(networks: list[str], commit: str | None) -> list[str]:
    if not commit or commit == "All":
        return networks
    return [n for n in networks if n.startswith(f"{commit}/")]


def resolve_run_directory(name: str) -> Path:
    """Resolve a 'commit/run' network name to its run dir, checking all RUN_ROOTS."""
    for root in RUN_ROOTS:
        p = root / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Run dir not found: {name}")


def move_run_dir(old_name: str, new_path: Path) -> Path:
    """Move a run dir to a new path, cleaning up the parent if it becomes empty.

    ``old_name`` is a 'commit/run' name; ``new_path`` is the full target dir.
    Returns the new path.
    """
    old_path = resolve_run_directory(old_name)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))
    with contextlib.suppress(OSError):
        old_path.parent.rmdir()
    return new_path


def to_plasma_config(geom: PlasmaGeometry, state: PlasmaState) -> PlasmaConfig:
    boundary = calculate_poloidal_boundary(
        jnp.linspace(0, 2 * jnp.pi, RotationalAngles.n_theta), geom
    )
    return PlasmaConfig(Geometry=geom, Boundary=boundary, State=state)


def apply_grid_layout(fig: go.Figure, n_items: int, height_per_row: int = 400) -> None:
    n_cols = min(n_items, 4)
    n_rows = math.ceil(n_items / n_cols)
    fig.update_layout(height=height_per_row * n_rows, margin={"l": 10, "r": 10, "t": 30, "b": 10})
