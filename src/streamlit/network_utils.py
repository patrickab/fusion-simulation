import contextlib
from datetime import datetime
import math
from pathlib import Path
import re
import shutil

import jax.numpy as jnp
import plotly.graph_objects as go

from src.engine.plasma import calculate_poloidal_boundary
from src.lib.config import Filepaths
from src.lib.geometry_config import PlasmaConfig, PlasmaGeometry, PlasmaState, RotationalAngles

# Single-config runs are stored directly below these roots.
RUN_ROOTS: list[Path] = [Filepaths.BENCHMARKS, Filepaths.BENCHMARK_ARCHIVE]
HPO_PREFIX = "hpo"


def _scan_run_dirs(base: Path) -> list[Path]:
    """Return direct child run dirs containing network.flax."""
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir() and (p / "network.flax").exists()]


def network_name(run_dir: Path) -> str:
    """Return a single-config network slug."""
    return run_dir.name


def parse_slug(slug: str) -> tuple[str, str, str]:
    """Split a <timestamp>_<name>_<commit> artifact slug."""
    parts = slug.split("_")
    if len(parts) < 8:
        raise ValueError(f"Invalid artifact slug: {slug}")
    timestamp = "_".join(parts[:6])
    try:
        datetime.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")
    except ValueError as exc:
        raise ValueError(f"Invalid artifact timestamp: {slug}") from exc
    name = "_".join(parts[6:-1])
    commit = parts[-1]
    if not name or not commit:
        raise ValueError(f"Invalid artifact slug: {slug}")
    return timestamp, name, commit


def renamed_slug(slug: str, new_name: str) -> str:
    """Rebuild a slug with a replacement name segment."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", new_name):
        raise ValueError("Name must contain only letters, numbers, underscores, or hyphens")
    timestamp, _, commit = parse_slug(slug)
    return f"{timestamp}_{new_name}_{commit}"


def is_hpo_name(name: str) -> bool:
    """Return whether a canonical network name addresses an HPO run."""
    return name.startswith(f"{HPO_PREFIX}/")


def split_hpo_name(name: str) -> tuple[str, str | None]:
    """Split hpo/<study>[/<run>] into its study and optional run slug."""
    parts = name.split("/")
    if (
        parts[0] != HPO_PREFIX
        or len(parts) not in (2, 3)
        or not all(parts[1:])
        or any(part in {".", ".."} for part in parts[1:])
    ):
        raise ValueError(f"Invalid HPO network name: {name}")
    return parts[1], parts[2] if len(parts) == 3 else None


def hpo_network_name(study: str, run: str) -> str:
    """Build the canonical name for one HPO trial network."""
    return f"{HPO_PREFIX}/{study}/{run}"


def get_available_networks(view_mode: str = "All") -> list[str]:
    """List single-config networks, with All retaining legacy HPO coverage."""
    names: list[str] = []
    if view_mode in ["Single-Configs", "New Benchmarks", "All"]:
        for run_dir in _scan_run_dirs(Filepaths.BENCHMARKS):
            names.append(network_name(run_dir))
    if view_mode in ["Archive", "All"]:
        for run_dir in _scan_run_dirs(Filepaths.BENCHMARK_ARCHIVE):
            names.append(network_name(run_dir))
    if view_mode == "All":
        for archived in (False, True):
            for study, runs in get_hpo_studies(archived).items():
                names.extend(hpo_network_name(study, run) for run in runs)
    return sorted(names)


def get_hpo_studies(archived: bool = False) -> dict[str, list[str]]:
    """List HPO study slugs and their direct trial-network slugs."""
    root = Filepaths.HPO_ARCHIVE if archived else Filepaths.HPO_ROOT
    if not root.exists():
        return {}
    return {
        study.name: sorted(
            run.name for run in study.iterdir() if run.is_dir() and (run / "network.flax").exists()
        )
        for study in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "_archive")
    }


def get_available_commits(networks: list[str]) -> list[str]:
    commits = set()
    for name in networks:
        try:
            study, _ = split_hpo_name(name) if is_hpo_name(name) else (name, None)
            commits.add(parse_slug(study)[2])
        except ValueError:
            continue
    return sorted(commits)


def filter_networks_by_commit(networks: list[str], commit: str | None) -> list[str]:
    if not commit or commit == "All":
        return networks
    filtered = []
    for name in networks:
        try:
            study, _ = split_hpo_name(name) if is_hpo_name(name) else (name, None)
            if parse_slug(study)[2] == commit:
                filtered.append(name)
        except ValueError:
            continue
    return filtered


def resolve_run_directory(name: str) -> Path:
    """Resolve a single slug or hpo/<study>/<run> network name."""
    if is_hpo_name(name):
        study, run = split_hpo_name(name)
        if run is None:
            raise FileNotFoundError(f"HPO run missing from name: {name}")
        for root in (Filepaths.HPO_ROOT, Filepaths.HPO_ARCHIVE):
            path = root / study / run
            if path.is_dir():
                return path
        raise FileNotFoundError(f"Run dir not found: {name}")
    if "/" in name or name.startswith("."):
        raise FileNotFoundError(f"Invalid network name: {name}")
    for root in RUN_ROOTS:
        path = root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Run dir not found: {name}")


def resolve_study_directory(study: str) -> Path:
    """Resolve an HPO study slug from its live or archived root."""
    if "/" in study or study.startswith("."):
        raise FileNotFoundError(f"Invalid HPO study slug: {study}")
    for root in (Filepaths.HPO_ROOT, Filepaths.HPO_ARCHIVE):
        path = root / study
        if path.is_dir():
            return path
    raise FileNotFoundError(f"HPO study not found: {study}")


def move_run_dir(old_name: str, new_path: Path) -> Path:
    """Move a run dir to a new path and return the target."""
    old_path = resolve_run_directory(old_name)
    if new_path.exists():
        raise FileExistsError(f"Target already exists: {new_path}")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))
    if is_hpo_name(old_name):
        with contextlib.suppress(OSError):
            old_path.parent.rmdir()
    return new_path


def move_study_dir(study: str, new_path: Path) -> Path:
    """Move an HPO study directory to a new path and return it."""
    old_path = resolve_study_directory(study)
    if new_path.exists():
        raise FileExistsError(f"Target already exists: {new_path}")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))
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
