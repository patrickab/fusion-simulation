"""In-process cache of loaded NetworkManagers, keyed by checkpoint name.

Mirrors the role `st.session_state` played in the Streamlit app: avoids
re-tracing JAX/JIT on every request for the same checkpoint.
"""

from pathlib import Path

from fastapi import HTTPException

from src.engine.network import NetworkManager
from src.lib.config import Filepaths
from src.lib.network_config import HyperParams

_managers: dict[str, NetworkManager] = {}


def resolve_run_directory(name: str) -> Path:
    """Resolve a 'commit/run' network name to its benchmark run dir.

    Checks the archive too so archived networks still load.
    """
    p = Filepaths.BENCHMARKS / name
    if p.is_dir():
        return p
    p = Filepaths.BENCHMARK_ARCHIVE / name
    if p.is_dir():
        return p
    raise HTTPException(404, f"Run dir not found: {name}")


def get_manager(name: str) -> NetworkManager:
    """Load (or reuse) the NetworkManager for a given checkpoint name."""
    if name in _managers:
        return _managers[name]

    run_dir = resolve_run_directory(name)
    config_path = run_dir / "config.json"
    config = HyperParams.from_json(config_path) if config_path.exists() else HyperParams()

    manager = NetworkManager(config)
    pinn_path = run_dir / "network.flax"
    if pinn_path.exists():
        params = manager.from_disk(pinn_path=pinn_path)
        manager.state = manager.state.replace(params=params)

    _managers[name] = manager
    return manager


def invalidate(name: str) -> None:
    _managers.pop(name, None)
