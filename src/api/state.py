"""In-process cache of loaded NetworkManagers, keyed by checkpoint name.

Mirrors the role `st.session_state` played in the Streamlit app: avoids
re-tracing JAX/JIT on every request for the same checkpoint.
"""

from src.engine.network import NetworkManager
from src.lib.config import Filepaths
from src.lib.network_config import HyperParams

_managers: dict[str, NetworkManager] = {}


def get_manager(name: str) -> NetworkManager:
    """Load (or reuse) the NetworkManager for a given checkpoint name."""
    if name in _managers:
        return _managers[name]

    pinn_path = Filepaths.NETWORKS / name
    config_path = pinn_path.with_suffix(".json")
    config = HyperParams.from_json(config_path) if config_path.exists() else HyperParams()

    manager = NetworkManager(config)
    if pinn_path.exists():
        params = manager.from_disk(pinn_path=pinn_path)
        manager.state = manager.state.replace(params=params)

    _managers[name] = manager
    return manager


def invalidate(name: str) -> None:
    _managers.pop(name, None)
