"""In-process cache of loaded NetworkManagers, keyed by checkpoint name.

Mirrors the role `st.session_state` played in the Streamlit app: avoids
re-tracing JAX/JIT on every request for the same checkpoint.
"""

from fastapi import HTTPException

from src.engine.network import NetworkManager
from src.engine.residual_correction import load_combined
from src.lib.network_config import HyperParams
from src.streamlit.network_utils import resolve_run_directory

_managers: dict[str, NetworkManager] = {}


def get_manager(name: str) -> NetworkManager:
    """Load (or reuse) the (network or composed) manager for a checkpoint name."""
    if name in _managers:
        return _managers[name]

    try:
        run_dir = resolve_run_directory(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None

    # A run with a trained stage-2 corrector renders its composed field via a
    # NetworkManager whose make_psi_fn() yields the full stage-1 + stage-2 output.
    if (run_dir / "stage2" / "network.flax").exists():
        manager: NetworkManager = load_combined(name)
        _managers[name] = manager
        return manager

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
    """Drop one cached manager."""
    _managers.pop(name, None)


def invalidate_prefix(prefix: str) -> None:
    """Drop cached managers below a renamed, archived, or deleted study."""
    for name in [key for key in _managers if key.startswith(prefix)]:
        _managers.pop(name, None)
