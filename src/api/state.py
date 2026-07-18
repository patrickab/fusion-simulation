"""In-process cache of loaded NetworkManagers, keyed by checkpoint name.

Mirrors the role `st.session_state` played in the Streamlit app: avoids
re-tracing JAX/JIT on every request for the same checkpoint.
"""

from fastapi import HTTPException

from src.engine.network import NetworkManager
from src.engine.residual_correction import load_checkpoint

_managers: dict[str, NetworkManager] = {}


def get_manager(name: str) -> NetworkManager:
    """Load (or reuse) the (network or composed) manager for a checkpoint name."""
    if name in _managers:
        return _managers[name]

    try:
        manager = load_checkpoint(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None

    _managers[name] = manager
    return manager


def invalidate(name: str) -> None:
    """Drop one cached manager."""
    _managers.pop(name, None)


def invalidate_prefix(prefix: str) -> None:
    """Drop cached managers below a renamed, archived, or deleted study."""
    for name in [key for key in _managers if key.startswith(prefix)]:
        _managers.pop(name, None)
