from typing import Callable

import jax
import jax.numpy as jnp

from src.engine.network import NetworkManager
from src.engine.physics import grad_shafranov_residual
from src.lib.geometry_config import PlasmaConfig
from src.lib.network_config import FluxInput


def build_psi_fn(
    manager: NetworkManager,
) -> Callable[[object, jnp.ndarray, jnp.ndarray, PlasmaConfig], jnp.ndarray]:
    """Build a ``psi_fn`` closure matching the training-time normalization/signature."""
    apply_fn = manager.model.apply

    def psi_fn(params: object, R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> jnp.ndarray:
        inp = FluxInput(R_sample=R, Z_sample=Z, config=cfg)
        p_n, r_n, z_n = inp.normalize()
        psi_n = apply_fn(params, r=r_n, z=z_n, **p_n)
        return (psi_n * cfg.State.F_axis * cfg.Geometry.a).squeeze()

    return psi_fn


def compute_gs_residual_on_points(
    manager: NetworkManager,
    config: PlasmaConfig,
    R_pts: jnp.ndarray,
    Z_pts: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate normalised Grad-Shafranov residual at supplied ``(R, Z)`` points."""
    psi_fn = build_psi_fn(manager)
    params = manager.state.params

    psi_vals = jax.vmap(lambda r, z: psi_fn(params, r, z, config))(R_pts, Z_pts)
    psi_axis = jax.lax.stop_gradient(jnp.min(psi_vals))

    residual_fn = jax.vmap(
        lambda r, z: grad_shafranov_residual(psi_fn, params, r, z, psi_axis, config)
    )
    return residual_fn(R_pts, Z_pts)
