from functools import partial

import jax
import jax.numpy as jnp

from src.lib.geometry_config import PlasmaConfig

MU_0 = 4 * jnp.pi * 1e-7
PSI_EDGE = 0.0  # Poloidal flux at plasma boundary
WEIGHT_BOUNDARY_CONDITION = 100.0


def toroidal_field_flux_function(
    psi: jnp.ndarray,
    psi_axis: float,
    F_axis: float,
    exponent: float = 1.0,
) -> jnp.ndarray:
    """Compute the toroidal field flux function F(ψ) = R B_φ.

    In ideal MHD, F(ψ) determines the toroidal magnetic field distribution.
    Asumes a simplified parametric profile where F decreases from the magnetic
    axis to the edge, governed by a power law.

    Math:
        F(ψ) = F_axis * (1 - ψ_norm^exponent)
        where ψ_norm = (ψ - ψ_axis) / (ψ_edge - ψ_axis)

    Args:
        psi: Poloidal flux ψ(R, Z).
        psi_axis: Flux at the magnetic axis.
        psi_edge: Flux at the plasma boundary.
        F_axis: Value of F at the magnetic axis (R * B_phi).
        exponent: Profile shape parameter (1.0 = linear).

    Returns:
        Calculated F(ψ) values.
    """
    # Ensure numerical stability: clamp depth to 1.0 to prevent gradient explosion
    # Use abs() to handle initial random weights where psi_axis > PSI_EDGE
    flux_depth = jnp.maximum(jnp.abs(PSI_EDGE - psi_axis), 1.0)

    psi_norm = (psi - psi_axis) / flux_depth
    base = jnp.maximum(0.0, jnp.minimum(1.0, psi_norm))
    return F_axis * (1.0 - (base + 1e-8) ** exponent)


def pressure_profile(
    psi: jnp.ndarray,
    psi_axis: float,
    p0: float,
    alpha: float = 1.0,
) -> jnp.ndarray:
    """Pressure profile: p(ψ).

    Simplifying assumption: Calculates the plasma pressure as a function of ψ

    Args:
        psi: Poloidal flux ψ(R, Z).
        psi_axis: Flux at the magnetic axis.
        p0: Pressure at the magnetic axis.
        alpha: Profile shape parameter.

    Returns:
        Calculated pressure p(ψ).
    """
    # Ensure numerical stability: clamp depth to 1.0 to prevent gradient explosion
    flux_depth = jnp.maximum(jnp.abs(PSI_EDGE - psi_axis), 1.0)

    psi_norm = (psi - psi_axis) / flux_depth
    base = jnp.maximum(0.0, jnp.minimum(1.0, psi_norm))
    return p0 * (1.0 - (base + 1e-8) ** alpha)


def shafranov_operator(
    psi_fn: callable,
    params: any,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    *args: any,
) -> jnp.ndarray:
    """Computes the Shafranov operator Δ*ψ for a given point (R, Z).

    Args:
        psi_fn: A function that takes (params, R, Z, *args) and returns a scalar ψ.
        params: Neural network parameters (PyTree/Dict).
        R, Z: Scalars or 0D arrays representing the coordinates.
        *args: Additional arguments passed to psi_fn (e.g., PlasmaConfig).

    Returns:
        Δ*ψ as a scalar.
    """
    R_stable = R + 1e-8

    def grad_psi(r: jnp.ndarray, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(psi_fn, argnums=(1, 2))(params, r, z, *args)

    (dpsi_dR, _), (d2psi_dR2, _) = jax.jvp(grad_psi, (R_stable, Z), (1.0, 0.0))
    _, (_, d2psi_dZ2) = jax.jvp(grad_psi, (R_stable, Z), (0.0, 1.0))

    return d2psi_dR2 - (1.0 / R_stable) * dpsi_dR + d2psi_dZ2


def grad_shafranov_residual(
    psi_fn: callable,
    params: any,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    psi_axis: float,
    config: PlasmaConfig,
) -> jnp.ndarray:
    """Compute the residual of the Grad-Shafranov equation at a given point.

    The Grad-Shafranov equation describes the force balance in an axisymmetric
    plasma equilibrium. A residual of zero indicates a valid physical state.

    Math:
        Residual = Δ*ψ + μ₀ R² p'(ψ) + F(ψ)F'(ψ)

    Args:
        psi_fn: Function computing ψ(R, Z) from parameters.
        params: Neural network parameters (PyTree).
        R: Radial coordinate.
        Z: Vertical coordinate.
        psi_axis: Flux at the magnetic axis.
        psi_edge: Flux at the plasma boundary.
        p0: Pressure at the magnetic axis.
        F_axis: Toroidal field flux function value at the axis.
        pressure_alpha: Shape parameter for pressure profile.
        field_exponent: Shape parameter for F(ψ) profile.

    Returns:
        The residual value (error) at the specified point.
    """
    delta_star = shafranov_operator(psi_fn, params, R, Z, config)
    psi = psi_fn(params, R, Z, config)

    # Profiles from config
    p0, alpha = config.State.p0, config.State.pressure_alpha
    F_axis, exponent = config.State.F_axis, config.State.field_exponent

    # Compute gradients via JVP
    _, dp_dpsi = jax.jvp(lambda p: pressure_profile(p, psi_axis, p0, alpha), (psi,), (1.0,))
    F_val, dF_dpsi = jax.jvp(
        lambda f: toroidal_field_flux_function(f, psi_axis, F_axis, exponent), (psi,), (1.0,)
    )

    rhs = -(MU_0 * R**2 * dp_dpsi) - (F_val * dF_dpsi)

    # Normalize by magnetic pressure scale (B_toroidal^2)
    # This handles high-field/low-beta regimes robustly
    scale = (F_axis / config.Geometry.R0) ** 2 + 1.0

    return (delta_star - rhs) / scale


@partial(jax.jit, static_argnums=(0,))
def pinn_loss_function(
    psi_fn: callable,
    params: any,
    R_interior: jnp.ndarray,
    Z_interior: jnp.ndarray,
    batch_config: PlasmaConfig,
) -> jnp.ndarray:
    """Computes the total PINN loss: L_total = L_residual + L_boundary

    Uses a double-vectorization strategy.

    Design Reasoning:
    1. Outer Vmap: Iterates over the batch of different plasma configurations.
    2. Inner Vmap: Iterates over the spatial samples (interior and boundary).

    This allows JAX to perform the entire forward pass and gradient calculation
    for the training batch in parallel, effectively fuses thousands of operations
    into optimized GPU kernels.
    """

    def single_config_loss(R: jnp.ndarray, Z: jnp.ndarray, config: PlasmaConfig) -> jnp.ndarray:
        # 1. Axis Estimation
        psi_int = jax.vmap(lambda r, z: psi_fn(params, r, z, config))(R, Z)
        psi_axis = jax.lax.stop_gradient(jnp.min(psi_int))

        # 2. PDE Residual Loss
        residual_fn = jax.vmap(
            lambda r, z: grad_shafranov_residual(psi_fn, params, r, z, psi_axis, config)
        )
        loss_res = jnp.mean(residual_fn(R, Z) ** 2)

        # 3. Boundary Condition Loss (Dirichlet & Neumann)
        R_b, Z_b = config.Boundary.R, config.Boundary.Z
        psi_b = jax.vmap(lambda r, z: psi_fn(params, r, z, config))(R_b, Z_b)
        loss_dir = jnp.mean((psi_b - PSI_EDGE) ** 2)

        def grad_psi(r: jnp.ndarray, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return jax.grad(psi_fn, argnums=(1, 2))(params, r, z, config)

        dR_b, dZ_b = jax.vmap(grad_psi)(R_b, Z_b)
        dpsi_dt = dR_b * config.Boundary.dR_dtheta + dZ_b * config.Boundary.dZ_dtheta
        loss_neu = jnp.mean(dpsi_dt**2)

        return loss_res + WEIGHT_BOUNDARY_CONDITION * (loss_dir + loss_neu)

    return jnp.mean(jax.vmap(single_config_loss)(R_interior, Z_interior, batch_config))
