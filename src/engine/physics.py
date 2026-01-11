from functools import partial

import jax
import jax.numpy as jnp

MU_0 = 4 * jnp.pi * 1e-7
PSI_EDGE = 0  # Poloidal flux at plasma boundary
WEIGHT_BOUNDARY_CONDITION = 100.0  # Weight for boundary loss terms


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
    psi_norm = (psi - psi_axis) / (PSI_EDGE - psi_axis)
    return F_axis * (1.0 - psi_norm**exponent)


def pressure_profile(
    psi: jnp.ndarray,
    psi_axis: float,
    p0: float,
    alpha: float = 1.0,
) -> jnp.ndarray:
    """Pressure profile: p(ψ).

    Calculates the plasma pressure.
    Note: Making pressure a function solely of ψ is a simplifying assumption
    standard in static equilibrium MHD.

    Args:
        psi: Poloidal flux ψ(R, Z).
        psi_axis: Flux at the magnetic axis.
        p0: Pressure at the magnetic axis.
        alpha: Profile shape parameter.

    Returns:
        Calculated pressure p(ψ).
    """
    psi_norm = (psi - psi_axis) / (PSI_EDGE - psi_axis)
    return p0 * (1.0 - psi_norm**alpha)


def shafranov_operator(psi_fn: callable, params: jnp.ndarray, R: float, Z: float) -> float:
    """Computes the Shafranov operator Δ*ψ for a given point (R, Z).

    Args:
        psi_fn: A function that takes (params, R, Z) and returns a scalar ψ.
        params: Neural network parameters.
        R, Z: Scalars or 0D arrays representing the coordinates.

    Returns:
        Δ*ψ as a scalar.
    """
    R_stable = R + 1e-8

    # Define gradient function w.r.t (R, Z)
    def grad_psi(r, z):  # noqa
        return jax.grad(psi_fn, argnums=(1, 2))(params, r, z)

    # Use JVP to compute gradients and diagonal Hessian terms efficiently
    # JVP 1: R-direction (1, 0) -> Gets (dpsi_dR, dpsi_dZ) and (d2psi_dR2, d2psi_dZdR)
    (dpsi_dR, _), (d2psi_dR2, _) = jax.jvp(grad_psi, (R_stable, Z), (1.0, 0.0))

    # JVP 2: Z-direction (0, 1) -> Gets (dpsi_dR, dpsi_dZ) and (d2psi_dRdZ, d2psi_dZ2)
    # We only need the second component of the tangent (d2psi_dZ2)
    _, (_, d2psi_dZ2) = jax.jvp(grad_psi, (R_stable, Z), (0.0, 1.0))

    return d2psi_dR2 - (1.0 / R_stable) * dpsi_dR + d2psi_dZ2


def grad_shafranov_residual(
    psi_fn: callable,
    params: jnp.ndarray,
    R: float,
    Z: float,
    psi_axis: float,
    p0: float,
    F_axis: float,
    pressure_alpha: float = 1.0,
    field_exponent: float = 1.0,
) -> float:
    """Compute the residual of the Grad-Shafranov equation at a given point.

    The Grad-Shafranov equation describes the force balance in an axisymmetric
    plasma equilibrium. A residual of zero indicates a valid physical state.

    Math:
        Residual = Δ*ψ + μ₀ R² p'(ψ) + F(ψ)F'(ψ)

    Args:
        psi_fn: Function computing ψ(R, Z) from parameters.
        params: Neural network parameters.
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
    # 1. Compute Operator (LHS)
    delta_star = shafranov_operator(psi_fn, params, R, Z)

    # 2. Compute State
    psi = psi_fn(params, R, Z)

    # 3. Compute Source Terms (RHS)
    # Pressure gradient term: p'(ψ)
    def p_fn(p: jnp.ndarray) -> jnp.ndarray:
        return pressure_profile(p, psi_axis, p0, pressure_alpha)

    # Use JVP for scalar derivative (forward mode)
    _, dp_dpsi = jax.jvp(p_fn, (psi,), (1.0,))

    # Toroidal field term: F(ψ)F'(ψ)
    def f_fn(f: jnp.ndarray) -> jnp.ndarray:
        return toroidal_field_flux_function(f, psi_axis, F_axis, field_exponent)

    # Use JVP to get both value and gradient in one pass
    F_val, dF_dpsi = jax.jvp(f_fn, (psi,), (1.0,))

    # 4. Assemble Equation
    # GS Eq: Δ*ψ = -μ₀ R² p' - F F'
    rhs = -(MU_0 * R**2 * dp_dpsi) - (F_val * dF_dpsi)

    return delta_star - rhs


# psi_fn must be makred as static for JIT compilation
@partial(jax.jit, static_argnums=(0,))
def pinn_loss_function(
    psi_fn: callable,
    params: jnp.ndarray,
    R_sample: jnp.ndarray,
    Z_sample: jnp.ndarray,
    boundary_R: jnp.ndarray,
    boundary_Z: jnp.ndarray,
    boundary_dR_dtheta: jnp.ndarray,
    boundary_dZ_dtheta: jnp.ndarray,
    p0: float,
    F_axis: float,
) -> float:
    """Computes the total PINN loss: L_total = L_residual + L_boundary"""
    # --- 1. Dynamic Axis Estimation ---
    # Predict psi for interior points
    psi_interior = jax.vmap(lambda r, z: psi_fn(params, r, z))(R_sample, Z_sample)

    # Heuristic: Estimate psi_axis by getting the minimum (magnetic axis) for this batch
    #            Assumes sufficiently large batchsize for good approximation
    # stop_gradient ensures we don't backprop through the choice of the axis value itself
    psi_axis_est = jax.lax.stop_gradient(jnp.min(psi_interior))

    # --- 2. Physics Residual Loss ---
    batch_residual_fn = jax.vmap(
        lambda r, z: grad_shafranov_residual(psi_fn, params, r, z, psi_axis_est, p0, F_axis)
    )
    residuals = batch_residual_fn(R_sample, Z_sample)
    loss_physics = jnp.mean(residuals**2)

    # --- 3. Boundary Condition Loss (Neumann) ---
    batch_boundary_gradients_psi = jax.vmap(jax.grad(psi_fn, argnums=(1, 2)), in_axes=(None, 0, 0))
    dpsi_dR_boundary, dpsi_dZ_boundary = batch_boundary_gradients_psi(
        params, boundary_R, boundary_Z
    )

    norm_grad = dpsi_dR_boundary * boundary_dZ_dtheta - dpsi_dZ_boundary * boundary_dR_dtheta
    loss_neumann = jnp.mean(norm_grad**2)

    # --- 4. Dirichlet Boundary Condition ---
    psi_boundary_pred = jax.vmap(lambda r, z: psi_fn(params, r, z))(boundary_R, boundary_Z)
    loss_dirichlet = jnp.mean((psi_boundary_pred - PSI_EDGE) ** 2)

    return loss_physics + WEIGHT_BOUNDARY_CONDITION * (loss_neumann + loss_dirichlet)
