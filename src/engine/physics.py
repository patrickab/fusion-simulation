from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from src.lib.geometry_config import PlasmaConfig

MU_0 = 4 * jnp.pi * 1e-7
PSI_EDGE = 0.0  # Poloidal flux at plasma boundary


def estimate_psi_axis(psi: jnp.ndarray) -> jnp.ndarray:
    """Magnetic-axis flux: mean of the top-|ψ| collocation samples (signed).

    Sign-agnostic so both ψ>0-at-axis (hard-BC envelope) and ψ<0-at-axis
    (legacy soft-BC) conventions pick the true extremum, not the edge at ψ≈0.
    """
    n_axis = max(1, psi.shape[0] // 20)
    idx = jnp.argsort(jnp.abs(psi))[-n_axis:]
    return jax.lax.stop_gradient(jnp.mean(psi[idx]))


PsiFn = Callable[[any, jnp.ndarray, jnp.ndarray, PlasmaConfig], jnp.ndarray]


def toroidal_field_flux_function(
    psi: jnp.ndarray,
    psi_axis: float,
    F_axis: float,
    exponent: float = 1.0,
    flux_scale: float = 1.0,
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
        flux_scale: Characteristic flux magnitude for this config (F_axis * a),
            used to floor the clamp below relative to the config's own scale.

    Returns:
        Calculated F(ψ) values.
    """
    # Signed flux depth: psi_norm must run 0→1 from axis to edge, which
    # requires dividing by (psi_edge - psi_axis) with its sign intact.
    # The magnitude floor prevents division by ~0 during early training
    # (random weights); sign is preserved so the profile direction is
    # correct for either ψ convention (ψ>0 axis or ψ<0 axis).
    raw_depth = PSI_EDGE - psi_axis
    flux_depth = jnp.where(
        raw_depth >= 0,
        jnp.maximum(raw_depth, 1e-3 * jnp.abs(flux_scale)),
        jnp.minimum(raw_depth, -1e-3 * jnp.abs(flux_scale)),
    )

    psi_norm = (psi - psi_axis) / flux_depth
    # Use softplus for C-infinity continuity (smooth gradients for Shafranov operator)
    # approximates clamp(psi_norm, 0, 1)
    base = jax.nn.softplus(psi_norm) - jax.nn.softplus(psi_norm - 1.0)
    return F_axis * (1.0 - (base + 1e-8) ** exponent)


def pressure_profile(
    psi: jnp.ndarray,
    psi_axis: float,
    p0: float,
    alpha: float = 1.0,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """Pressure profile: p(ψ).

    Simplifying assumption: Calculates the plasma pressure as a function of ψ

    Args:
        psi: Poloidal flux ψ(R, Z).
        psi_axis: Flux at the magnetic axis.
        p0: Pressure at the magnetic axis.
        alpha: Profile shape parameter.
        flux_scale: Characteristic flux magnitude for this config (F_axis * a),
            used to floor the clamp below relative to the config's own scale.

    Returns:
        Calculated pressure p(ψ).
    """
    # Signed flux depth: see toroidal_field_flux_function for rationale.
    raw_depth = PSI_EDGE - psi_axis
    flux_depth = jnp.where(
        raw_depth >= 0,
        jnp.maximum(raw_depth, 1e-3 * jnp.abs(flux_scale)),
        jnp.minimum(raw_depth, -1e-3 * jnp.abs(flux_scale)),
    )

    psi_norm = (psi - psi_axis) / flux_depth
    # Use softplus for C-infinity continuity
    base = jax.nn.softplus(psi_norm) - jax.nn.softplus(psi_norm - 1.0)
    return p0 * (1.0 - (base + 1e-8) ** alpha)


def _second_derivative(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Diagonal second derivative via JVP-over-JVP (forward-mode only).

    Preferred over:
        jax.hessian / jax.grad:
            - allocates reverse-mode tapes
            - computes full Hessian but only diagonal is needed.
        jacfwd-over-jacfwd:
            - traces fn twice, preventing XLA from sharing primal evaluation across both passes.

    A single jax.jvp with a unit tangent yields (f(x), f'(x)) in one forward
    pass. Nesting twice gives f''(x) with no cross-derivatives and no tapes.
    Measured speedup: 1.31x vs jax.grad + jax.jvp.
    """
    d_fn = lambda x_val: jax.jvp(fn, (x_val,), (jnp.ones_like(x_val),))[1]  # noqa
    return jax.jvp(d_fn, (x,), (jnp.ones_like(x),))[1]


# Rematerialize the high-order PDE graph while retaining its primal flux for profile losses.
@partial(jax.checkpoint, static_argnums=0)
def shafranov_operator_and_psi(
    psi_fn: Callable,
    params: any,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    *args: any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the Grad-Shafranov operator Δ*ψ and primal ψ.

    Δ*ψ = ∂²ψ/∂R² - (1/R)(∂ψ/∂R) + ∂²ψ/∂Z²

    The nested R pass returns the primal, first, and second terms together,
    avoiding a separate network traversal for dpsi/dR.
    """
    R_stable = R + 1e-8

    # 1D closures - freeze non-active variables for the compiler
    psi_along_R = lambda r: psi_fn(params, r, Z, *args)  # noqa
    psi_along_Z = lambda z: psi_fn(params, R_stable, z, *args)  # noqa

    tangent_R = jnp.ones_like(R_stable)
    first_order_R = lambda r: jax.jvp(psi_along_R, (r,), (tangent_R,))  # noqa
    (psi, dpsi_dR), (_, d2psi_dR2) = jax.jvp(first_order_R, (R_stable,), (tangent_R,))
    d2psi_dZ2 = _second_derivative(psi_along_Z, Z)

    delta_star = d2psi_dR2 - (1.0 / R_stable) * dpsi_dR + d2psi_dZ2
    return delta_star, psi


def _grad_shafranov_residual_from_operator(
    delta_star: jnp.ndarray,
    psi: jnp.ndarray,
    R: jnp.ndarray,
    psi_axis: float,
    config: PlasmaConfig,
) -> jnp.ndarray:
    flux_scale = config.State.F_axis * config.Geometry.a

    dp_dpsi = jax.jvp(
        lambda p: pressure_profile(
            p, psi_axis, config.State.p0, config.State.pressure_alpha, flux_scale
        ),
        (psi,),
        (jnp.ones_like(psi),),
    )[1]

    F_val, dF_dpsi = jax.jvp(
        lambda p: toroidal_field_flux_function(
            p, psi_axis, config.State.F_axis, config.State.field_exponent, flux_scale
        ),
        (psi,),
        (jnp.ones_like(psi),),
    )

    rhs = -(MU_0 * R**2 * dp_dpsi) - (F_val * dF_dpsi)

    # Normalize by magnetic pressure scale (B_toroidal^2)
    # This handles high-field/low-beta regimes robustly
    scale = (config.State.F_axis / config.Geometry.R0) ** 2 + 1.0
    return (delta_star - rhs) / scale


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
    delta_star, psi = shafranov_operator_and_psi(psi_fn, params, R, Z, config)
    return _grad_shafranov_residual_from_operator(delta_star, psi, R, psi_axis, config)


@partial(jax.jit, static_argnums=(0,), static_argnames=("soft_bc",))
def pinn_loss_function(
    psi_fn: callable,
    params: any,
    R_interior: jnp.ndarray,
    Z_interior: jnp.ndarray,
    batch_config: PlasmaConfig,
    weight_boundary_condition: float,
    huber_delta: float,
    *,
    weight_flux_scale: float = 1.0,
    soft_bc: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes the total PINN loss.

    Hard-BC (default): L = L_residual + w_scale·L_flux_scale. Dirichlet BC is
    structural via the envelope; the BC term is diagnostic-only.

    Soft-BC (``soft_bc=True``): legacy-style L = L_residual + w_scale·L_flux_scale
    + w_bc·(L_dirichlet + L_neumann) on raw ψ without an envelope.
    """

    def single_config_loss(
        R: jnp.ndarray, Z: jnp.ndarray, config: PlasmaConfig
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # 1. Axis for PDE profiles: |ψ| extremum (stop-grad — discovery, not trainable).
        delta_star, psi_int = jax.vmap(
            lambda r, z: shafranov_operator_and_psi(psi_fn, params, r, z, config)
        )(R, Z)
        psi_axis = estimate_psi_axis(psi_int)

        # 2. Collapse guard: hinge on the interior-mean flux. The GS equation
        # sets the amplitude self-consistently (legacy converged ~20-40 Wb with
        # no anchor), so an equality target would fight the PDE; the hinge only
        # forbids the trivial ψ≈0 minimum and pins the ψ>0-at-axis convention.
        # A pointwise anchor at (R₀,0) is exploitable by Fourier-feature nets —
        # one sharp spike at the anchor zeroes the hinge while the bulk stays ≈0
        # (observed in run pinn_2026_07_12_19_44_42); the mean anchors the bulk.
        # Floor 0.05·F_axis·a sits below the legacy-quality bulk mean on all
        # eval configs, so the hinge (and its gradient) vanish for good solutions.
        floor = 0.05 * config.State.F_axis * config.Geometry.a
        loss_scale = jnp.maximum(0.0, 1.0 - jnp.mean(psi_int) / floor) ** 2

        # 3. PDE Residual Loss
        res_vals = _grad_shafranov_residual_from_operator(delta_star, psi_int, R, psi_axis, config)
        loss_res = jnp.where(
            huber_delta > 0,
            jnp.mean(optax.losses.huber_loss(res_vals, delta=huber_delta)),
            jnp.mean(res_vals**2),
        )

        # 4. Boundary losses
        R_b, Z_b = config.Boundary.R, config.Boundary.Z
        psi_b = jax.vmap(lambda r, z: psi_fn(params, r, z, config))(R_b, Z_b)
        loss_dir = jnp.mean((psi_b - PSI_EDGE) ** 2)
        if soft_bc:

            def grad_psi(r: jnp.ndarray, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
                return jax.grad(psi_fn, argnums=(1, 2))(params, r, z, config)

            dR_b, dZ_b = jax.vmap(grad_psi)(R_b, Z_b)
            dpsi_dt = dR_b * config.Boundary.dR_dtheta + dZ_b * config.Boundary.dZ_dtheta
            loss_boundary = loss_dir + jnp.mean(dpsi_dt**2)
        else:
            loss_boundary = jax.lax.stop_gradient(loss_dir)

        return loss_res, loss_boundary, loss_scale

    losses = jax.vmap(single_config_loss)(R_interior, Z_interior, batch_config)
    loss_res = jnp.mean(losses[0])
    loss_boundary = jnp.mean(losses[1])
    loss_scale = jnp.mean(losses[2])
    # Adaptive resampling ranks configs by per_config_loss — only include terms
    # that actually train (hard-BC boundary loss is stop-gradient diagnostic).
    per_config_loss = losses[0] + weight_flux_scale * losses[2]
    total = loss_res + weight_flux_scale * loss_scale
    if soft_bc:
        per_config_loss = per_config_loss + weight_boundary_condition * losses[1]
        total = total + weight_boundary_condition * loss_boundary
    return total, loss_res, loss_boundary, per_config_loss


def get_b_field(
    psi_fn: PsiFn,
    params: any,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    config: PlasmaConfig,
) -> jnp.ndarray:
    """Calculate magnetic field from flux function via Grad-Shafranov equilibrium.

    Theory: B = ∇ψ * ∇φ + F(ψ)∇φ in axisymmetric geometry
    Yields: B_R = -(1/R)∂ψ/∂Z, B_Z = (1/R)∂ψ/∂R, B_φ = F(ψ)/R

    Args:
        psi_fn: Function computing ψ(params, R, Z, config).
        params: Neural network parameters or any psi_fn state.
        R: Major radial coordinates [m].
        Z: Vertical coordinates [m].
        config: Plasma geometry and state parameters.

    Returns:
        (N, 3) array of [B_R, B_Z, B_φ] in Tesla.
    """
    R_arr = jnp.atleast_1d(R).flatten()
    Z_arr = jnp.atleast_1d(Z).flatten()

    def scalar_psi(r: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        rn = jnp.asarray(r)
        zn = jnp.asarray(z)
        return psi_fn(params, rn, zn, config).squeeze()

    grad_psi_fn = jax.vmap(jax.grad(scalar_psi, argnums=(0, 1)))
    dpsi_dR, dpsi_dZ = grad_psi_fn(R_arr, Z_arr)

    BR = -dpsi_dZ / R_arr
    BZ = dpsi_dR / R_arr

    psi_vals = jax.vmap(scalar_psi)(R_arr, Z_arr)
    psi_axis = scalar_psi(config.Geometry.R0, 0.0)

    flux_scale = config.State.F_axis * config.Geometry.a
    F_val = toroidal_field_flux_function(
        psi_vals, psi_axis, config.State.F_axis, config.State.field_exponent, flux_scale
    )
    Bphi = F_val / R_arr

    return jnp.column_stack([BR, BZ, Bphi])


def get_b_field_cartesian(
    psi_fn: PsiFn,
    params: any,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    config: PlasmaConfig,
) -> jnp.ndarray:
    """Transform cylindrical B-field to Cartesian coordinates.

    Coordinate transform: (R, φ, Z) → (X, Y, Z) where R = √(X²+Y²), φ = atan2(Y, X)
    Basis transform: e_R = cos(φ)e_X + sin(φ)e_Y, e_φ = -sin(φ)e_X + cos(φ)e_Y

    Args:
        psi_fn: Function computing ψ(params, R, Z, config).
        params: Neural network parameters or any psi_fn state.
        X: Cartesian X coordinates [m].
        Y: Cartesian Y coordinates [m].
        Z: Cartesian Z coordinates [m].
        config: Plasma geometry and state parameters.

    Returns:
        (N, 3) array of [B_X, B_Y, B_Z] in Tesla.
    """
    X_arr = jnp.asarray(X)
    Y_arr = jnp.asarray(Y)
    Z_arr = jnp.asarray(Z)

    R = jnp.sqrt(X_arr**2 + Y_arr**2)
    phi = jnp.arctan2(Y_arr, X_arr)

    B_cyl = get_b_field(psi_fn, params, R, Z_arr, config)
    BR, BZ_cyl, Bphi = B_cyl[:, 0], B_cyl[:, 1], B_cyl[:, 2]

    BX = BR * jnp.cos(phi) - Bphi * jnp.sin(phi)
    BY = BR * jnp.sin(phi) + Bphi * jnp.cos(phi)

    return jnp.column_stack([BX, BY, BZ_cyl])
