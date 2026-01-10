"""Module for parametric expression of a tokamak plasma surfaces"""

import jax
import jax.numpy as jnp

from src.lib.geometry_config import (
    FusionPlasma,
    PlasmaBoundary,
    PlasmaConfig,
    RotationalAngles,
)


def get_poloidal_points(theta: float, plasma_config: PlasmaConfig) -> tuple[float, float]:
    """
    Calculates a single (R, Z) point for a given theta.
    Intentionally not vectorized to perform Nx2 instead of NxN jacobian computations.
    """
    # unpack parameters for readability
    major_radius = plasma_config.R0
    minor_radius = plasma_config.a
    triangularity = plasma_config.delta
    elongation = plasma_config.kappa

    shaped_theta = theta + triangularity * jnp.sin(theta)

    # calculate coordinates
    R = major_radius + minor_radius * jnp.cos(shaped_theta)
    Z = elongation * minor_radius * jnp.sin(theta)
    return R, Z


def calculate_poloidal_boundary(
    theta: jnp.ndarray, plasma_config: PlasmaConfig, phi: float = 0.0
) -> PlasmaBoundary:
    """Compute R-Z coordinates for a shaped tokamak plasma boundary.

    Args:
        theta: poloidal angles
        plasma_config: geometric parameters (R0, a, kappa, delta)
        phi: toroidal angle (rad)

    Returns:
        2D boundary coordinates
    """
    # Use jax.jvp to compute values and derivatives simultaneously.
    # Since get_poloidal_points is element-wise, passing a tangent of ones
    # correctly computes the element-wise gradients for both scalars and arrays.
    tangent = jnp.ones_like(theta)
    (R, Z), (dR_dtheta, dZ_dtheta) = jax.jvp(
        lambda t: get_poloidal_points(t, plasma_config), (theta,), (tangent,)
    )

    return PlasmaBoundary(
        R=R,
        Z=Z,
        theta=theta,
        dR_dtheta=dR_dtheta,
        dZ_dtheta=dZ_dtheta,
        R_center=plasma_config.R0,
        Z_center=0.0,
        phi=phi,
    )


def calculate_fusion_plasma(plasma_boundary: PlasmaBoundary) -> FusionPlasma:
    """Revolve 2D poloidal cross-section into 3D toroidal volume via rotational symmetry.

    The function maps the (R, Z) boundary across a range of toroidal angles (φ) to create
    a parametric mesh, then transforms the resulting cylindrical coordinates into
    Cartesian (X, Y, Z) space.

    Args:
        plasma_boundary: The 2D R-Z coordinates defining the plasma edge.

    Returns:
        A FusionPlasma object containing the 3D Cartesian mesh and original boundary.
    """
    R_poloidal = plasma_boundary.R
    Z_poloidal = plasma_boundary.Z
    theta_poloidal = plasma_boundary.theta

    # 2D meshgrid for revolution: each row is a toroidal angle, each column a poloidal point
    phi = RotationalAngles.PHI
    R_grid, phi_grid = jnp.meshgrid(R_poloidal, phi)

    # Repeat Z and theta along the toroidal direction
    Z_grid = jnp.tile(Z_poloidal, (RotationalAngles.n_phi, 1))
    theta_grid = jnp.tile(theta_poloidal, (RotationalAngles.n_phi, 1))

    # Convert cylindrical (R, φ, Z) → Cartesian (X, Y, Z)
    X = R_grid * jnp.cos(phi_grid)
    Y = R_grid * jnp.sin(phi_grid)
    Z = Z_grid

    return FusionPlasma(
        X=X,
        Y=Y,
        Z=Z,
        R=R_grid,
        phi=phi_grid,
        theta=theta_grid,
        R_center=plasma_boundary.R_center,
        Z_center=plasma_boundary.Z_center,
        Boundary=plasma_boundary,
    )
