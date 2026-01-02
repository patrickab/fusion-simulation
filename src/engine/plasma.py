"""Module for parametric expression of a tokamak plasma surfaces"""

import jax.numpy as jnp

from src.lib.geometry_config import (
    FusionPlasma,
    PlasmaBoundary,
    PlasmaConfig,
    RotationalAngles,
)


def calculate_poloidal_boundary(plasma_config: PlasmaConfig) -> PlasmaBoundary:
    """Compute R-Z coordinates for a shaped tokamak plasma boundary.

    Args:
        plasma_config: geometric parameters (R0, a, kappa, delta)

    Returns:
        2D boundary coordinates
    """
    theta = RotationalAngles.THETA

    # Shaping terms for readability
    major_radius = plasma_config.R0
    minor_radius = plasma_config.a
    triangularity = plasma_config.delta
    elongation = plasma_config.kappa

    # Triangularity modifies the poloidal angle
    shaped_theta = theta + triangularity * jnp.sin(theta)

    # Standard tokamak cross-section: shifted circle with shaping
    R_plasma = major_radius + minor_radius * jnp.cos(shaped_theta)
    Z_plasma = elongation * minor_radius * jnp.sin(theta)

    return PlasmaBoundary(R_2d=R_plasma, Z_2d=Z_plasma)


def generate_fusion_plasma(plasma_boundary: PlasmaBoundary) -> FusionPlasma:
    """Revolve 2D poloidal cross-section into 3D toroidal volume via rotational symmetry.

    The function maps the (R, Z) boundary across a range of toroidal angles (φ) to create
    a parametric mesh, then transforms the resulting cylindrical coordinates into
    Cartesian (X, Y, Z) space.

    Args:
        plasma_boundary: The 2D R-Z coordinates defining the plasma edge.

    Returns:
        A FusionPlasma object containing the 3D Cartesian mesh and original boundary.
    """
    R_poloidal = plasma_boundary.R_2d
    Z_poloidal = plasma_boundary.Z_2d

    # 2D meshgrid for revolution: each row is a toroidal angle, each column a poloidal point
    phi = RotationalAngles.PHI
    R_grid, phi_grid = jnp.meshgrid(R_poloidal, phi)

    # Repeat Z along the toroidal direction
    Z_grid = jnp.tile(Z_poloidal, (RotationalAngles.n_phi, 1))

    # Convert cylindrical (R, φ, Z) → Cartesian (X, Y, Z)
    X = R_grid * jnp.cos(phi_grid)
    Y = R_grid * jnp.sin(phi_grid)
    Z = Z_grid

    return FusionPlasma(X=X, Y=Y, Z=Z, Boundary=plasma_boundary)
