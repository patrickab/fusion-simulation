"""Module for parametric expression of a tokamak plasma surfaces"""

import jax.numpy as jnp

from src.lib.geometry_config import (
    FusionPlasma,
    PlasmaBoundary,
    PlasmaConfig,
    RotationalAngles,
)


def calculate_poloidal_boundary(plasma_config: PlasmaConfig) -> PlasmaBoundary:
    """
    Calculate the poloidal plasma boundary in R-Z coordinates.
    """

    # Define plasma boundary
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
    """
    Generates a 3D toroidal surface by rotating a poloidal cross-section around the Z-axis.

    This function creates a tokamak-like surface with elongation (kappa) and triangularity (delta).
    The process works as follows:

    1. Generate poloidal coordinates
    2. Create a meshgrid that extends this 2D shape into 3D space:
       - (1) np.meshgrid(R_2D, phi) creates two 2D arrays:
         * R_grid: Contains the R coordinates repeated for each toroidal angle
         * phi_grid: Contains the toroidal angles repeated for each point on the poloidal contour
       - This effectively creates a parametric surface where each toroidal section (phi=constant)
         has identical poloidal cross-sections
    3. Extend Z coordinates by repeating the Z_2D array for each toroidal angle using np.tile
    4. Transform from cylindrical coordinates (R, φ, Z) to Cartesian coordinates (X, Y, Z):
       - X = R * cos(φ)
       - Y = R * sin(φ)
       - Z remains unchanged

    This transformation maps the toroidal surface into 3D Cartesian space, where each poloidal
    cross-section is identical but rotated around the Z-axis according to the toroidal angle φ.
    """

    # Poloidal contour in R-Z
    R_poloidal = plasma_boundary.R_2d
    Z_poloidal = plasma_boundary.Z_2d

    # Toroidal angles for revolution
    phi = RotationalAngles.PHI

    # 2D meshgrid for revolution: each row is a toroidal angle, each column a poloidal point
    R_grid, phi_grid = jnp.meshgrid(R_poloidal, phi)

    # Repeat Z along the toroidal direction to match R_grid/phi_grid shape
    Z_grid = jnp.tile(Z_poloidal, (RotationalAngles.n_phi, 1))

    # Convert cylindrical (R, φ, Z) → Cartesian (X, Y, Z)
    X = R_grid * jnp.cos(phi_grid)
    Y = R_grid * jnp.sin(phi_grid)
    Z = Z_grid

    return FusionPlasma(X=X, Y=Y, Z=Z, Boundary=plasma_boundary)
