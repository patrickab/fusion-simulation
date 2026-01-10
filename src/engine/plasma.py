"""Module for parametric expression of a tokamak plasma surfaces"""

import jax
import jax.numpy as jnp

from src.lib.geometry_config import (
    CartesianCoordinates,
    CylindricalCoordinates,
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

    # Create cylindrical coordinates with constant phi
    phi_array = jnp.full_like(R, phi)
    coords = CylindricalCoordinates(R=R, phi=phi_array, Z=Z)

    return PlasmaBoundary(
        coords=coords,
        theta=theta,
        dR_dtheta=dR_dtheta,
        dZ_dtheta=dZ_dtheta,
        R_center=plasma_config.R0,
        Z_center=0.0,
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

    # Create coordinate objects
    cylindrical_coords = CylindricalCoordinates(R=R_grid, phi=phi_grid, Z=Z_grid)
    cartesian_coords: CartesianCoordinates = cylindrical_coords.to_cartesian()

    return FusionPlasma(
        cartesian_coords=cartesian_coords,
        cylindrical_coords=cylindrical_coords,
        theta=theta_grid,
        R_center=plasma_boundary.R_center,
        Z_center=plasma_boundary.Z_center,
        Boundary=plasma_boundary,
    )


def is_point_in_plasma(
    coords_test: CylindricalCoordinates | CartesianCoordinates,
    plasma: PlasmaBoundary | FusionPlasma,
) -> jnp.ndarray:
    """
    Determine whether a point (or array of points) lies inside the plasma volume.

    The algorithm exploits the toroidal symmetry of the Tokamak:
        1. Project 3D coordinates onto the 2D Poloidal plane (R, Z).
        2. Transform (R, Z) into local polar coordinates (r, theta) relative to the magnetic axis.
        3. Interpolate the boundary radius at angle theta.
        4. Compare test radius vs boundary radius.

    Complexity: O(log N) per point due to binary search interpolation.

    Args:
        coords_test: Spatial coordinates of test points.
                     Accepts Cartesian (X, Y, Z) or Cylindrical (R, φ, Z).
        plasma: The plasma definition.
                Accepts 2D PlasmaBoundary or 3D FusionPlasma.

    Returns
    -------
    is_inside : jnp.ndarray (bool)
        Boolean mask. True if the point lies strictly inside the boundary.
    """
    # 1. Resolve Plasma Source
    # If a full 3D plasma is passed, extract the 2D boundary definition
    boundary = plasma.Boundary if isinstance(plasma, FusionPlasma) else plasma

    # 2. Normalize to 2D Poloidal Coordinates (R, Z)
    # If Cartesian, project R = sqrt(X^2 + Y^2). If Cylindrical, use R directly.
    if isinstance(coords_test, CartesianCoordinates):
        R_test = jnp.sqrt(coords_test.X**2 + coords_test.Y**2)
        Z_test = coords_test.Z
    else:
        R_test = coords_test.R
        Z_test = coords_test.Z

    # 3. Transform to Local Polar Coordinates (relative to Magnetic Axis)
    # We shift the origin from the machine center (0,0) to the plasma center (R0, Z0)
    dR = R_test - boundary.R_center
    dZ = Z_test - boundary.Z_center

    r_test = jnp.sqrt(dR**2 + dZ**2)
    alpha_test = jnp.arctan2(dZ, dR)

    # 4. Interpolate Boundary Radius
    # We find the radius of the boundary at the exact angle of the test point.
    # period=2pi ensures correct wrapping for angles near -pi/pi.
    r_boundary = jnp.interp(
        alpha_test,
        boundary.alpha_geom,
        boundary.r_geom,
        period=2 * jnp.pi,
    )

    # 5. Check Containment
    return r_test < r_boundary
