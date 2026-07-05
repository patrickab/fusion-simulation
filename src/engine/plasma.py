"""Module for parametric expression of a tokamak plasma surfaces"""

import jax
import jax.numpy as jnp

from src.lib.geometry_config import (
    CartesianCoordinates,
    CylindricalCoordinates,
    FusionPlasma,
    PlasmaBoundary,
    PlasmaGeometry,
    RotationalAngles,
)


def get_poloidal_points(
    theta: float, plasma_geometry: PlasmaGeometry, scaling_factor: float = 1.0
) -> tuple[float, float]:
    """
    Calculates a single (R, Z) point for a given theta.
    Intentionally not vectorized to perform Nx2 instead of NxN jacobian computations.
    """
    # unpack parameters for readability
    major_radius = plasma_geometry.R0
    minor_radius = plasma_geometry.a
    triangularity = plasma_geometry.delta
    elongation = plasma_geometry.kappa

    shaped_theta = theta + triangularity * jnp.sin(theta)

    # calculate coordinates
    R = major_radius + scaling_factor * minor_radius * jnp.cos(shaped_theta)
    Z = elongation * scaling_factor * minor_radius * jnp.sin(theta)
    return R, Z


def calculate_poloidal_boundary(
    theta: jnp.ndarray, plasma_geometry: PlasmaGeometry, phi: float = 0.0
) -> PlasmaBoundary:
    """Compute R-Z coordinates for a shaped tokamak plasma boundary.

    Args:
        theta: poloidal angles
        plasma_geometry: geometric parameters (R0, a, kappa, delta)
        phi: toroidal angle (rad)

    Returns:
        2D boundary coordinates
    """
    # Use jax.jvp to compute values and derivatives simultaneously.
    # Since get_poloidal_points is element-wise, passing a tangent of ones
    # correctly computes the element-wise gradients for both scalars and arrays.
    tangent = jnp.ones_like(theta)
    (R, Z), (dR_dtheta, dZ_dtheta) = jax.jvp(
        lambda t: get_poloidal_points(t, plasma_geometry, 1.0), (theta,), (tangent,)
    )

    # Create cylindrical coordinates with constant phi
    phi_array = jnp.full_like(R, phi)
    coords = CylindricalCoordinates(R=R, phi=phi_array, Z=Z)

    return PlasmaBoundary(
        coords=coords,
        theta=theta,
        dR_dtheta=dR_dtheta,
        dZ_dtheta=dZ_dtheta,
        R_center=plasma_geometry.R0,
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


def boundary_normalized_radius(
    R: jnp.ndarray,
    Z: jnp.ndarray,
    boundary: PlasmaBoundary,
) -> jnp.ndarray:
    """Radius of (R, Z) relative to the boundary curve; exactly 1.0 on the boundary.

    Interpolates the boundary's precomputed (R, Z) polyline by poloidal angle
    (same technique as is_point_in_plasma). Used both for inside/outside tests
    and as the hard-boundary-condition envelope in network.denormalize_psi.

    Complexity: O(log N) per point due to binary search interpolation.
    """
    dR = R - boundary.R_center
    dZ = Z - boundary.Z_center
    # epsilon avoids the sqrt gradient singularity exactly at the magnetic axis
    r_test = jnp.sqrt(dR**2 + dZ**2 + 1e-12)
    alpha_test = jnp.arctan2(dZ, dR)

    dR_boundary = boundary.R - boundary.R_center
    dZ_boundary = boundary.Z - boundary.Z_center
    r_geom = jnp.sqrt(dR_boundary**2 + dZ_boundary**2)
    alpha_geom = jnp.arctan2(dZ_boundary, dR_boundary)

    # Sort by angle to ensure valid interpolation input
    sort_indices = jnp.argsort(alpha_geom)
    alpha_geom = alpha_geom[sort_indices]
    r_geom = r_geom[sort_indices]

    # period=2pi ensures correct wrapping for angles near -pi/pi.
    r_boundary = jnp.interp(alpha_test, alpha_geom, r_geom, period=2 * jnp.pi)
    return r_test / r_boundary


def is_point_in_plasma(
    coords_test: CylindricalCoordinates | CartesianCoordinates,
    plasma: PlasmaBoundary | FusionPlasma,
) -> jnp.ndarray:
    """
    Determine whether a point (or array of points) lies inside the plasma volume.

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
    boundary = plasma.Boundary if isinstance(plasma, FusionPlasma) else plasma

    if isinstance(coords_test, CartesianCoordinates):
        R_test = jnp.sqrt(coords_test.X**2 + coords_test.Y**2)
        Z_test = coords_test.Z
    else:
        R_test = coords_test.R
        Z_test = coords_test.Z

    return boundary_normalized_radius(R_test, Z_test, boundary) < 1.0
