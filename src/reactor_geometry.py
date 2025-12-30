"""Module for parametric expression of a tokamak plasma surfaces"""

import numpy as np

from src.lib.geometry_config import (
    FusionPlasma,
    PlasmaBoundary,
    PlasmaConfig,
    RotationalAngles,
    ToroidalCoil2D,
    ToroidalCoil3D,
    ToroidalCoilConfig,
)

COIL_RESOLUTION_3D = 64  # Number of points in the toroidal direction for 3D coils


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
    shaped_theta = theta + triangularity * np.sin(theta)

    # Standard tokamak cross-section: shifted circle with shaping
    R_plasma = major_radius + minor_radius * np.cos(shaped_theta)
    Z_plasma = elongation * minor_radius * np.sin(theta)

    return PlasmaBoundary(R_2d=R_plasma, Z_2d=Z_plasma)


def calculate_toroidal_coil_boundary(plasma_boundary: PlasmaBoundary, toroid_coil_config: ToroidalCoilConfig) -> ToroidalCoil2D:
    """
    Compute toroidal coil 2D cross-section by offsetting plasma boundary along normal vectors.
    The inner boundary is offset by `distance_from_plasma`, and the outer boundary is defined
    by adding the coil thickness along the poloidal normals.
    """

    # Base plasma boundary
    R = plasma_boundary.R_2d
    Z = plasma_boundary.Z_2d
    dR_dtheta = np.gradient(R, RotationalAngles.THETA)
    dZ_dtheta = np.gradient(Z, RotationalAngles.THETA)

    # Outward normal is a 90° rotation of the tangent in the R-Z plane
    N_R = dZ_dtheta
    N_Z = -dR_dtheta
    normal_magnitude = np.sqrt(N_R**2 + N_Z**2)

    # Avoid division by zero
    N_R /= normal_magnitude
    N_Z /= normal_magnitude

    # Inner coil boundary: offset from plasma along outward normal
    inner_offset = toroid_coil_config.distance_from_plasma
    R_inner = R + inner_offset * N_R
    Z_inner = Z + inner_offset * N_Z

    # Radial thickness of the coil (inner → outer)
    coil_thickness = toroid_coil_config.radial_thickness

    # Coil centerline: halfway between inner and outer surfaces
    center_offset = 0.5 * coil_thickness
    R_center = R_inner + center_offset * N_R
    Z_center = Z_inner + center_offset * N_Z

    # Outer coil boundary: full thickness from inner surface
    R_outer = R_inner + coil_thickness * N_R
    Z_outer = Z_inner + coil_thickness * N_Z

    return ToroidalCoil2D(
        R_inner=R_inner,
        R_outer=R_outer,
        R_center=R_center,
        Z_inner=Z_inner,
        Z_center=Z_center,
        Z_outer=Z_outer,
    )


def calculate_2d_geometry(
    plasma_config: PlasmaConfig, toroid_coil_config: ToroidalCoilConfig
) -> tuple[PlasmaBoundary, ToroidalCoil2D]:
    """
    Returns R and Z coordinates for a 2D poloidal plasma boundary shape.
    Cross-section of a tokamak plasma in the poloidal plane.
    """
    plasma_boundary = calculate_poloidal_boundary(plasma_config)
    toroidal_coil_2d = calculate_toroidal_coil_boundary(plasma_boundary, toroid_coil_config)
    return plasma_boundary, toroidal_coil_2d


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
    R_grid, phi_grid = np.meshgrid(R_poloidal, phi)

    # Repeat Z along the toroidal direction to match R_grid/phi_grid shape
    Z_grid = np.tile(Z_poloidal, (RotationalAngles.n_phi, 1))

    # Convert cylindrical (R, φ, Z) → Cartesian (X, Y, Z)
    X = R_grid * np.cos(phi_grid)
    Y = R_grid * np.sin(phi_grid)
    Z = Z_grid

    return FusionPlasma(X=X, Y=Y, Z=Z, Boundary=plasma_boundary)


def generate_toroidal_coils_3d(
    toroidal_coil_2d: ToroidalCoil2D,
    toroid_coil_config: ToroidalCoilConfig,
) -> list[ToroidalCoil3D]:
    """
    Generate full 3D geometry for toroidal coils from 2D cross-section using efficient numpy operations.
    """

    # Central toroidal positions of each coil (evenly spaced around 2π)
    phi_centers = np.linspace(0, 2 * np.pi, toroid_coil_config.n_field_coils, endpoint=False)
    coils = []

    # 2D cross-section in the poloidal plane
    r_inner_2d = toroidal_coil_2d.R_inner
    z_inner_2d = toroidal_coil_2d.Z_inner
    r_outer_2d = toroidal_coil_2d.R_outer
    z_outer_2d = toroidal_coil_2d.Z_outer

    # Total toroidal angular extent of a single coil
    phi_span = np.deg2rad(toroid_coil_config.angular_span)

    for phi_center in phi_centers:
        # Angular limits for this coil around its central toroidal angle
        phi_start = phi_center - phi_span / 2
        phi_end = phi_center + phi_span / 2

        # Discretized toroidal sweep for this coil
        phi_sweep = np.linspace(phi_start, phi_end, COIL_RESOLUTION_3D)

        # Inner surface: revolve inner 2D contour along phi_sweep
        cos_phi = np.cos(phi_sweep)
        sin_phi = np.sin(phi_sweep)

        X_inner = np.outer(cos_phi, r_inner_2d)
        Y_inner = np.outer(sin_phi, r_inner_2d)
        Z_inner = np.tile(z_inner_2d, (COIL_RESOLUTION_3D, 1))

        # Outer surface: revolve outer 2D contour along phi_sweep
        X_outer = np.outer(cos_phi, r_outer_2d)
        Y_outer = np.outer(sin_phi, r_outer_2d)
        Z_outer = np.tile(z_outer_2d, (COIL_RESOLUTION_3D, 1))

        # End caps: connect inner and outer surfaces at the start and end of the toroidal sweep
        X_cap_start = np.vstack([X_inner[0], X_outer[0]])
        Y_cap_start = np.vstack([Y_inner[0], Y_outer[0]])
        Z_cap_start = np.vstack([Z_inner[0], Z_outer[0]])

        X_cap_end = np.vstack([X_inner[-1], X_outer[-1]])
        Y_cap_end = np.vstack([Y_inner[-1], Y_outer[-1]])
        Z_cap_end = np.vstack([Z_inner[-1], Z_outer[-1]])

        # Central poloidal plane of the coil: average over toroidal direction
        central_plane_x = X_inner.mean(axis=0)
        central_plane_y = Y_inner.mean(axis=0)
        central_plane_z = Z_inner.mean(axis=0)
        CentralPlane = np.column_stack((central_plane_x, central_plane_y, central_plane_z))

        coil = ToroidalCoil3D(
            X_inner=X_inner,
            Y_inner=Y_inner,
            Z_inner=Z_inner,
            X_outer=X_outer,
            Y_outer=Y_outer,
            Z_outer=Z_outer,
            X_cap_start=X_cap_start,
            Y_cap_start=Y_cap_start,
            Z_cap_start=Z_cap_start,
            X_cap_end=X_cap_end,
            Y_cap_end=Y_cap_end,
            Z_cap_end=Z_cap_end,
            CentralPlane=CentralPlane,
            ToroidalCoil2D=toroidal_coil_2d,
        )
        coils.append(coil)

    return coils
