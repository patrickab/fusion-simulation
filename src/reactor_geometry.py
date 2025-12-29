"""Module for parametric expression of a tokamak plasma surfaces"""

import numpy as np

from src.lib.config import Filepaths
from src.lib.geometry_config import (
    FusionPlasma,
    PlasmaBoundary,
    PlasmaConfig,
    RotationalAngles,
    ToroidalCoil2D,
    ToroidalCoil3D,
    ToroidalCoilConfig,
)
from src.lib.visualization import initialize_plotter, render_fusion_plasma, visualize_2d_geometry

COIL_RESOLUTION_3D = 64  # Number of points in the toroidal direction for 3D coils


def calculate_poloidal_boundary(plasma_config: PlasmaConfig) -> PlasmaBoundary:
    """
    Calculate the poloidal plasma boundary in R-Z coordinates.
    """

    # Define plasma boundary
    theta = RotationalAngles.THETA
    R_plasma = plasma_config.R0 + plasma_config.a * np.cos(theta + plasma_config.delta * np.sin(theta))
    Z_plasma = plasma_config.kappa * plasma_config.a * np.sin(theta)
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

    # Compute tangents
    dR = np.gradient(R, RotationalAngles.THETA)
    dZ = np.gradient(Z, RotationalAngles.THETA)

    # Compute unit normals (rotate tangent by 90 degrees)
    N_R = dZ
    N_Z = -dR
    norm = np.sqrt(N_R**2 + N_Z**2)
    N_R /= norm
    N_Z /= norm

    # Offset inner boundary
    inner_offset = toroid_coil_config.distance_from_plasma
    R_inner = R + inner_offset * N_R
    Z_inner = Z + inner_offset * N_Z

    center_offset = 0.5 * toroid_coil_config.radial_thickness
    R_center = R_inner + center_offset * N_R
    Z_center = Z_inner + center_offset * N_Z

    # Offset outer boundary
    coil_thickness = toroid_coil_config.radial_thickness
    R_outer = R_inner + coil_thickness * N_R
    Z_outer = Z_inner + coil_thickness * N_Z

    return ToroidalCoil2D(R_inner=R_inner, R_outer=R_outer, R_center=R_center, Z_inner=Z_inner, Z_center=Z_center, Z_outer=Z_outer)


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

    # Create 2D meshgrid for revolution
    R_grid, phi_grid = np.meshgrid(plasma_boundary.R_2d, RotationalAngles.PHI)
    Z_grid = np.tile(plasma_boundary.Z_2d, (RotationalAngles.n_phi, 1))

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
    phi_angles = np.linspace(0, 2 * np.pi, toroid_coil_config.n_field_coils, endpoint=False)
    coils = []

    # Prepare 2D cross-section
    r_inner_2d = toroidal_coil_2d.R_inner
    z_inner_2d = toroidal_coil_2d.Z_inner
    r_outer_2d = toroidal_coil_2d.R_outer
    z_outer_2d = toroidal_coil_2d.Z_outer

    phi_span = np.deg2rad(toroid_coil_config.angular_span)

    for phi_center in phi_angles:
        # Angular span for each coil
        phi_start = phi_center - phi_span / 2
        phi_end = phi_center + phi_span / 2

        # Toroidal sweep
        phi_sweep = np.linspace(phi_start, phi_end, COIL_RESOLUTION_3D)

        # Stack surfaces using broadcasting
        X_inner = np.outer(np.cos(phi_sweep), r_inner_2d)
        Y_inner = np.outer(np.sin(phi_sweep), r_inner_2d)
        Z_inner = np.tile(z_inner_2d, (COIL_RESOLUTION_3D, 1))

        X_outer = np.outer(np.cos(phi_sweep), r_outer_2d)
        Y_outer = np.outer(np.sin(phi_sweep), r_outer_2d)
        Z_outer = np.tile(z_outer_2d, (COIL_RESOLUTION_3D, 1))

        # Caps: first and last toroidal sweep
        X_cap_start = np.vstack([X_inner[0], X_outer[0]])
        Y_cap_start = np.vstack([Y_inner[0], Y_outer[0]])
        Z_cap_start = np.vstack([Z_inner[0], Z_outer[0]])

        X_cap_end = np.vstack([X_inner[-1], X_outer[-1]])
        Y_cap_end = np.vstack([Y_inner[-1], Y_outer[-1]])
        Z_cap_end = np.vstack([Z_inner[-1], Z_outer[-1]])

        # Center plane as X Y Z coordinates
        CentralPlane = np.column_stack((X_inner.mean(axis=0), Y_inner.mean(axis=0), Z_inner.mean(axis=0)))

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


if __name__ == "__main__":
    plasma_config = PlasmaConfig(
        R0=6.2,  # Major radius (m)
        a=3.2,  # Minor radius (m)
        kappa=1.7,  # Elongation factor
        delta=0.33,  # Triangularity factor
    )
    toroid_coil_config = ToroidalCoilConfig(
        distance_from_plasma=1.5,  # Distance from plasma surface (m)
        radial_thickness=0.8,  # Radial thickness of the coil (m)
        vertical_thickness=0.2,  # Vertical thickness of the coil (m)
        angular_span=6,  # Angular span of the coil (degrees)
        n_field_coils=8,  # Number of field coils
    )

    plasma_boundary, toroidal_coil_2d = calculate_2d_geometry(plasma_config=plasma_config, toroid_coil_config=toroid_coil_config)

    fusion_plasma = generate_fusion_plasma(plasma_boundary=plasma_boundary)
    toroidal_coils_3d = generate_toroidal_coils_3d(toroidal_coil_2d=toroidal_coil_2d, toroid_coil_config=toroid_coil_config)

    plotter = initialize_plotter(shape=(1, 2))

    plotter.subplot(0, 0)
    visualize_2d_geometry(plotter=plotter, plasma_boundary=plasma_boundary, toroidal_coil_2d=toroidal_coil_2d)
    plotter.subplot(0, 1)
    render_fusion_plasma(
        plotter=plotter,
        fusion_plasma=fusion_plasma,
        toroidal_coils=toroidal_coils_3d,
    )

    # Render the visualization
    plotter.show(title="Fusion Reactor Visualization", interactive=True)

    # Save geometry to PLY files
    fusion_plasma.to_ply_structuregrid(Filepaths.PLASMA_SURFACE)
    for i, coil in enumerate(toroidal_coils_3d, start=1):
        coil.to_ply(base_path=Filepaths.TOROIDAL_COIL_3D_DIR, coil_id=i)
