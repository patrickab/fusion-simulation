import numpy as np

from src.lib.geometry_config import (
    COIL_RESOLUTION_3D,
    PlasmaBoundary,
    RotationalAngles,
    ToroidalCoil2D,
    ToroidalCoil3D,
    ToroidalCoilConfig,
)


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
