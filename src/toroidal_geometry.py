import jax
import jax.numpy as jnp

from src.engine.plasma import calculate_poloidal_boundary
from src.lib.geometry_config import (
    COIL_RESOLUTION_3D,
    PlasmaBoundary,
    PlasmaConfig,
    ToroidalCoil2D,
    ToroidalCoil3D,
    ToroidalCoilConfig,
)


def calculate_toroidal_coil_boundary(
    theta: jnp.ndarray, plasma_config: PlasmaConfig, toroid_coil_config: ToroidalCoilConfig
) -> ToroidalCoil2D:
    """
    Compute toroidal coil 2D cross-section by offsetting plasma boundary along normal vectors.
    """
    # Base plasma boundary (contains R, Z and their derivatives)
    plasma_boundary: PlasmaBoundary = calculate_poloidal_boundary(
        theta=theta, plasma_config=plasma_config
    )
    R = plasma_boundary.R
    Z = plasma_boundary.Z
    grad_R = plasma_boundary.dR_dtheta
    grad_Z = plasma_boundary.dZ_dtheta

    # Normal vector (90° rotation of tangent) and normalization
    N_R_raw, N_Z_raw = grad_Z, -grad_R
    norm_mag = jnp.sqrt(N_R_raw**2 + N_Z_raw**2)
    N_R, N_Z = N_R_raw / norm_mag, N_Z_raw / norm_mag

    # Offset boundaries
    R_inner = R + toroid_coil_config.distance_from_plasma * N_R
    Z_inner = Z + toroid_coil_config.distance_from_plasma * N_Z
    R_outer = R_inner + toroid_coil_config.radial_thickness * N_R
    Z_outer = Z_inner + toroid_coil_config.radial_thickness * N_Z

    return ToroidalCoil2D(
        R_inner=R_inner,
        R_outer=R_outer,
        R_center=(R_inner + R_outer) / 2,
        Z_inner=Z_inner,
        Z_outer=Z_outer,
        Z_center=(Z_inner + Z_outer) / 2,
    )


@jax.jit
def _generate_single_coil_jit(
    phi_center: jnp.ndarray, phi_sweep: jnp.ndarray, toroidal_coil_2d: ToroidalCoil2D
) -> ToroidalCoil3D:
    """JIT-compiled kernel for a single coil geometry calculation."""
    # Shift phi sweep to coil center
    phi_local = phi_sweep + phi_center
    cos_phi = jnp.cos(phi_local)
    sin_phi = jnp.sin(phi_local)

    # 2D cross-section in the poloidal plane
    r_inner_2d = toroidal_coil_2d.R_inner
    z_inner_2d = toroidal_coil_2d.Z_inner
    r_outer_2d = toroidal_coil_2d.R_outer
    z_outer_2d = toroidal_coil_2d.Z_outer

    # Inner surface
    X_inner = jnp.outer(cos_phi, r_inner_2d)
    Y_inner = jnp.outer(sin_phi, r_inner_2d)
    Z_inner = jnp.tile(z_inner_2d, (COIL_RESOLUTION_3D, 1))

    # Outer surface
    X_outer = jnp.outer(cos_phi, r_outer_2d)
    Y_outer = jnp.outer(sin_phi, r_outer_2d)
    Z_outer = jnp.tile(z_outer_2d, (COIL_RESOLUTION_3D, 1))

    # End caps
    X_cap_start = jnp.vstack([X_inner[0], X_outer[0]])
    Y_cap_start = jnp.vstack([Y_inner[0], Y_outer[0]])
    Z_cap_start = jnp.vstack([Z_inner[0], Z_outer[0]])

    X_cap_end = jnp.vstack([X_inner[-1], X_outer[-1]])
    Y_cap_end = jnp.vstack([Y_inner[-1], Y_outer[-1]])
    Z_cap_end = jnp.vstack([Z_inner[-1], Z_outer[-1]])

    # Central plane (average over toroidal direction)
    central_plane_x = X_inner.mean(axis=0)
    central_plane_y = Y_inner.mean(axis=0)
    central_plane_z = Z_inner.mean(axis=0)
    CentralPlane = jnp.column_stack((central_plane_x, central_plane_y, central_plane_z))

    return ToroidalCoil3D(
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


def generate_toroidal_coils_3d(
    toroidal_coil_2d: ToroidalCoil2D,
    toroid_coil_config: ToroidalCoilConfig,
) -> ToroidalCoil3D:
    """
    Generate full 3D geometry for toroidal coils using partial JIT to handle dynamic coil counts.
    """
    # Orchestration in Python to handle dynamic shapes
    n_coils = int(toroid_coil_config.n_field_coils)
    angular_span_rad = jnp.deg2rad(toroid_coil_config.angular_span)

    phi_centers = jnp.linspace(0, 2 * jnp.pi, n_coils, endpoint=False)
    phi_sweep = jnp.linspace(-angular_span_rad / 2, angular_span_rad / 2, COIL_RESOLUTION_3D)

    # Vectorized over coils using vmap; phi_centers is the batched dimension
    return jax.vmap(_generate_single_coil_jit, in_axes=(0, None, None))(
        phi_centers, phi_sweep, toroidal_coil_2d
    )
