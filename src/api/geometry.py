"""Reactor geometry: plasma boundary + coil meshes, in 2D and 3D."""

import numpy as np

from src.engine.plasma import calculate_fusion_plasma
from src.engine.stellarator import calculate_stellarator_surface, calculate_toroidal_slice
from src.lib.geometry_config import PlasmaGeometry, ToroidalCoilConfig
from src.lib.visualization import calculate_2d_geometry
from src.toroidal_geometry import generate_toroidal_coils_3d


def _stride2d(arr: np.ndarray, stride: int) -> np.ndarray:
    return np.asarray(arr)[::stride, ::stride]


def _to_list(arr: object) -> list:
    return np.asarray(arr, dtype=np.float64).tolist()


def build_geometry_response(
    R0: float,
    a: float,
    kappa: float,
    delta: float,
    show_coils: bool,
    coil_cfg: ToroidalCoilConfig,
    mesh_stride: int,
) -> dict:
    plasma_geometry = PlasmaGeometry(R0=R0, a=a, kappa=kappa, delta=delta)

    plasma_boundary, toroidal_coil_2d = calculate_2d_geometry(
        plasma_geometry=plasma_geometry, toroid_coil_config=coil_cfg
    )

    response: dict = {
        "boundary2d": {
            "R": _to_list(plasma_boundary.R),
            "Z": _to_list(plasma_boundary.Z),
        },
        "coil2d": None,
        "plasma3d": None,
        "coils3d": None,
    }

    if show_coils:
        response["coil2d"] = {
            "R_inner": _to_list(toroidal_coil_2d.R_inner),
            "Z_inner": _to_list(toroidal_coil_2d.Z_inner),
            "R_outer": _to_list(toroidal_coil_2d.R_outer),
            "Z_outer": _to_list(toroidal_coil_2d.Z_outer),
            "R_center": _to_list(toroidal_coil_2d.R_center),
            "Z_center": _to_list(toroidal_coil_2d.Z_center),
        }

    fusion_plasma = calculate_fusion_plasma(plasma_boundary=plasma_boundary)
    X = _stride2d(fusion_plasma.X, mesh_stride)
    Y = _stride2d(fusion_plasma.Y, mesh_stride)
    Z = _stride2d(fusion_plasma.Z, mesh_stride)
    response["plasma3d"] = {
        "n_phi": X.shape[0],
        "n_theta": X.shape[1],
        "X": _to_list(X.flatten()),
        "Y": _to_list(Y.flatten()),
        "Z": _to_list(Z.flatten()),
    }

    if show_coils:
        toroidal_coils_3d = generate_toroidal_coils_3d(
            toroidal_coil_2d=toroidal_coil_2d, toroid_coil_config=coil_cfg
        )
        coils_out = []
        n_coils = len(toroidal_coils_3d)
        for i in range(n_coils):
            coil = toroidal_coils_3d[i]
            X_in = _stride2d(coil.X_inner, mesh_stride)
            Y_in = _stride2d(coil.Y_inner, mesh_stride)
            Z_in = _stride2d(coil.Z_inner, mesh_stride)
            X_out = _stride2d(coil.X_outer, mesh_stride)
            Y_out = _stride2d(coil.Y_outer, mesh_stride)
            Z_out = _stride2d(coil.Z_outer, mesh_stride)
            coils_out.append(
                {
                    "n_phi": X_in.shape[0],
                    "n_theta": X_in.shape[1],
                    "X_inner": _to_list(X_in.flatten()),
                    "Y_inner": _to_list(Y_in.flatten()),
                    "Z_inner": _to_list(Z_in.flatten()),
                    "X_outer": _to_list(X_out.flatten()),
                    "Y_outer": _to_list(Y_out.flatten()),
                    "Z_outer": _to_list(Z_out.flatten()),
                }
            )
        response["coils3d"] = coils_out

    return response


def build_stellarator_geometry_response(
    R0: float,
    a: float,
    kappa: float,
    n_field_periods: int,
    helical_amplitude: float,
    mesh_stride: int,
) -> dict:
    """Build an idealized Fourier stellarator surface for the geometry explorer."""
    boundary_R, boundary_Z = calculate_toroidal_slice(
        R0, a, kappa, n_field_periods, helical_amplitude
    )
    surface = calculate_stellarator_surface(R0, a, kappa, n_field_periods, helical_amplitude)
    X = _stride2d(surface.X, mesh_stride)
    Y = _stride2d(surface.Y, mesh_stride)
    Z = _stride2d(surface.Z, mesh_stride)
    return {
        "boundary2d": {"R": _to_list(boundary_R), "Z": _to_list(boundary_Z)},
        "plasma3d": {
            "n_phi": X.shape[0],
            "n_theta": X.shape[1],
            "X": _to_list(X.flatten()),
            "Y": _to_list(Y.flatten()),
            "Z": _to_list(Z.flatten()),
        },
    }
