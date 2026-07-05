"""Reactor geometry: plasma boundary + coil meshes, in 2D and 3D."""

import numpy as np

from src.engine.plasma import calculate_fusion_plasma
from src.lib.geometry_config import PlasmaGeometry, ToroidalCoilConfig
from src.lib.visualization import calculate_2d_geometry
from src.toroidal_geometry import generate_toroidal_coils_3d


def _stride2d(arr: np.ndarray, stride: int) -> np.ndarray:
    return np.asarray(arr)[::stride, ::stride]


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
            "R": np.asarray(plasma_boundary.R).tolist(),
            "Z": np.asarray(plasma_boundary.Z).tolist(),
        },
        "coil2d": None,
        "plasma3d": None,
        "coils3d": None,
    }

    if show_coils:
        response["coil2d"] = {
            "R_inner": np.asarray(toroidal_coil_2d.R_inner).tolist(),
            "Z_inner": np.asarray(toroidal_coil_2d.Z_inner).tolist(),
            "R_outer": np.asarray(toroidal_coil_2d.R_outer).tolist(),
            "Z_outer": np.asarray(toroidal_coil_2d.Z_outer).tolist(),
            "R_center": np.asarray(toroidal_coil_2d.R_center).tolist(),
            "Z_center": np.asarray(toroidal_coil_2d.Z_center).tolist(),
        }

    fusion_plasma = calculate_fusion_plasma(plasma_boundary=plasma_boundary)
    X = _stride2d(fusion_plasma.X, mesh_stride)
    Y = _stride2d(fusion_plasma.Y, mesh_stride)
    Z = _stride2d(fusion_plasma.Z, mesh_stride)
    response["plasma3d"] = {
        "n_phi": X.shape[0],
        "n_theta": X.shape[1],
        "X": X.flatten().tolist(),
        "Y": Y.flatten().tolist(),
        "Z": Z.flatten().tolist(),
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
                    "X_inner": X_in.flatten().tolist(),
                    "Y_inner": Y_in.flatten().tolist(),
                    "Z_inner": Z_in.flatten().tolist(),
                    "X_outer": X_out.flatten().tolist(),
                    "Y_outer": Y_out.flatten().tolist(),
                    "Z_outer": Z_out.flatten().tolist(),
                }
            )
        response["coils3d"] = coils_out

    return response
