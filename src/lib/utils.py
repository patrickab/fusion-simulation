from pathlib import Path, PosixPath

import jax
import numpy as np
import pyvista as pv

from src.lib.geometry_config import FusionPlasma, ToroidalCoil3D


def _coils_to_polydata(
    coils: list[ToroidalCoil3D] | list[pv.PolyData] | list[Path | PosixPath] | list[dict[str, pv.PolyData]] | None,
) -> list[dict[str, pv.PolyData]]:
    """
    Normalize various coil representations into a list of dicts of PolyData meshes.

    Accepted inputs:
      - None: return empty list
      - list[dict[str, PolyData]]: returned as-is
      - list[ToroidalCoil3D]: convert each to inner/outer/caps PolyData
      - list[pv.PolyData]: each entry is treated as a single-part coil {"coil": mesh}
      - list[Path]: each path is read as PolyData and treated as {"coil": mesh}
    """
    if coils is None or not coils:
        return []

    first = jax.tree_util.tree_map(lambda x: x[0], coils)

    # Case 1: already list[dict[str, PolyData]]
    if isinstance(first, dict):
        return coils

    # Case 2: list[ToroidalCoil3D]
    if isinstance(first, ToroidalCoil3D):
        out: list[dict[str, pv.PolyData]] = []
        for coil in coils:
            parts: dict[str, pv.PolyData] = {}

            x_inner, y_inner, z_inner = np.array(coil.X_inner), np.array(coil.Y_inner), np.array(coil.Z_inner)
            inner = pv.StructuredGrid(x_inner, y_inner, z_inner).extract_surface()
            parts["inner"] = inner

            x_outer, y_outer, z_outer = np.array(coil.X_outer), np.array(coil.Y_outer), np.array(coil.Z_outer)
            outer = pv.StructuredGrid(x_outer, y_outer, z_outer).extract_surface()
            parts["outer"] = outer

            if coil.X_cap_start.size > 0:
                x_cap_start, y_cap_start, z_cap_start = (
                    np.array(coil.X_cap_start),
                    np.array(coil.Y_cap_start),
                    np.array(coil.Z_cap_start),
                )
                cap_start = pv.StructuredGrid(x_cap_start, y_cap_start, z_cap_start).extract_surface()
                parts["cap_start"] = cap_start

            if coil.X_cap_end.size > 0:
                x_cap_end, y_cap_end, z_cap_end = np.array(coil.X_cap_end), np.array(coil.Y_cap_end), np.array(coil.Z_cap_end)
                cap_end = pv.StructuredGrid(x_cap_end, y_cap_end, z_cap_end).extract_surface()
                parts["cap_end"] = cap_end

            out.append(parts)
        return out

    # Case 3: list[pv.PolyData] -> wrap each as a single-part coil
    if isinstance(first, pv.PolyData):
        out_pd: list[dict[str, pv.PolyData]] = []
        for mesh in coils:
            out_pd.append({"coil": mesh})
        return out_pd

    # Case 4: list[Path] -> read each as PolyData and wrap
    if isinstance(first, Path | PosixPath):
        out_path: list[dict[str, pv.PolyData]] = []
        for path in coils:
            mesh = pv.read(path)
            out_path.append({"coil": mesh})
        return out_path

    raise TypeError("coils must be a list of ToroidalCoil3D, pv.PolyData, Path, or dict[str, pv.PolyData]")


def _plasma_to_polydata(plasma: FusionPlasma | pv.PolyData | Path) -> pv.PolyData:
    """Convert FusionPlasma, PolyData, or a file path to a PolyData surface mesh."""
    if isinstance(plasma, Path):
        try:
            return pv.read(plasma)
        except FileNotFoundError:
            raise FileNotFoundError(f"PLY file not found at path: {plasma}") from None

    if isinstance(plasma, FusionPlasma):
        return pv.StructuredGrid(np.array(plasma.X), np.array(plasma.Y), np.array(plasma.Z)).extract_surface()

    if isinstance(plasma, pv.PolyData):
        return plasma

    raise TypeError("plasma must be FusionPlasma, pv.PolyData, or Path to a PLY file.")
