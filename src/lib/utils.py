from pathlib import Path, PosixPath

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

    first = coils[0]

    # Case 1: already list[dict[str, PolyData]]
    if isinstance(first, dict):
        return coils

    # Case 2: list[ToroidalCoil3D]
    if isinstance(first, ToroidalCoil3D):
        out: list[dict[str, pv.PolyData]] = []
        for coil in coils:
            parts: dict[str, pv.PolyData] = {}

            inner = pv.StructuredGrid(coil.X_inner, coil.Y_inner, coil.Z_inner).extract_surface()
            parts["inner"] = inner

            outer = pv.StructuredGrid(coil.X_outer, coil.Y_outer, coil.Z_outer).extract_surface()
            parts["outer"] = outer

            if coil.X_cap_start.size > 0:
                cap_start = pv.StructuredGrid(coil.X_cap_start, coil.Y_cap_start, coil.Z_cap_start).extract_surface()
                parts["cap_start"] = cap_start

            if coil.X_cap_end.size > 0:
                cap_end = pv.StructuredGrid(coil.X_cap_end, coil.Y_cap_end, coil.Z_cap_end).extract_surface()
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
