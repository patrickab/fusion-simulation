"""Utility functions for visualizing fusion simulation geometry."""

import os
from typing import Optional

import numpy as np
import pyvista as pv

from src.lib.config import Filepaths
from src.lib.geometry_config import FusionPlasma, PlasmaBoundary, RotationalAngles, ToroidalCoil2D, ToroidalCoil3D
from src.lib.linalg_utils import convert_rz_to_xyz


def _to_polydata_plasma(plasma: FusionPlasma | pv.PolyData) -> pv.PolyData:
    """Convert FusionPlasma or PolyData to a PolyData surface mesh."""
    if isinstance(plasma, FusionPlasma):
        return pv.StructuredGrid(plasma.X, plasma.Y, plasma.Z).extract_surface()
    if isinstance(plasma, pv.PolyData):
        return plasma
    raise TypeError("plasma must be FusionPlasma or pv.PolyData")


def _to_polydata_coils(coils: list[ToroidalCoil3D] | list[dict[str, pv.PolyData]]) -> list[dict[str, pv.PolyData]]:
    """
    Normalize coil representation to a list of dicts of PolyData meshes.

    - If input is list[dict[str, PolyData]]: return as-is.
    - If input is list[ToroidalCoil3D]: build PolyData surfaces from their grids.
    """
    if not coils:
        return []

    # Case 1: already list[dict[str, PolyData]]
    if isinstance(coils[0], dict):
        return coils  # type: ignore[return-value]

    # Case 2: list[ToroidalCoil3D]
    out: list[dict[str, pv.PolyData]] = []
    for coil in coils:  # type: ignore[assignment]
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


def plot_plasma(
    plotter: pv.Plotter,
    plasma: FusionPlasma | pv.PolyData,
    show_wireframe: bool = False,
    opacity: float = 0.4,
    cmap: str = "plasma",
    name_prefix: str = "plasma",
) -> None:
    """Add a plasma surface (FusionPlasma or PolyData) to the plotter."""
    mesh = _to_polydata_plasma(plasma)
    mesh.compute_normals(inplace=True)

    plotter.add_mesh(
        mesh,
        color=None,
        scalars=mesh.points[:, 2],
        cmap=cmap,
        smooth_shading=True,
        opacity=opacity,
        show_edges=False,
        lighting=True,
        specular=0.4,
        specular_power=15,
        name=f"{name_prefix}_surface",
    )

    if show_wireframe:
        sparse_wireframe = get_sparse_wireframe(fusion_plasma=mesh)
        plotter.add_mesh(
            sparse_wireframe,
            style="wireframe",
            color="cyan",
            line_width=1,
            opacity=0.2,
            name=f"{name_prefix}_wireframe",
        )


def plot_coils(
    plotter: pv.Plotter,
    coils: list[ToroidalCoil3D] | list[dict[str, pv.PolyData]],
    name_prefix: str = "Toroidal Coil",
) -> None:
    """Add toroidal coils (3D objects or loaded meshes) to the plotter."""
    coil_mesh_sets = _to_polydata_coils(coils)

    for i, coil_parts in enumerate(coil_mesh_sets, start=1):
        base_name = f"{name_prefix} {i}"
        for part_name, mesh in coil_parts.items():
            pretty_part = part_name.replace("_", " ").title()
            plotter.add_mesh(
                mesh,
                color=(0.8, 0.8, 0.85),
                opacity=1.0,
                name=f"{base_name} {pretty_part}",
                specular=0.8,
                specular_power=128,
            )


def export_plasmasurface(
    fusion_plasma: FusionPlasma,
    filename: str = Filepaths.PLASMA_SURFACE,
) -> None:
    """Converts the toroidal plasma surface to a polygonal mesh & stores it as .ply"""

    grid = pv.StructuredGrid(fusion_plasma.X, fusion_plasma.Y, fusion_plasma.Z).extract_surface()

    grid.save(filename)
    print(f"✅ Exported plasma surface to: {os.path.abspath(filename)}")


def display_cylindrical_angles(
    plotter: pv.Plotter,
    n_angles: int = 8,
    radius: float = 10.0,
    color: str = "grey",
) -> None:
    """
    Adds cylindrical coordinate system lines for theta angles to the plotter.
    """
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    for angle in angles:
        x = np.array([0, radius * np.cos(angle)])
        y = np.array([0, radius * np.sin(angle)])
        z = np.array([0, 0])
        plotter.add_lines(
            np.column_stack((x, y, z)),
            color=color,
            width=1,
            label=f"Theta {np.degrees(angle):.1f}°",
        )


def display_phi_coordinates(
    plotter: pv.Plotter,
    n_angles: int = 16,
) -> None:
    """
    Adds cylindrical coordinate system lines for phi angles to the plotter.
    """
    # Add cylindrical coordinate system (phi)
    n_angles = 9
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    for angle in angles:
        x = np.array([0, 0])
        y = np.array([0, 10 * np.cos(angle)])
        z = np.array([0, 10 * np.sin(angle)])
        plotter.add_lines(
            np.column_stack((x, y, z)),
            color="lightgray",
            width=1,
            label=f"Phi {np.degrees(angle):.1f}°",
        )


def set_camera_relative_to_body(plotter: pv.Plotter, distance_factor: float = 2.0, view_up: tuple[int, int, int] = (0, 0, 1)) -> None:
    """
    Set camera position relative to the bounding box of all actors.

    Parameters:
        plotter: The PyVista plotter instance
        distance_factor: Multiplier for how far the camera is from the center (default 2x size)
        view_up: Tuple indicating the up direction in camera view (default Z-up)
    """
    bounds = plotter.bounds
    center = [(bounds[1] + bounds[0]) / 2, (bounds[3] + bounds[2]) / 2, (bounds[5] + bounds[4]) / 2]
    size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    # Position the camera along +X axis, at distance_factor * size from center
    position = (center[0] + distance_factor * size, center[1], center[2])

    plotter.camera_position = [position, center, view_up]


def visualize_2d_geometry(plotter: pv.Plotter, plasma_boundary: PlasmaBoundary, toroidal_coil_2d: ToroidalCoil2D) -> None:
    """Display 2D boundaries of plasma and toroidal coil in the plotter."""
    plotter.add_title("2D Poloidal Cross-Section", font_size=16, color="white")

    # Rotate plasma boundary and toroidal coil 2d by 90 degrees around the x-axis
    radians_90 = 90 * np.pi / 180
    coordinates_xyz_plasma_boundary = convert_rz_to_xyz(R=plasma_boundary.R_2d, Z=plasma_boundary.Z_2d, phi=radians_90)
    coordinates_xyz_inner_coil = convert_rz_to_xyz(R=toroidal_coil_2d.R_inner, Z=toroidal_coil_2d.Z_inner, phi=radians_90)
    coordinates_xyz_outer_coil = convert_rz_to_xyz(R=toroidal_coil_2d.R_outer, Z=toroidal_coil_2d.Z_outer, phi=radians_90)
    coordinates_xyz_coil_center = convert_rz_to_xyz(R=toroidal_coil_2d.R_center, Z=toroidal_coil_2d.Z_center, phi=radians_90)

    # Add plasma boundary (2D poloidal cross-section)
    plotter.add_lines(
        np.column_stack(coordinates_xyz_plasma_boundary),
        color="cyan",
        width=2,
    )

    # Add toroidal coil 2D boundaries (inner and outer)
    plotter.add_lines(
        np.column_stack(coordinates_xyz_inner_coil),
        color="purple",
        width=2,
        label="Toroidal Coil Inner Boundary",
    )
    plotter.add_lines(
        np.column_stack(coordinates_xyz_outer_coil),
        color="purple",
        width=2,
        label="Toroidal Coil Outer Boundary",
    )

    plotter.add_lines(
        np.column_stack(coordinates_xyz_coil_center),
        color="purple",
        width=2,
        label="Toroidal Coil Center Boundary",
    )

    plotter.camera_position = "iso"


def get_sparse_wireframe(fusion_plasma: FusionPlasma | pv.PolyData, N: int = 8) -> Optional[pv.PolyData]:
    """Generate a sparse wireframe mesh from FusionPlasma or PolyData."""

    if isinstance(fusion_plasma, FusionPlasma):
        x = fusion_plasma.X[::N, ::N]
        y = fusion_plasma.Y[::N, ::N]
        z = fusion_plasma.Z[::N, ::N]
    elif isinstance(fusion_plasma, pv.PolyData):
        points = fusion_plasma.points.reshape((RotationalAngles.n_phi, RotationalAngles.n_theta, 3))
        sparse_points = points[::N, ::N, :]
        x = sparse_points[:, :, 0]
        y = sparse_points[:, :, 1]
        z = sparse_points[:, :, 2]
    else:
        raise TypeError("Input data must be of type FusionPlasma or pv.PolyData.")

    return pv.StructuredGrid(x, y, z).extract_surface()


def visualize_3d_geometry(
    plotter: pv.Plotter,
    fusion_plasma: FusionPlasma,
    toroidal_coils_3d: list[ToroidalCoil3D],
) -> None:
    """Visualize the 3D geometry of the fusion plasma and toroidal coils."""
    plot_plasma(
        plotter=plotter,
        plasma=fusion_plasma,
        show_wireframe=True,
        opacity=0.8,
        cmap="plasma",
        name_prefix="plasma",
    )

    plot_coils(
        plotter=plotter,
        coils=toroidal_coils_3d,
        name_prefix="Toroidal Coil",
    )

    plotter.add_title("3D Toroidal Geometry", font_size=16, color="white")


def initialize_plotter(shape: tuple[int, int] = (1, 1)) -> pv.Plotter:
    """Initialize a PyVista plotter with a black background and trackball style."""

    plotter = pv.Plotter(shape=shape, border=True, border_color="white")
    plotter.set_background("black")
    return plotter


def render_fusion_plasma(
    plasma_file_path: str = Filepaths.PLASMA_SURFACE,
    show_cylindrical_angles: bool = False,
    show_wireframe: bool = False,
    show_coils: bool = False,
    coil_dir: str | os.PathLike = Filepaths.TOROIDAL_COIL_3D_DIR,
) -> None:
    """Load and render a fusion plasma surface from a ``.ply`` file.

    Parameters
    ----------
    ply_file_path:
        Path to the PLY file containing the plasma mesh.
    show_cylindrical_angles:
        If ``True`` display angle guides for cylindrical coordinates.
    """

    # Read plasma mesh from disk
    mesh_fusion_plasma = pv.read(plasma_file_path)

    # Create a plotter
    plotter = initialize_plotter()

    # Plot plasma
    plot_plasma(
        plotter=plotter,
        plasma=mesh_fusion_plasma,
        show_wireframe=show_wireframe,
        opacity=0.6,
        cmap="plasma",
        name_prefix="plasma",
    )

    # Optionally add toroidal coils from PLY files
    if show_coils:
        coil_mesh_sets = ToroidalCoil3D.from_ply(coil_dir)
        plot_coils(
            plotter=plotter,
            coils=coil_mesh_sets,
            name_prefix="Toroidal Coil",
        )

    if show_cylindrical_angles:
        radius = np.max(np.linalg.norm(mesh_fusion_plasma.points[:, :2], axis=1))
        display_cylindrical_angles(plotter=plotter, n_angles=8, radius=radius)

    # Add light for dramatic effect
    light = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), color="white", intensity=0.5)
    plotter.add_light(light)

    # Add axes and bounds
    plotter.add_axes(line_width=2)
    plotter.show_bounds(color="white")

    # Render the visualization
    plotter.show(title="Fusion Plasma Surface")
