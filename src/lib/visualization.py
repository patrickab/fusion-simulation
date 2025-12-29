"""Utility functions for visualizing fusion simulation geometry."""

from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv

from src.lib.geometry_config import FusionPlasma, PlasmaBoundary, RotationalAngles, ToroidalCoil2D, ToroidalCoil3D
from src.lib.linalg_utils import convert_rz_to_xyz
from src.lib.utils import _coils_to_polydata, _plasma_to_polydata


def plot_toroidal_coils(
    plotter: pv.Plotter,
    toroidal_coils: list[ToroidalCoil3D] | list[pv.PolyData] | list[Path] | list[dict[str, pv.PolyData]],
    name_prefix: str = "Toroidal Coil",
) -> None:
    """Add toroidal coils (3D objects or loaded meshes) to the plotter."""
    coil_mesh_sets = _coils_to_polydata(toroidal_coils)

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


def plot_plasma(
    plotter: pv.Plotter,
    plasma: FusionPlasma | pv.PolyData | Path,
    show_wireframe: bool = False,
    opacity: float = 0.4,
    cmap: str = "plasma",
    name_prefix: str = "plasma",
) -> None:
    """Add a plasma surface (FusionPlasma or PolyData) to the plotter."""

    def get_sparse_wireframe(plasma_mesh: pv.PolyData, step: int = 8) -> Optional[pv.PolyData]:
        """Generate a sparse wireframe mesh from a structured plasma PolyData."""
        # The plasma mesh is assumed to be a structured grid flattened into points.
        # We reconstruct the (phi, theta, 3) shape using the known sampling counts.
        points = plasma_mesh.points.reshape((RotationalAngles.n_phi, RotationalAngles.n_theta, 3))
        sparse_points = points[::step, ::step, :]

        x = sparse_points[:, :, 0]
        y = sparse_points[:, :, 1]
        z = sparse_points[:, :, 2]

        return pv.StructuredGrid(x, y, z).extract_surface()

    plasma_mesh = _plasma_to_polydata(plasma)
    plasma_mesh.compute_normals(inplace=True)

    plotter.add_mesh(
        plasma_mesh,
        color=None,
        scalars=plasma_mesh.points[:, 2],
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
        sparse_wireframe = get_sparse_wireframe(plasma_mesh=plasma_mesh)
        plotter.add_mesh(
            sparse_wireframe,
            style="wireframe",
            color="cyan",
            line_width=1,
            opacity=0.2,
            name=f"{name_prefix}_wireframe",
        )


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
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radius = 10.0

    for angle in angles:
        x = np.array([0.0, 0.0])
        y = np.array([0.0, radius * np.cos(angle)])
        z = np.array([0.0, radius * np.sin(angle)])

        plotter.add_lines(
            np.column_stack((x, y, z)),
            color="lightgray",
            width=1,
            label=f"Phi {np.degrees(angle):.1f}°",
        )


def visualize_2d_geometry(plotter: pv.Plotter, plasma_boundary: PlasmaBoundary, toroidal_coil_2d: ToroidalCoil2D) -> None:
    """Display 2D boundaries of plasma and toroidal coil in the plotter."""
    plotter.add_title("2D Poloidal Cross-Section", font_size=16, color="white")

    # Rotate plasma boundary and toroidal coil 2d by 90 degrees around the x-axis
    rotation_phi = np.deg2rad(90.0)

    plasma_boundary_xyz = convert_rz_to_xyz(
        R=plasma_boundary.R_2d,
        Z=plasma_boundary.Z_2d,
        phi=rotation_phi,
    )
    coil_inner_xyz = convert_rz_to_xyz(
        R=toroidal_coil_2d.R_inner,
        Z=toroidal_coil_2d.Z_inner,
        phi=rotation_phi,
    )
    coil_outer_xyz = convert_rz_to_xyz(
        R=toroidal_coil_2d.R_outer,
        Z=toroidal_coil_2d.Z_outer,
        phi=rotation_phi,
    )
    coil_center_xyz = convert_rz_to_xyz(
        R=toroidal_coil_2d.R_center,
        Z=toroidal_coil_2d.Z_center,
        phi=rotation_phi,
    )

    # Add plasma boundary (2D poloidal cross-section)
    plotter.add_lines(
        np.column_stack(plasma_boundary_xyz),
        color="cyan",
        width=2,
    )

    # Add toroidal coil 2D boundaries (inner and outer)
    plotter.add_lines(
        np.column_stack(coil_inner_xyz),
        color="purple",
        width=2,
        label="Toroidal Coil Inner Boundary",
    )
    plotter.add_lines(
        np.column_stack(coil_outer_xyz),
        color="purple",
        width=2,
        label="Toroidal Coil Outer Boundary",
    )

    plotter.add_lines(
        np.column_stack(coil_center_xyz),
        color="purple",
        width=2,
        label="Toroidal Coil Center Boundary",
    )

    plotter.camera_position = "iso"


def initialize_plotter(shape: tuple[int, int] = (1, 1)) -> pv.Plotter:
    """Initialize a PyVista plotter with a black background and trackball style."""

    plotter = pv.Plotter(shape=shape, border=True, border_color="white")
    plotter.set_background("black")
    return plotter


def render_fusion_plasma(
    fusion_plasma: Path | FusionPlasma | pv.PolyData,
    toroidal_coils: list[ToroidalCoil3D] | list[pv.PolyData] | list[Path] | list[dict[str, pv.PolyData]],
    show_cylindrical_angles: bool = True,
    show_wireframe: bool = True,
    plotter: pv.Plotter | None = None,
) -> None:
    """Load and render a fusion plasma surface from a ``.ply`` file.

    Parameters
    ----------
    ply_file_path:
        Path to the PLY file containing the plasma mesh.
    show_cylindrical_angles:
        If ``True`` display angle guides for cylindrical coordinates.
    """
    mesh_fusion_plasma = _plasma_to_polydata(fusion_plasma)

    # Create a plotter
    if plotter is None:
        plotter = initialize_plotter()
        is_own_plotter = True
    else:
        is_own_plotter = False

    # Plot plasma (pass original argument; plot_plasma will normalize)
    plot_plasma(
        plotter=plotter,
        plasma=fusion_plasma,
        show_wireframe=show_wireframe,
        opacity=0.6,
        cmap="plasma",
        name_prefix="plasma",
    )

    # Optionally add toroidal coils from PLY files
    if toroidal_coils:
        plot_toroidal_coils(
            plotter=plotter,
            toroidal_coils=toroidal_coils,
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
    plotter.add_title("3D Toroidal Geometry", font_size=16, color="white")

    if is_own_plotter:
        plotter.show(title="Fusion Plasma Surface")
