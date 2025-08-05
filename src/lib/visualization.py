"""Utility functions for visualizing fusion simulation geometry."""

import os

import numpy as np
import pyvista as pv

from src.lib.config import Filepaths
from src.lib.geometry_config import FusionPlasma, PlasmaBoundary, ToroidalCoil2D, ToroidalCoil3D
from src.lib.linalg_utils import convert_rz_to_xyz

PATH_PLASMA_SURFACE = Filepaths.ROOT + "/" + Filepaths.OUTPUT_DIR + "/" + Filepaths.REACTOR_POLYGONIAL_MESH


def export_polygonal_plasmasurface(
    fusion_plasma: FusionPlasma,
    filename: str = Filepaths.REACTOR_POLYGONIAL_MESH,
) -> None:
    """Converts the toroidal plasma surface to a polygonal mesh & stores it as .ply"""

    grid = pv.StructuredGrid(fusion_plasma.X, fusion_plasma.Y, fusion_plasma.Z).extract_surface()

    grid.save(filename)
    print(f"✅ Exported plasma surface to: {os.path.abspath(filename)}")


def display_theta_coordinates(
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


def visualize_3d_geometry(
    plotter: pv.Plotter,
    fusion_plasma: FusionPlasma,
    toroidal_coils_3d: list[ToroidalCoil3D],
) -> None:
    """Visualize the 3D geometry of the fusion plasma and toroidal coils.

    Parameters
    ----------
    plotter:
        PyVista plotter used for rendering.
    fusion_plasma:
        Plasma surface geometry.
    toroidal_coils_3d:
        List of coils generated from :func:`generate_toroidal_coils_3d`.
    """

    # Add plasma surface mesh to plotter
    mesh_fusion_plasma = pv.StructuredGrid(fusion_plasma.X, fusion_plasma.Y, fusion_plasma.Z).extract_surface()
    mesh_fusion_plasma.compute_normals(inplace=True)  # Compute normals for smooth shading

    plotter.add_mesh(
        mesh_fusion_plasma,
        color=None,
        scalars=mesh_fusion_plasma.points[:, 2],  # Use Z-axis position for coloring
        cmap="plasma",
        smooth_shading=True,
        opacity=0.6,
        show_edges=False,
        lighting=True,
        specular=0.2,
        specular_power=15,
        name="plasma",
    )

    colors = ["silver", "gold"]
    for coil_idx, coil in enumerate(toroidal_coils_3d, start=1):
        radial_thickness = float(
            np.mean(coil.ToroidalCoil2D.R_outer - coil.ToroidalCoil2D.R_inner)
        )
        segment_length = 2 * radial_thickness

        # Approximate centerline of coil to measure arc length
        x_c = (coil.X_inner + coil.X_outer).mean(axis=1)
        y_c = (coil.Y_inner + coil.Y_outer).mean(axis=1)
        z_c = (coil.Z_inner + coil.Z_outer).mean(axis=1)
        ds = np.sqrt(np.diff(x_c) ** 2 + np.diff(y_c) ** 2 + np.diff(z_c) ** 2)
        s = np.insert(np.cumsum(ds), 0, 0)

        edges = [*np.arange(0, s[-1], segment_length), s[-1]]
        idx_edges = np.searchsorted(s, edges)
        idx_edges = np.unique(idx_edges)
        if idx_edges[-1] != len(s) - 1:
            idx_edges = np.append(idx_edges, len(s) - 1)

        for seg_idx in range(len(idx_edges) - 1):
            start = idx_edges[seg_idx]
            end = idx_edges[seg_idx + 1]
            if end <= start:
                continue

            name = f"Coil {coil_idx} Block {seg_idx + 1}"
            color = colors[seg_idx % len(colors)]

            # Slice inner/outer surfaces for this block
            X_inner = coil.X_inner[start : end + 1, :]
            Y_inner = coil.Y_inner[start : end + 1, :]
            Z_inner = coil.Z_inner[start : end + 1, :]
            X_outer = coil.X_outer[start : end + 1, :]
            Y_outer = coil.Y_outer[start : end + 1, :]
            Z_outer = coil.Z_outer[start : end + 1, :]

            inner_mesh = pv.StructuredGrid(X_inner, Y_inner, Z_inner).extract_surface()
            plotter.add_mesh(
                inner_mesh,
                color=color,
                opacity=0.9,
                name=f"{name} Inner",
                specular=0.8,
                specular_power=128,
            )

            outer_mesh = pv.StructuredGrid(X_outer, Y_outer, Z_outer).extract_surface()
            plotter.add_mesh(
                outer_mesh,
                color=color,
                opacity=0.9,
                name=f"{name} Outer",
                label=name,
                specular=0.8,
                specular_power=128,
            )

            # Start and end caps for block
            cap_start_mesh = pv.StructuredGrid(
                np.vstack([coil.X_inner[start], coil.X_outer[start]]),
                np.vstack([coil.Y_inner[start], coil.Y_outer[start]]),
                np.vstack([coil.Z_inner[start], coil.Z_outer[start]]),
            ).extract_surface()
            plotter.add_mesh(
                cap_start_mesh,
                color=color,
                opacity=0.9,
                name=f"{name} Start",
                specular=0.8,
                specular_power=128,
            )

            cap_end_mesh = pv.StructuredGrid(
                np.vstack([coil.X_inner[end], coil.X_outer[end]]),
                np.vstack([coil.Y_inner[end], coil.Y_outer[end]]),
                np.vstack([coil.Z_inner[end], coil.Z_outer[end]]),
            ).extract_surface()
            plotter.add_mesh(
                cap_end_mesh,
                color=color,
                opacity=0.9,
                name=f"{name} End",
                specular=0.8,
                specular_power=128,
            )

    plotter.add_legend()
    plotter.add_title("3D Toroidal Geometry", font_size=16, color="white")
    set_camera_relative_to_body(plotter, distance_factor=3)
    display_theta_coordinates(plotter)
    # display_phi_coordinates(plotter)


def initialize_plotter(shape: tuple[int, int] = (1, 1)) -> pv.Plotter:
    """Initialize a PyVista plotter with a black background and trackball style."""

    plotter = pv.Plotter(shape=shape, border=True, border_color="white")

    plotter.set_background("black")
    plotter.enable_trackball_style()
    plotter.camera.enable_parallel_projection()
    return plotter


def render_fusion_plasma(
    ply_file_path: str = PATH_PLASMA_SURFACE,
    show_cylindrical_angles: bool = False,
) -> None:
    """Load and render a fusion plasma surface from a ``.ply`` file.

    Parameters
    ----------
    ply_file_path:
        Path to the PLY file containing the plasma mesh.
    show_cylindrical_angles:
        If ``True`` display angle guides for cylindrical coordinates.
    """

    # Mark unused arguments
    _ = show_cylindrical_angles

    # Load the plasma mesh
    mesh = pv.read(ply_file_path)

    # Create a plotter
    plotter = pv.Plotter(window_size=(1024, 768))
    plotter.background_color.set_background(color="black")

    # Compute normals for smooth shading
    mesh.compute_normals(inplace=True)

    # Choose an appealing color map
    cmap = "plasma"

    # Add mesh to plotter
    plotter.add_mesh(
        mesh,
        color=None,
        scalars=mesh.points[:, 2],
        cmap=cmap,
        smooth_shading=True,
        opacity=0.9,
        show_edges=False,
        lighting=True,
        specular=0.4,
        specular_power=15,
        name="plasma",
    )

    # Add light for dramatic effect
    light = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), color="white", intensity=0.9)
    plotter.add_light(light)

    # Set camera view
    plotter.camera_position = "iso"

    # Add axes and bounds
    plotter.add_axes(line_width=2)
    plotter.show_bounds(color="white")

    # Render the visualization
    plotter.show(title="Fusion Plasma Surface")
