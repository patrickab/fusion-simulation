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
    """Visualize the 3D geometry of the fusion plasma and toroidal coils."""

    # Add plasma surface mesh to plotter
    mesh_fusion_plasma = pv.StructuredGrid(fusion_plasma.X, fusion_plasma.Y, fusion_plasma.Z).extract_surface()
    mesh_fusion_plasma.compute_normals(inplace=True)  # Compute normals for smooth shading

    # Create a sparser wireframe by subsampling the mesh points
    # Subsample every Nth point along each axis (e.g., every 4th point)
    def add_sparse_wireframe(plotter: pv.Plotter, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Add a sparse wireframe mesh to the plotter."""

        # Create a structured grid from the sparse points
        N = 8
        x = fusion_plasma.X[::N, ::N]
        y = fusion_plasma.Y[::N, ::N]
        z = fusion_plasma.Z[::N, ::N]
        sparse_mesh = pv.StructuredGrid(x, y, z).extract_surface()
        plotter.add_mesh(sparse_mesh, style="wireframe", color="cyan", line_width=2, opacity=0.7)

    add_sparse_wireframe(plotter, fusion_plasma.X, fusion_plasma.Y, fusion_plasma.Z)

    plasma_opacity = 0.8
    plotter.add_mesh(
        mesh_fusion_plasma,
        color=None,
        scalars=mesh_fusion_plasma.points[:, 2],  # Use Z-axis position for coloring
        cmap="plasma",
        smooth_shading=True,
        opacity=plasma_opacity,
        show_edges=False,
        lighting=True,
        specular=0.2,
        specular_power=15,
        name="plasma_surface",
    )

    coil_opacity = 1
    coil_color = (0.8, 0.8, 0.85)
    toroidal_coil_names = [f"Toroidal Coil {i + 1}" for i in range(len(toroidal_coils_3d))]
    for coil, name in zip(toroidal_coils_3d, toroidal_coil_names, strict=False):
        # Inner surface
        inner_mesh = pv.StructuredGrid(coil.X_inner, coil.Y_inner, coil.Z_inner).extract_surface()
        plotter.add_mesh(
            inner_mesh,
            color=coil_color,
            opacity=coil_opacity,
            name=f"{name} Inner",
            specular=0.8,
            specular_power=128,
        )

        # Outer surface
        outer_mesh = pv.StructuredGrid(coil.X_outer, coil.Y_outer, coil.Z_outer).extract_surface()
        plotter.add_mesh(
            outer_mesh,
            color=coil_color,
            opacity=coil_opacity,
            name=f"{name} Outer",
            specular=0.8,
            specular_power=128,
        )

        # Start cap
        cap_start_mesh = pv.StructuredGrid(coil.X_cap_start, coil.Y_cap_start, coil.Z_cap_start).extract_surface()
        plotter.add_mesh(
            cap_start_mesh,
            color=coil_color,
            opacity=coil_opacity,
            name=f"{name} Cap Start",
            specular=0.8,
            specular_power=128,
            render_points_as_spheres=True,
            point_size=6,
        )

        # End cap
        cap_end_mesh = pv.StructuredGrid(coil.X_cap_end, coil.Y_cap_end, coil.Z_cap_end).extract_surface()
        plotter.add_mesh(
            cap_end_mesh,
            color=coil_color,
            opacity=coil_opacity,
            name=f"{name} Cap End",
            specular=0.8,
            specular_power=128,
            render_points_as_spheres=True,
            point_size=6,
        )

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

    # Load the plasma mesh
    mesh = pv.read(ply_file_path)

    # Create a plotter
    plotter = pv.Plotter(window_size=(1024, 768))
    plotter.background_color.set_background(color="black")

    # Compute normals for smooth shading
    mesh.compute_normals(inplace=True)

    # Choose an appealing color map
    cmap = "plasma"

    # Add mesh to plotter (surface)
    plotter.add_mesh(
        mesh,
        color=None,
        scalars=mesh.points[:, 2],
        cmap=cmap,
        smooth_shading=True,
        opacity=0.4,
        show_edges=False,
        lighting=True,
        specular=0.4,
        specular_power=15,
        name="plasma_surface",
    )
    # Overlay wireframe for emphasis
    plotter.add_mesh(
        mesh,
        style="wireframe",
        color="cyan",
        line_width=2,
        name="plasma_wireframe",
    )

    # Add light for dramatic effect
    light = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), color="white", intensity=0.9)
    plotter.add_light(light)

    # Set camera view
    plotter.camera_position = "iso"

    # Add axes and bounds
    plotter.add_axes(line_width=2)
    plotter.show_bounds(color="white")

    if show_cylindrical_angles:
        radius = np.max(np.linalg.norm(mesh.points[:, :2], axis=1))
        add_cylindrical_angle_guides(plotter, radius)

    # Render the visualization
    plotter.show(title="Fusion Plasma Surface")
