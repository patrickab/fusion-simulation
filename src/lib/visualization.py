"""Utility functions for visualizing fusion simulation geometry."""

from __future__ import annotations
import numpy as np
import pyvista as pv

from src.lib.config import Filepaths

PATH_PLASMA_SURFACE = Filepaths.ROOT + "/" + Filepaths.OUTPUT_DIR + "/" + Filepaths.REACTOR_POLYGONIAL_MESH


def add_cylindrical_angle_guides(plotter: pv.Plotter, radius: float, n_angles: int = 8) -> None:
    """Add radial lines and labels indicating cylindrical coordinate angles.

    Parameters
    ----------
    plotter:
        Active :class:`pyvista.Plotter` instance.
    radius:
        Length of each radial line.
    n_angles:
        Number of angle markers (default eight, every 45 degrees).
    """

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    for angle in angles:
        start = (0.0, 0.0, 0.0)
        end = (radius * np.cos(angle), radius * np.sin(angle), 0.0)
        plotter.add_mesh(pv.Line(start, end), color="white", line_width=1)
        label_position = (1.1 * end[0], 1.1 * end[1], 0.0)
        plotter.add_point_labels(
            [label_position],
            [f"{np.degrees(angle):.0f}°"],
            text_color="white",
            font_size=10,
            point_size=0,
        )


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
    plotter.set_background(color="black")

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

    if show_cylindrical_angles:
        radius = np.max(np.linalg.norm(mesh.points[:, :2], axis=1))
        add_cylindrical_angle_guides(plotter, radius)

    # Render the visualization
    plotter.show(title="Fusion Plasma Surface")
