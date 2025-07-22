"""Visualization of Fusion Plasma Surface using PyVista"""

import pyvista as pv

from src.lib.config import Filepaths

PATH_PLASMA_SURFACE = (
    Filepaths.ROOT + "/" + Filepaths.OUTPUT_DIR + "/" + Filepaths.REACTOR_POLYGONIAL_MESH
)


def render_fusion_plasma(ply_file_path: str=PATH_PLASMA_SURFACE) -> None:
    """
    Load and render a fusion plasma surface from a .ply file using PyVista.

    Parameters:
        ply_file_path (str): Path to the PLY file containing the plasma mesh.
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
        scalars=mesh.points[:, 2],  # Use Z-axis position for coloring
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


if __name__ == "__main__":
    render_fusion_plasma()
