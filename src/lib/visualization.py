"""Utility functions for visualizing fusion simulation geometry."""

from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv

from src.engine.plasma import calculate_poloidal_boundary, generate_fusion_plasma
from src.lib.config import Filepaths
from src.lib.geometry_config import (
    FusionPlasma,
    PlasmaBoundary,
    PlasmaConfig,
    RotationalAngles,
    ToroidalCoil2D,
    ToroidalCoil3D,
    ToroidalCoilConfig,
)
from src.lib.linalg_utils import convert_rz_to_xyz
from src.lib.utils import _coils_to_polydata, _plasma_to_polydata
from src.toroidal_geometry import calculate_toroidal_coil_boundary, generate_toroidal_coils_3d


def initialize_plotter(shape: tuple[int, int] = (1, 1)) -> pv.Plotter:
    """Create PyVista plotter with standard theme."""
    plotter = pv.Plotter(shape=shape, border=True, border_color="white")
    plotter.set_background("black")
    return plotter


def calculate_2d_geometry(
    plasma_config: PlasmaConfig, toroid_coil_config: ToroidalCoilConfig
) -> tuple[PlasmaBoundary, ToroidalCoil2D]:
    """Compute 2D plasma and coil boundaries."""
    theta = RotationalAngles.THETA
    plasma_boundary = calculate_poloidal_boundary(theta=theta, plasma_config=plasma_config)
    toroidal_coil_2d = calculate_toroidal_coil_boundary(
        theta=theta, plasma_config=plasma_config, toroid_coil_config=toroid_coil_config
    )
    return plasma_boundary, toroidal_coil_2d


def plot_toroidal_coils(
    plotter: pv.Plotter,
    toroidal_coils: list[ToroidalCoil3D]
    | list[pv.PolyData]
    | list[Path]
    | list[dict[str, pv.PolyData]],
    name_prefix: str = "Toroidal Coil",
) -> None:
    """Add 3D coil meshes to active plotter.

    Args:
        plotter: target plotter
        toroidal_coils: coil data or paths
        name_prefix: label prefix
    """
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
    """Add plasma surface and optional wireframe to plotter.

    Args:
        plotter: target plotter
        plasma: plasma data or path
        show_wireframe: toggle sparse grid
        opacity: transparency
        cmap: colormap
        name_prefix: label prefix
    """

    def get_sparse_wireframe(plasma_mesh: pv.PolyData, step: int = 8) -> Optional[pv.PolyData]:
        """Generate a sparse wireframe mesh from a structured plasma PolyData."""
        # Extract and reshape points into structured grid format
        points = plasma_mesh.points.reshape((RotationalAngles.n_phi, RotationalAngles.n_theta, 3))

        # Create sparse sampling
        sparse_points = points[::step, ::step, :]

        # Extract coordinate arrays for structured grid
        x, y, z = sparse_points[:, :, 0], sparse_points[:, :, 1], sparse_points[:, :, 2]
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
    """Add radial reference lines for toroidal angles.

    Args:
        plotter: target plotter
        n_angles: number of lines
        radius: line length
        color: line color
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


def render_2d_geometry(
    plotter: pv.Plotter, plasma_boundary: PlasmaBoundary, toroidal_coil_2d: ToroidalCoil2D
) -> None:
    """Plot 2D cross-section of plasma and coils.

    Args:
        plotter: target plotter
        plasma_boundary: 2D plasma
        toroidal_coil_2d: 2D coil
    """
    plotter.add_title("2D Poloidal Cross-Section", font_size=16, color="white")

    ROTATION_ANGLE = np.deg2rad(90.0)
    COIL_BOUNDARIES = [
        (toroidal_coil_2d.R_inner, toroidal_coil_2d.Z_inner, "Inner"),
        (toroidal_coil_2d.R_outer, toroidal_coil_2d.Z_outer, "Outer"),
        (toroidal_coil_2d.R_center, toroidal_coil_2d.Z_center, "Center"),
    ]

    plasma_points_3d = convert_rz_to_xyz(
        R=plasma_boundary.R,
        Z=plasma_boundary.Z,
        phi=ROTATION_ANGLE,
    )

    # Add plasma boundary (2D poloidal cross-section)
    plotter.add_lines(
        np.column_stack(plasma_points_3d),
        color="cyan",
        width=2,
        label="Plasma Boundary",
    )

    # Add all toroidal coil boundaries
    for R_coil, Z_coil, boundary_type in COIL_BOUNDARIES:
        coil_points_3d = convert_rz_to_xyz(R=R_coil, Z=Z_coil, phi=ROTATION_ANGLE)
        plotter.add_lines(
            np.column_stack(coil_points_3d),
            color="purple",
            width=2,
            label=f"Toroidal Coil {boundary_type} Boundary",
        )

    plotter.camera_position = "iso"


def render_fusion_plasma(
    fusion_plasma: Path | FusionPlasma | pv.PolyData,
    toroidal_coils: list[ToroidalCoil3D]
    | list[pv.PolyData]
    | list[Path]
    | list[dict[str, pv.PolyData]],
    show_cylindrical_angles: bool = True,
    show_wireframe: bool = True,
    plotter: pv.Plotter | None = None,
) -> None:
    """Orchestrate 3D rendering of plasma and coils.

    Args:
        fusion_plasma: plasma data
        toroidal_coils: coil list
        show_cylindrical_angles: toggle guides
        show_wireframe: toggle wireframe
        plotter: optional existing plotter
    """

    # Early return if no valid plasma data
    if fusion_plasma is None:
        return

    # Create plotter if not provided
    plotter_was_provided = plotter is not None
    plotter = plotter or initialize_plotter()

    # Convert plasma to mesh format
    plasma_mesh = _plasma_to_polydata(fusion_plasma)

    # Plot plasma surface
    plot_plasma(
        plotter=plotter,
        plasma=fusion_plasma,
        show_wireframe=show_wireframe,
        opacity=0.8,
        cmap="plasma",
        name_prefix="plasma",
    )

    # Add toroidal coils if provided
    if toroidal_coils:
        plot_toroidal_coils(
            plotter=plotter,
            toroidal_coils=toroidal_coils,
            name_prefix="Toroidal Coil",
        )

    # Add cylindrical angle guides if requested
    if show_cylindrical_angles:
        max_radius = np.max(np.linalg.norm(plasma_mesh.points[:, :2], axis=1))
        display_cylindrical_angles(plotter=plotter, n_angles=8, radius=max_radius)

    # Enhance visualization with lighting and reference elements
    light = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), color="white", intensity=0.5)
    plotter.add_light(light)

    plotter.add_axes(line_width=2)
    plotter.show_bounds(color="white")
    plotter.add_title("3D Toroidal Geometry", font_size=16, color="white")

    # Only show plotter if created internally
    if not plotter_was_provided:
        plotter.show(title="Fusion Plasma Surface")


def render_all_geometries() -> None:
    """Execute full simulation pipeline from config to visualization and export.

    Note: Modifies filesystem and opens interactive window.
    """
    plasma_config = PlasmaConfig(
        R0=6.2,  # Major radius (m)
        a=3.2,  # Minor radius (m)
        kappa=1.7,  # Elongation factor
        delta=0.33,  # Triangularity factor
    )
    toroid_coil_config = ToroidalCoilConfig(
        distance_from_plasma=1.5,  # Distance from plasma surface (m)
        radial_thickness=0.8,  # Radial thickness of the coil (m)
        vertical_thickness=0.2,  # Vertical thickness of the coil (m)
        angular_span=6,  # Angular span of the coil (degrees)
        n_field_coils=8,  # Number of field coils
    )

    plasma_boundary, toroidal_coil_2d = calculate_2d_geometry(
        plasma_config=plasma_config, toroid_coil_config=toroid_coil_config
    )

    fusion_plasma = generate_fusion_plasma(plasma_boundary=plasma_boundary)
    toroidal_coils_3d = generate_toroidal_coils_3d(
        toroidal_coil_2d=toroidal_coil_2d, toroid_coil_config=toroid_coil_config
    )

    plotter = initialize_plotter(shape=(1, 2))

    plotter.subplot(0, 0)
    render_2d_geometry(
        plotter=plotter, plasma_boundary=plasma_boundary, toroidal_coil_2d=toroidal_coil_2d
    )
    plotter.subplot(0, 1)
    render_fusion_plasma(
        plotter=plotter,
        fusion_plasma=fusion_plasma,
        toroidal_coils=toroidal_coils_3d,
    )

    # Render the visualization
    plotter.show(title="Fusion Reactor Visualization", interactive=True)

    # Save geometry to PLY files
    fusion_plasma.to_ply_structuregrid(Filepaths.PLASMA_SURFACE)
    for i, coil in enumerate(toroidal_coils_3d, start=1):
        coil.to_ply(base_path=Filepaths.TOROIDAL_COIL_3D_DIR, coil_id=i)


if __name__ == "__main__":
    render_all_geometries()
