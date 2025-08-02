"""Module for parametric expression of a tokamak plasma surfaces"""

from dataclasses import dataclass
import os

import numpy as np
import pyvista as pv

from src.lib.config import Filepaths, RotationalAngles
from src.lib.linalg_utils import convert_rz_to_xyz


@dataclass
class PlasmaConfig:
    """
    Configuration class for toroidal geometry parameters.
    """

    R0: float  # Major radius of the torus (m)
    a: float  # Minor radius of the torus (m)
    kappa: float  # Elongation factor
    delta: float  # Triangularity factor


@dataclass
class ToroidalCoilConfig:
    """
    Class representing a toroidal field coil.
    Contains parameters for the coil's geometry and position.
    """

    distance_from_plasma: float  # Plasma Boundary <-> Field Coil Center (m)
    radial_thickness: float  # radial thickness of the coil (m)
    vertical_thickness: float  # vertical thickness of the coil (m)
    angular_span: float  # angular span of the coil (degrees) - defines the coil's extent in the toroidal direction
    n_field_coils: int  # of field coils


@dataclass
class PlasmaBoundary:
    """
    R-Z coordinates of poloidal plasma boundary
    """

    R_2d: np.ndarray  # R coordinates (m)
    Z_2d: np.ndarray  # Z coordinates (m)


@dataclass
class FusionPlasma:
    """
    Class representing a toroidal surface.
    Contains the 3D coordinates of the toroidal plasma surface.
    """

    X: np.ndarray  # X coordinates (m)
    Y: np.ndarray  # Y coordinates (m)
    Z: np.ndarray  # Z coordinates (m)
    Boundary: PlasmaBoundary

    def to_ply(self, filename: str = Filepaths.REACTOR_POLYGONIAL_MESH) -> None:
        """
        Exports the toroidal plasma surface to a polygonal mesh in .ply format.
        """
        grid = pv.StructuredGrid(self.X, self.Y, self.Z).extract_surface()
        grid.save(filename)
        print(f"✅ Exported plasma surface to: {os.path.abspath(filename)}")


@dataclass
class ToroidalCoil2D:
    """
    Represents 2D toroidal field coil.
    """

    R_inner: np.ndarray  # Inner R coordinates (m)
    R_outer: np.ndarray  # Outer R coordinates (m)

    Z_inner: np.ndarray  # Inner Z coordinates (m)
    Z_outer: np.ndarray  # Outer Z coordinates (m)


@dataclass
class ToroidalCoil:
    """
    Class representing a toroidal field coil.
    Contains the 3D coordinates of the toroidal field coil surface.
    """

    X: np.ndarray  # X coordinates (m)
    Y: np.ndarray  # Y coordinates (m)
    Z: np.ndarray  # Z coordinates (m)
    Center: np.ndarray  # Center coordinates (m)
    ToroidalCoil2d: ToroidalCoil2D


def calculate_poloidal_boundary(plasma_config: PlasmaConfig) -> PlasmaBoundary:
    """
    Calculate the poloidal plasma boundary in R-Z coordinates.
    """

    # Define plasma boundary
    theta = RotationalAngles.THETA
    R_plasma = plasma_config.R0 + plasma_config.a * np.cos(theta + plasma_config.delta * np.sin(theta))
    Z_plasma = plasma_config.kappa * plasma_config.a * np.sin(theta)
    return PlasmaBoundary(R_2d=R_plasma, Z_2d=Z_plasma)


def calculate_toroidal_coil_boundary(plasma_boundary: PlasmaBoundary, toroid_coil_config: ToroidalCoilConfig) -> ToroidalCoil2D:
    """
    Compute toroidal coil 2D cross-section by offsetting plasma boundary along normal vectors.
    The inner boundary is offset by `distance_from_plasma`, and the outer boundary is defined
    by adding the coil thickness along the poloidal normals.
    """

    # Base plasma boundary
    R = plasma_boundary.R_2d
    Z = plasma_boundary.Z_2d

    # Compute tangents
    dR = np.gradient(R, RotationalAngles.THETA)
    dZ = np.gradient(Z, RotationalAngles.THETA)

    # Compute unit normals (rotate tangent by 90 degrees)
    N_R = dZ
    N_Z = -dR
    norm = np.sqrt(N_R**2 + N_Z**2)
    N_R /= norm
    N_Z /= norm

    # Offset inner boundary
    inner_offset = toroid_coil_config.distance_from_plasma
    R_inner = R + inner_offset * N_R
    Z_inner = Z + inner_offset * N_Z

    # Offset outer boundary
    coil_thickness = toroid_coil_config.radial_thickness
    R_outer = R_inner + coil_thickness * N_R
    Z_outer = Z_inner + coil_thickness * N_Z

    return ToroidalCoil2D(R_inner=R_inner, R_outer=R_outer, Z_inner=Z_inner, Z_outer=Z_outer)


def calculate_2d_geometry(
    plasma_config: PlasmaConfig, toroid_coil_config: ToroidalCoilConfig
) -> tuple[PlasmaBoundary, ToroidalCoil2D]:
    """
    Returns R and Z coordinates for a 2D poloidal plasma boundary shape.
    Cross-section of a tokamak plasma in the poloidal plane.
    """

    plasma_boundary = calculate_poloidal_boundary(plasma_config)
    toroidal_coil_2d = calculate_toroidal_coil_boundary(plasma_boundary, toroid_coil_config)

    return plasma_boundary, toroidal_coil_2d


def generate_fusion_plasma(plasma_boundary: PlasmaBoundary) -> FusionPlasma:
    """
    Generates a 3D toroidal surface by rotating a poloidal cross-section around the Z-axis.

    This function creates a tokamak-like surface with elongation (kappa) and triangularity (delta).
    The process works as follows:

    1. Generate poloidal coordinates
    2. Create a meshgrid that extends this 2D shape into 3D space:
       - (1) np.meshgrid(R_2D, phi) creates two 2D arrays:
         * R_grid: Contains the R coordinates repeated for each toroidal angle
         * phi_grid: Contains the toroidal angles repeated for each point on the poloidal contour
       - This effectively creates a parametric surface where each toroidal section (phi=constant)
         has identical poloidal cross-sections
    3. Extend Z coordinates by repeating the Z_2D array for each toroidal angle using np.tile
    4. Transform from cylindrical coordinates (R, φ, Z) to Cartesian coordinates (X, Y, Z):
       - X = R * cos(φ)
       - Y = R * sin(φ)
       - Z remains unchanged

    This transformation maps the toroidal surface into 3D Cartesian space, where each poloidal
    cross-section is identical but rotated around the Z-axis according to the toroidal angle φ.
    """

    # Create 2D meshgrid for revolution
    R_grid, phi_grid = np.meshgrid(plasma_boundary.R_2d, RotationalAngles.PHI)
    Z_grid = np.tile(plasma_boundary.Z_2d, (RotationalAngles.n_phi, 1))

    # Convert cylindrical (R, φ, Z) → Cartesian (X, Y, Z)
    X = R_grid * np.cos(phi_grid)
    Y = R_grid * np.sin(phi_grid)
    Z = Z_grid

    return FusionPlasma(X=X, Y=Y, Z=Z, Boundary=plasma_boundary)


def generate_toroidal_coils_3d(
    toroidal_coil_2d: ToroidalCoil2D,
    toroid_coil_config: ToroidalCoilConfig,
) -> list[ToroidalCoil]:
    """
    Generate full 3D geometry for toroidal coils from 2D cross-section.
    """
    phi_angles = np.linspace(0, 2 * np.pi, toroid_coil_config.n_field_coils, endpoint=False)
    coils = []

    # Calculate the 3D coordinates for the toroidal coil at angle phi
    for phi in phi_angles:
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Inner surface
        X_inner = toroidal_coil_2d.R_inner * cos_phi
        Y_inner = toroidal_coil_2d.R_inner * sin_phi
        Z_inner = toroidal_coil_2d.Z_inner

        # Outer surface
        X_outer = toroidal_coil_2d.R_outer * cos_phi
        Y_outer = toroidal_coil_2d.R_outer * sin_phi
        Z_outer = toroidal_coil_2d.Z_outer

        X = np.vstack((X_inner, X_outer))
        Y = np.vstack((Y_inner, Y_outer))
        Z = np.vstack((Z_inner, Z_outer))

        center_R = 0.5 * (toroidal_coil_2d.R_inner + toroidal_coil_2d.R_outer)
        center_Z = 0.5 * (toroidal_coil_2d.Z_inner + toroidal_coil_2d.Z_outer)
        center_X = np.mean(center_R) * cos_phi
        center_Y = np.mean(center_R) * sin_phi
        center = np.array([center_X, center_Y, np.mean(center_Z)])
        coils.append(ToroidalCoil(X=X, Y=Y, Z=Z, Center=center, ToroidalCoil2d=toroidal_coil_2d))

    return coils


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
    n_angles = 16
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


if __name__ == "__main__":
    plasma_config = PlasmaConfig(
        R0=6.2,  # Major radius (m)
        a=2.0,  # Minor radius (m)
        kappa=1.7,  # Elongation factor
        delta=0.33,  # Triangularity factor
    )
    toroid_coil_config = ToroidalCoilConfig(
        distance_from_plasma=1.5,  # Distance from plasma surface (m)
        radial_thickness=0.8,  # Radial thickness of the coil (m)
        vertical_thickness=0.2,  # Vertical thickness of the coil (m)
        angular_span=30,  # Angular span of the coil (degrees)
        n_field_coils=32,  # Number of field coils
    )

    plasma_boundary, toroidal_coil_2d = calculate_2d_geometry(plasma_config=plasma_config, toroid_coil_config=toroid_coil_config)

    fusion_plasma = generate_fusion_plasma(plasma_boundary=plasma_boundary)
    toroidal_coils_3d = generate_toroidal_coils_3d(toroidal_coil_2d=toroidal_coil_2d, toroid_coil_config=toroid_coil_config)

    # Visualization setup
    plotter = pv.Plotter(window_size=(1024, 768))
    plotter.set_background(color="black")
    plotter.add_title("Fusion Plasma Surface with Toroidal Coil and Plasma Boundary", font_size=16, color="white")

    cmap = "plasma"

    # Add plasma surface mesh to plotter
    mesh_fusion_plasma = pv.read(Filepaths.REACTOR_POLYGONIAL_MESH)
    mesh_fusion_plasma.compute_normals(inplace=True)  # Compute normals for smooth shading

    plotter.add_mesh(
        mesh_fusion_plasma,
        color=None,
        scalars=mesh_fusion_plasma.points[:, 2],  # Use Z-axis position for coloring
        cmap=cmap,
        smooth_shading=True,
        opacity=0.9,
        show_edges=False,
        lighting=True,
        specular=0.4,
        specular_power=15,
        name="plasma",
    )

    # Rotate plasma boundary and toroidal coil 2d by 90 degrees around the x-axis
    radians_90 = 90 * np.pi / 180
    plasma_x, plasma_y, plasma_z = convert_rz_to_xyz(R=plasma_boundary.R_2d, Z=plasma_boundary.Z_2d, phi=radians_90)
    coil_inner_x, coil_inner_y, coil_inner_z = convert_rz_to_xyz(
        R=toroidal_coil_2d.R_inner, Z=toroidal_coil_2d.Z_inner, phi=radians_90
    )
    coil_outer_x, coil_outer_y, coil_outer_z = convert_rz_to_xyz(
        R=toroidal_coil_2d.R_outer, Z=toroidal_coil_2d.Z_outer, phi=radians_90
    )

    toroidal_coil_names = [f"Toroidal Coil {i + 1}" for i in range(toroid_coil_config.n_field_coils)]
    for coil, name in zip(toroidal_coils_3d, toroidal_coil_names, strict=False):
        coil_mesh = pv.StructuredGrid(coil.X, coil.Y, coil.Z).extract_surface()
        plotter.add_mesh(
            coil_mesh,
            color="silver",
            opacity=0.6,
            name=name,
            specular=0.8,  # Add specular reflection for a metallic-like appearance
            specular_power=20,
        )

    # Add plasma boundary (2D poloidal cross-section)
    plotter.add_lines(
        np.column_stack((plasma_x, plasma_y, plasma_z)),
        color="cyan",
        width=2,
        label="Plasma Boundary",
    )

    # Add toroidal coil 2D boundaries (inner and outer)
    plotter.add_lines(
        np.column_stack((coil_inner_x, coil_inner_y, coil_inner_z)),
        color="yellow",
        width=2,
        label="Toroidal Coil Inner Boundary",
    )
    plotter.add_lines(
        np.column_stack((coil_outer_x, coil_outer_y, coil_outer_z)),
        color="orange",
        width=2,
        label="Toroidal Coil Outer Boundary",
    )

    # display_theta_coordinates(plotter)
    # display_phi_coordinates(plotter)

    # Add light for dramatic effect
    light = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), color="white", intensity=0.9)
    plotter.add_light(light)

    # Set camera view
    plotter.camera_position = "iso"

    # Add axes and bounds
    plotter.add_axes(line_width=2)
    plotter.show_bounds(color="white")

    # Render the visualization
    plotter.show(title="Fusion Plasma Surface with Toroidal Coil and Plasma Boundary")
