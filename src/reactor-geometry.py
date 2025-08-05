"""Module for parametric expression of a tokamak plasma surfaces"""

from dataclasses import dataclass
import os

import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay

from src.lib.config import Filepaths, RotationalAngles

COIL_RESOLUTION_3D = 64  # Number of points in the toroidal direction for 3D coils


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

    def to_ply_structuregrid(self, filename: str = Filepaths.REACTOR_POLYGONIAL_MESH) -> None:
        """
        Exports the toroidal plasma surface to a polygonal mesh in .ply format.
        """
        grid = pv.StructuredGrid(self.X, self.Y, self.Z).extract_surface()
        grid.save(filename)
        print(f"✅ Exported plasma surface to: {os.path.abspath(filename)}")


@dataclass
class ToroidalCoil2D:
    """
    Represents 2D toroidal field coil in the poloidal (R-Z) plane.
    Also provides Delaunay triangulation for meshing.
    """

    R_inner: np.ndarray
    R_outer: np.ndarray
    R_center: np.ndarray

    Z_inner: np.ndarray
    Z_outer: np.ndarray
    Z_center: np.ndarray

    def express_delaunay_triangles(self) -> Delaunay:
        """
        Returns Delaunay triangulation of the inner and outer boundaries in the R-Z plane.
        """
        # Stack inner and outer boundary points
        points = np.column_stack((np.concatenate([self.R_inner, self.R_outer]), np.concatenate([self.Z_inner, self.Z_outer])))
        return Delaunay(points)


@dataclass
class ToroidalCoil3D:
    """
    Represents a full 3D toroidal field coil geometry.
    Suitable for visualization, export, and simulation.
    Also provides Delaunay triangulation for meshing.
    """

    X_inner: np.ndarray  # (n_phi, COIL_RESOLUTION_3D) inner wall swept
    Y_inner: np.ndarray
    Z_inner: np.ndarray

    X_outer: np.ndarray  # (n_phi, COIL_RESOLUTION_3D) outer wall swept
    Y_outer: np.ndarray
    Z_outer: np.ndarray

    X_cap_start: np.ndarray  # (2, n_pts) cap at start of toroidal span
    Y_cap_start: np.ndarray
    Z_cap_start: np.ndarray

    X_cap_end: np.ndarray  # (2, n_pts) cap at end of toroidal span
    Y_cap_end: np.ndarray
    Z_cap_end: np.ndarray

    CentralPlane: np.ndarray

    ToroidalCoil2D: ToroidalCoil2D  # Original 2D definition (for regeneration/reuse)

    def express_delaunay_triangles(self) -> Delaunay:
        """
        Returns Delaunay triangulation of the coil's inner and outer surfaces in 3D.
        """
        # Flatten inner and outer surface points
        points = np.column_stack(
            (
                np.concatenate([self.X_inner.flatten(), self.X_outer.flatten()]),
                np.concatenate([self.Y_inner.flatten(), self.Y_outer.flatten()]),
                np.concatenate([self.Z_inner.flatten(), self.Z_outer.flatten()]),
            )
        )
        return Delaunay(points)


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

    center_offset = 0.5 * toroid_coil_config.radial_thickness
    R_center = R_inner + center_offset * N_R
    Z_center = Z_inner + center_offset * N_Z

    # Offset outer boundary
    coil_thickness = toroid_coil_config.radial_thickness
    R_outer = R_inner + coil_thickness * N_R
    Z_outer = Z_inner + coil_thickness * N_Z

    return ToroidalCoil2D(R_inner=R_inner, R_outer=R_outer, R_center=R_center, Z_inner=Z_inner, Z_center=Z_center, Z_outer=Z_outer)


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
) -> list[ToroidalCoil3D]:
    """
    Generate full 3D geometry for toroidal coils from 2D cross-section using efficient numpy operations.
    """
    phi_angles = np.linspace(0, 2 * np.pi, toroid_coil_config.n_field_coils, endpoint=False)
    coils = []

    # Prepare 2D cross-section
    r_inner_2d = toroidal_coil_2d.R_inner
    z_inner_2d = toroidal_coil_2d.Z_inner
    r_outer_2d = toroidal_coil_2d.R_outer
    z_outer_2d = toroidal_coil_2d.Z_outer

    phi_span = np.deg2rad(toroid_coil_config.angular_span)

    for phi_center in phi_angles:
        # Angular span for each coil
        phi_start = phi_center - phi_span / 2
        phi_end = phi_center + phi_span / 2

        # Toroidal sweep
        phi_sweep = np.linspace(phi_start, phi_end, COIL_RESOLUTION_3D)

        # Stack surfaces using broadcasting
        X_inner = np.outer(np.cos(phi_sweep), r_inner_2d)
        Y_inner = np.outer(np.sin(phi_sweep), r_inner_2d)
        Z_inner = np.tile(z_inner_2d, (COIL_RESOLUTION_3D, 1))

        X_outer = np.outer(np.cos(phi_sweep), r_outer_2d)
        Y_outer = np.outer(np.sin(phi_sweep), r_outer_2d)
        Z_outer = np.tile(z_outer_2d, (COIL_RESOLUTION_3D, 1))

        # Caps: first and last toroidal sweep
        X_cap_start = np.vstack([X_inner[0], X_outer[0]])
        Y_cap_start = np.vstack([Y_inner[0], Y_outer[0]])
        Z_cap_start = np.vstack([Z_inner[0], Z_outer[0]])

        X_cap_end = np.vstack([X_inner[-1], X_outer[-1]])
        Y_cap_end = np.vstack([Y_inner[-1], Y_outer[-1]])
        Z_cap_end = np.vstack([Z_inner[-1], Z_outer[-1]])

        # Center plane as X Y Z coordinates
        CentralPlane = np.column_stack((X_inner.mean(axis=0), Y_inner.mean(axis=0), Z_inner.mean(axis=0)))

        coil = ToroidalCoil3D(
            X_inner=X_inner,
            Y_inner=Y_inner,
            Z_inner=Z_inner,
            X_outer=X_outer,
            Y_outer=Y_outer,
            Z_outer=Z_outer,
            X_cap_start=X_cap_start,
            Y_cap_start=Y_cap_start,
            Z_cap_start=Z_cap_start,
            X_cap_end=X_cap_end,
            Y_cap_end=Y_cap_end,
            Z_cap_end=Z_cap_end,
            CentralPlane=CentralPlane,
            ToroidalCoil2D=toroidal_coil_2d,
        )
        coils.append(coil)

    return coils


if __name__ == "__main__":
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

    plasma_boundary, toroidal_coil_2d = calculate_2d_geometry(plasma_config=plasma_config, toroid_coil_config=toroid_coil_config)

    fusion_plasma = generate_fusion_plasma(plasma_boundary=plasma_boundary)
    toroidal_coils_3d = generate_toroidal_coils_3d(toroidal_coil_2d=toroidal_coil_2d, toroid_coil_config=toroid_coil_config)

    plotter = initialize_plotter(shape=(1, 2))

    plotter.subplot(0, 0)
    visualize_2d_geometry(plotter=plotter, plasma_boundary=plasma_boundary, toroidal_coil_2d=toroidal_coil_2d)
    plotter.subplot(0, 1)
    visualize_3d_geometry(
        plotter=plotter,
        fusion_plasma=fusion_plasma,
        toroidal_coils_3d=toroidal_coils_3d,
    )

    light = pv.Light(position=(5, 5, 5), focal_point=(0, 0, 0), intensity=0.2)
    plotter.add_light(light)

    # Set camera view
    # plotter.camera_position = "iso"

    # Add axes and bounds
    # plotter.add_axes = True

    # plotter.show_bounds(grid="front", location="outer", color="grey")

    # Render the visualization
    title_bar = "Fusion Reactor Visualization"
    plotter.show(title=title_bar, interactive=True)
