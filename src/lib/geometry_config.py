from dataclasses import dataclass
import os

import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay

from src.lib.config import Filepaths


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
