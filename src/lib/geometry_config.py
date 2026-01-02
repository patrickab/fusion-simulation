import json
import os

from flax.struct import dataclass
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay

from src.lib.config import Filepaths

COIL_RESOLUTION_3D = 64  # Number of points in the toroidal direction for 3D coils

@dataclass
class RotationalAngles:
    """Define toroidal and poloidal rotational angle arrays."""

    n_phi = 360  # Number of points in toroidal direction
    n_theta = 360  # Number of points in poloidal direction

    PHI = jnp.linspace(0, 2 * jnp.pi, n_phi)  # Azimuthal angle in radians
    THETA = jnp.linspace(0, 2 * jnp.pi, n_theta)  # Polar angle in radians

@dataclass
class PlasmaConfig:
    """Define geometric parameters for the plasma torus shape."""

    R0: float  # Major radius of the torus (m)
    a: float  # Minor radius of the torus (m)
    kappa: float  # Elongation factor
    delta: float  # Triangularity factor

@dataclass
class ToroidalCoilConfig:
    """Define geometric and positional parameters for toroidal field coils."""

    distance_from_plasma: float  # Plasma Boundary <-> Field Coil Center (m)
    radial_thickness: float  # radial thickness of the coil (m)
    vertical_thickness: float  # vertical thickness of the coil (m)
    angular_span: float  # angular span of the coil (degrees) - defines the coil's extent in the toroidal direction
    n_field_coils: int  # of field coils

@dataclass
class PlasmaBoundary:
    """Store poloidal plasma boundary coordinates in R-Z plane."""

    R_2d: jnp.ndarray  # R coordinates (m)
    Z_2d: jnp.ndarray  # Z coordinates (m)

@dataclass
class FusionPlasma:
    """Store 3D coordinates and boundary of the toroidal plasma surface."""

    X: jnp.ndarray  # X coordinates (m)
    Y: jnp.ndarray  # Y coordinates (m)
    Z: jnp.ndarray  # Z coordinates (m)
    Boundary: PlasmaBoundary

    def to_ply_structuregrid(self, filename: str = Filepaths.PLASMA_SURFACE) -> None:
        """Export plasma surface to PLY file.

        Args:
            filename: Target file path.

        Note: Writes to filesystem.
        """
        grid = pv.StructuredGrid(self.X, self.Y, self.Z).extract_surface()
        grid.save(filename)
        print(f"✅ Exported plasma surface to: {os.path.abspath(filename)}")

@dataclass
class ToroidalCoil2D:
    """Represent 2D toroidal field coil in the poloidal R-Z plane."""

    R_inner: jnp.ndarray
    R_outer: jnp.ndarray
    R_center: jnp.ndarray

    Z_inner: jnp.ndarray
    Z_outer: jnp.ndarray
    Z_center: jnp.ndarray

    def express_delaunay_triangles(self) -> Delaunay:
        """Return Delaunay triangulation of R-Z boundary points."""
        # Stack inner and outer boundary points
        points = np.column_stack((np.concatenate([self.R_inner, self.R_outer]), np.concatenate([self.Z_inner, self.Z_outer])))
        return Delaunay(points)

@dataclass
class ToroidalCoil3D:
    """Represent 3D toroidal field coil geometry for visualization."""

    X_inner: jnp.ndarray  # (n_phi, COIL_RESOLUTION_3D) inner wall swept
    Y_inner: jnp.ndarray
    Z_inner: jnp.ndarray

    X_outer: jnp.ndarray  # (n_phi, COIL_RESOLUTION_3D) outer wall swept
    Y_outer: jnp.ndarray
    Z_outer: jnp.ndarray

    X_cap_start: jnp.ndarray  # (2, n_pts) cap at start of toroidal span
    Y_cap_start: jnp.ndarray
    Z_cap_start: jnp.ndarray

    X_cap_end: jnp.ndarray  # (2, n_pts) cap at end of toroidal span
    Y_cap_end: jnp.ndarray
    Z_cap_end: jnp.ndarray

    CentralPlane: jnp.ndarray

    ToroidalCoil2D: ToroidalCoil2D  # Original 2D definition (for regeneration/reuse)

    def express_delaunay_triangles(self) -> Delaunay:
        """Return Delaunay triangulation of 3D coil surface points."""
        # Flatten inner and outer surface points
        points = np.column_stack(
            (
                np.concatenate([self.X_inner.flatten(), self.X_outer.flatten()]),
                np.concatenate([self.Y_inner.flatten(), self.Y_outer.flatten()]),
                np.concatenate([self.Z_inner.flatten(), self.Z_outer.flatten()]),
            )
        )
        return Delaunay(points)

    def to_ply(self, base_path: str | os.PathLike, coil_id: int) -> None:
        """Export coil components and metadata to PLY and JSON.

        Args:
            base_path: Output directory path.
            coil_id: Unique integer identifier.

        Note: Creates directory and multiple files.
        """
        base_path = os.fspath(base_path)
        os.makedirs(base_path, exist_ok=True)

        def _save_structured(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray, suffix: str) -> str:
            grid = pv.StructuredGrid(x, y, z).extract_surface()
            filename = f"coil_{coil_id:02d}_{suffix}.ply"
            path = os.path.join(base_path, filename)
            grid.save(path)
            return filename

        inner_file = _save_structured(self.X_inner, self.Y_inner, self.Z_inner, "inner")
        outer_file = _save_structured(self.X_outer, self.Y_outer, self.Z_outer, "outer")
        cap_start_file = _save_structured(self.X_cap_start, self.Y_cap_start, self.Z_cap_start, "cap_start")
        cap_end_file = _save_structured(self.X_cap_end, self.Y_cap_end, self.Z_cap_end, "cap_end")

        meta = {
            "inner": inner_file,
            "outer": outer_file,
            "cap_start": cap_start_file,
            "cap_end": cap_end_file,
        }
        meta_path = os.path.join(base_path, f"coil_{coil_id:02d}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
