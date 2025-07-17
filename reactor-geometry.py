"""Module for generation and visualization of a tokamak plasma surface"""

import os
import numpy as np
import pyvista as pv

def tokamak_cross_section(theta, r0=2.0, a=0.5, kappa=1.7, delta=0.3):
    """
    Returns R and Z coordinates for a 2D poloidal plasma boundary shape.
    Cross-section of a tokamak plasma in the poloidal plane.

    Parameters
    ----------
    - theta: array of angle(s) in radians
    - r0 : Major radius of the torus
    - a : Minor radius of the torus
    - kappa: elongation factor
    - delta: triangularity factor

    Returns
    -------
    - R: radial coordinate (m)
    - Z: vertical coordinate (m)
    """
    R = r0 + a * np.cos(theta + delta * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    return R, Z

def generate_toroidal_surface(n_theta: int=200, n_phi: int=100, r0: float=1.5, a: float=0.5, kappa: float=1.7, delta: float=0.3):
    """
    Generates a 3D toroidal surface by rotating a poloidal cross-section around the Z-axis.
    
    This function creates a tokamak-like surface with elongation (kappa) and triangularity (delta).
    The process works as follows:
    
    1. Generate poloidal coordinates
    2. Create a meshgrid that extends this 2D shape into 3D space by:
       - np.meshgrid(R_2D, phi) creates two 2D arrays:
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
    
    Parameters
    ----------
    n_theta : Number of points (poloidal direction)
    n_phi : Number of points (toroidal direction)
    r0 : Major radius of the torus
    a : Minor radius of the torus
    kappa : Elongation factor (vertical stretching of the cross-section)
    delta : Triangularity factor (controls the D-shape of the cross-section)

    Returns
    -------
    X, Y, Z : 3D Cartesian coordinates of the toroidal surface, each with shape (n_phi, n_theta)
    """
    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)

    R_2D, Z_2D = tokamak_cross_section(theta, r0, a, kappa, delta)

    # Create 2D meshgrid for revolution
    R_grid, phi_grid = np.meshgrid(R_2D, phi)
    Z_grid = np.tile(Z_2D, (n_phi, 1))

    # Convert cylindrical (R, φ, Z) → Cartesian (X, Y, Z)
    X = R_grid * np.cos(phi_grid)
    Y = R_grid * np.sin(phi_grid)
    Z = Z_grid

    return X, Y, Z

def export_polygonal_plasmasurface(filename='plasma_surface.ply'):
    """Converts the toroidal plasma surface to a polygonal mesh & stores it as .ply"""
    X, Y, Z = generate_toroidal_surface()
    grid = pv.StructuredGrid(X, Y, Z).extract_surface()
    grid.save(filename)
    print(f"✅ Exported plasma surface to: {os.path.abspath(filename)}")


if __name__ == "__main__":
    export_polygonal_plasmasurface()
