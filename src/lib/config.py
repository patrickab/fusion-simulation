"""Configuration module for this repository."""

import numpy as np


class Filepaths:
    """
    Configuration class for file paths used in the application.
    """

    ROOT = "~/fusion-simulation"
    OUTPUT_DIR = "data"
    REACTOR_POLYGONIAL_MESH = "plasma_surface.ply"
    REACTOR_TOROIDAL_FIELD_COILS = "toroidal_field_coils.ply"


class RotationalAngles:
    """
    Configuration class for rotational angles used in the application.
    """

    n_phi = 360  # Number of points in toroidal direction
    n_theta = 360  # Number of points in poloidal direction

    PHI = np.linspace(0, 2 * np.pi, n_phi)  # Azimuthal angle in radians
    THETA = np.linspace(0, 2 * np.pi, n_theta)  # Polar angle in radians
