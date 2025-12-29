"""Configuration module for this repository."""

from pathlib import Path


class Filepaths:
    """
    Configuration class for file paths used in the application.
    """

    ROOT = Path.cwd()
    OUTPUT_DIR = Path(ROOT) / "data"
    PLASMA_SURFACE = OUTPUT_DIR / "plasma_surface.ply"
    TOROIDAL_COILS = OUTPUT_DIR / "toroidal_field_coils.ply"