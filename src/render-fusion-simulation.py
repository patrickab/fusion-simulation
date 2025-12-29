"""Command-line interface for rendering fusion plasma visualizations."""

import argparse

from src.lib.config import Filepaths
from src.lib.visualization import render_fusion_plasma

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render fusion plasma surface.")
    parser.add_argument(
        "--fusion-plasma-file", type=str, help="Path to the fusion plasma data file.", default=Filepaths.PLASMA_SURFACE
    )
    parser.add_argument(
        "--show-angles",
        action="store_true",
        help="Display cylindrical coordinate angles in the plot.",
    )
    parser.add_argument(
        "--show-wireframe",
        action="store_true",
        help="Display a sparse wireframe overlay of the plasma surface.",
    )
    args = parser.parse_args()

    render_fusion_plasma(
        plasma_file_path=args.fusion_plasma_file, show_cylindrical_angles=args.show_angles, show_wireframe=args.show_wireframe
    )
