"""Command-line interface for rendering fusion plasma visualizations."""

import argparse
from pathlib import Path

from src.lib.config import Filepaths
from src.lib.visualization import render_fusion_plasma

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render fusion plasma surface.")

    parser.add_argument(
        "--fusion-plasma-filepath",
        type=str,
        help="Path to the fusion plasma data file.",
        default=Filepaths.PLASMA_SURFACE,
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
    parser.add_argument(
        "--coils",
        nargs="?",
        const=Filepaths.TOROIDAL_COIL_3D_DIR,
        default=None,
        help="Display toroidal coils. Provide an optional directory path; otherwise, uses default.",
    )

    args = parser.parse_args()

    toroidal_coils = None
    if args.coils is not None:
        coil_dir = Path(args.coils)

        if not coil_dir.exists():
            raise FileNotFoundError(f"Coil directory does not exist: {coil_dir}")

        toroidal_coils = [f for f in coil_dir.iterdir() if f.is_file() and f.suffix == ".ply"]

    render_fusion_plasma(
        fusion_plasma=args.fusion_plasma_filepath,
        toroidal_coils=toroidal_coils,
        show_cylindrical_angles=args.show_angles,
        show_wireframe=args.show_wireframe,
    )
