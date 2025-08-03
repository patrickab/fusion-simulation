"""Command-line interface for rendering fusion plasma visualizations."""

import argparse

from src.lib.visualization import render_fusion_plasma

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render fusion plasma surface.")
    parser.add_argument(
        "--show-angles",
        action="store_true",
        help="Display cylindrical coordinate angles in the plot.",
    )
    args = parser.parse_args()

    render_fusion_plasma(show_cylindrical_angles=args.show_angles)
