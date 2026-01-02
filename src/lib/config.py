"""Configuration module for this repository."""

from pathlib import Path
from typing import Iterator, Self

import jax
import jax.numpy as jnp
import numpy as np


class Filepaths:
    """
    Configuration class for file paths used in the application.
    """

    ROOT = Path.cwd()
    OUTPUT_DIR = Path(ROOT) / "data"
    PLASMA_SURFACE = OUTPUT_DIR / "plasma_surface.ply"
    TOROIDAL_COILS = OUTPUT_DIR / "toroidal_field_coils.ply"
    TOROIDAL_COIL_3D_DIR = OUTPUT_DIR / "toroidal_coils_3d"


class BaseModel:
    """Mixin providing type-safe iteration and serialization for Flax dataclasses."""

    def __iter__(self) -> Iterator[Self]:
        """Iterates over the batch dimension, yielding individual instances."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """Returns the size of the batch dimension (axis 0)."""
        # We look at the first field that is an array to determine 'length'
        for val in self.__dict__.values():
            if hasattr(val, "shape") and len(val.shape) > 0:
                return val.shape[0]
        return 0

    def __getitem__(self, idx: int) -> Self:
        """Returns a single instance sliced from the batch."""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for batch size {len(self)}")
        # Use tree_map to slice every array field in the dataclass simultaneously
        return jax.tree_util.tree_map(lambda x: x[idx], self)

    def to_dict(self) -> dict[str, any]:
        """
        Recursively converts the dataclass to a dictionary.
        Handles JAX/NumPy arrays by converting them to standard Python lists.
        """
        out: dict[str, any] = {}
        for k, v in self:
            if hasattr(v, "tolist"):  # Handle jnp.ndarray and np.ndarray
                out[k] = v.tolist()
            elif isinstance(v, BaseModel):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    def __repr__(self) -> str:
        """String representation."""
        cls_name = self.__class__.__name__
        lines: list[str] = []
        for k, v in self:
            if isinstance(v, (jnp.ndarray, np.ndarray)):
                lines.append(f"  {k}: {type(v).__name__}[shape={v.shape}, dtype={v.dtype}]")
            elif isinstance(v, (float, int, str, bool)):
                lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {k}: {type(v).__name__}")
        return f"{cls_name}(\n" + ",\n".join(lines) + "\n)"
