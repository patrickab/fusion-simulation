from pydantic import BaseModel


class CoilConfigIn(BaseModel):
    distance_from_plasma: float = 1.5
    radial_thickness: float = 0.8
    vertical_thickness: float = 0.2
    angular_span: float = 6.0
    n_field_coils: int = 8


class GeometryRequest(BaseModel):
    R0: float
    a: float
    kappa: float
    delta: float
    view_mode: str = "Both"  # "2D Geometry" | "3D Geometry" | "Both"
    show_coils: bool = True
    coil: CoilConfigIn = CoilConfigIn()
    mesh_stride: int = 3  # downsample factor for 3D mesh transmission


class SampleRequest(BaseModel):
    seed: int = 0
    sample_size: int = 4


class BFieldRequest(BaseModel):
    seed: int = 0
    sample_size: int = 4
    n_lines: int = 24


class GridRequest(BaseModel):
    seed: int = 0
    sample_size: int = 4
    resolution: int = 100


class RenameRequest(BaseModel):
    new_name: str


class BenchmarkRequest(BaseModel):
    networks: list[str]
    commit: str | None = None
    mode: str = "Both"  # "Flux Prediction" | "GS Residual" | "Both"
    seed: int = 0
    sample_size: int = 4
    resolution: int = 80
