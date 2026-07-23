from typing import Self

from pydantic import BaseModel, Field, model_validator

from src.engine.model_evaluation import EVAL_RESOLUTION


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


class StellaratorGeometryRequest(BaseModel):
    R0: float = Field(gt=0.0)
    a: float = Field(gt=0.0)
    kappa: float = Field(ge=0.5, le=3.0)
    n_field_periods: int = Field(ge=2, le=10)
    helical_amplitude: float = Field(ge=0.0, le=0.45)
    mesh_stride: int = Field(default=2, ge=1, le=16)

    @model_validator(mode="after")
    def surface_stays_clear_of_axis(self) -> Self:
        if self.a * (1.0 + self.helical_amplitude) >= self.R0:
            raise ValueError("R0 must exceed the maximum radial boundary excursion")
        return self


class SampleRequest(BaseModel):
    seed: int = 0
    sample_size: int = 4


class FieldLinesRequest(BaseModel):
    seed: int = 0
    sample_size: int = 4
    n_lines: int = 24


class GridRequest(BaseModel):
    seed: int = 0
    sample_size: int = 4
    resolution: int = Field(default=100, ge=4, le=EVAL_RESOLUTION)


class RenameRequest(BaseModel):
    new_name: str


class BenchmarkRequest(BaseModel):
    networks: list[str]
    commit: str | None = None
    mode: str = "Both"  # "Flux Prediction" | "GS Residual" | "Both"
    seed: int = 0
    sample_size: int = 4
    resolution: int = Field(default=100, ge=4, le=EVAL_RESOLUTION)
