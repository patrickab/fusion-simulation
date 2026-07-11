"""FastAPI layer wrapping the existing src/ physics + network code for the React frontend."""

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api import benchmark, geometry, network, state
from src.api.schemas import (
    BenchmarkRequest,
    BFieldRequest,
    GeometryRequest,
    GridRequest,
    RenameRequest,
    SampleRequest,
)
from src.lib.config import Filepaths
from src.lib.geometry_config import ToroidalCoilConfig
from src.streamlit.network_utils import get_available_networks, move_network_files

app = FastAPI(title="fusion-simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/networks")
def list_networks(view_mode: str = "New Benchmarks") -> list[str]:
    return get_available_networks(view_mode)


@app.get("/api/network/{name:path}/config")
def network_config(name: str) -> dict:
    config_path = (Filepaths.NETWORKS / name).with_suffix(".json")
    if not config_path.exists():
        raise HTTPException(404, f"No config found for {name}")
    return json.loads(config_path.read_text())


@app.post("/api/network/{name:path}/archive")
def archive_network(name: str) -> dict:
    Filepaths.NETWORK_ARCHIVE.mkdir(parents=True, exist_ok=True)
    new_path_stem = Filepaths.NETWORK_ARCHIVE / (Filepaths.NETWORKS / name).stem
    move_network_files(name, new_path_stem)
    state.invalidate(name)
    return {"ok": True}


@app.post("/api/network/{name:path}/rename")
def rename_network(name: str, body: RenameRequest) -> dict:
    old_path = Filepaths.NETWORKS / name
    new_path_stem = old_path.parent / body.new_name
    move_network_files(name, new_path_stem)
    state.invalidate(name)
    return {"name": str(new_path_stem.with_suffix(".flax").relative_to(Filepaths.NETWORKS))}


@app.delete("/api/network/{name:path}")
def delete_network(name: str) -> dict:
    target_path = Filepaths.NETWORKS / name
    target_path.unlink(missing_ok=True)
    target_path.with_suffix(".json").unlink(missing_ok=True)
    state.invalidate(name)
    return {"ok": True}


@app.post("/api/network/{name:path}/sample")
def network_sample(name: str, body: SampleRequest) -> dict:
    manager = state.get_manager(name)
    return network.build_sample_response(manager, body.seed, body.sample_size, body.kpi_sample_size)


@app.post("/api/network/{name:path}/flux")
def network_flux(name: str, body: GridRequest) -> list[dict]:
    manager = state.get_manager(name)
    return network.build_flux_grids(manager, body.seed, body.sample_size, body.resolution)


@app.post("/api/network/{name:path}/residual")
def network_residual(name: str, body: GridRequest) -> list[dict]:
    manager = state.get_manager(name)
    return network.build_residual_grids(manager, body.seed, body.sample_size, body.resolution)


@app.post("/api/network/{name:path}/bfield")
def network_bfield(name: str, body: BFieldRequest) -> dict:
    manager = state.get_manager(name)
    return network.build_bfield_grid(manager, body.seed, body.sample_size, body.n_lines)


@app.post("/api/geometry")
def reactor_geometry(body: GeometryRequest) -> dict:
    coil_cfg = ToroidalCoilConfig(**body.coil.model_dump())
    return geometry.build_geometry_response(
        R0=body.R0,
        a=body.a,
        kappa=body.kappa,
        delta=body.delta,
        show_coils=body.show_coils,
        coil_cfg=coil_cfg,
        mesh_stride=body.mesh_stride,
    )


@app.post("/api/benchmark")
def run_benchmark(body: BenchmarkRequest) -> StreamingResponse:
    return StreamingResponse(
        benchmark.run_benchmark(
            body.networks,
            body.commit,
            body.mode,
            body.seed,
            body.sample_size,
            body.resolution,
            body.kpi_sample_size,
        ),
        media_type="text/event-stream",
    )
