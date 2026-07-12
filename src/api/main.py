"""FastAPI layer wrapping the existing src/ physics + network code for the React frontend."""

import contextlib
import json
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api import benchmark, geometry, network, state
from src.api.schemas import (
    BenchmarkRequest,
    BFieldRequest,
    GeometryRequest,
    GridRequest,
    RenameRequest,
    SampleRequest,
)
from src.api.state import resolve_run_directory
from src.lib.config import Filepaths
from src.lib.geometry_config import ToroidalCoilConfig
from src.streamlit.network_utils import get_available_networks

app = FastAPI(title="fusion-simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


Filepaths.BENCHMARKS.mkdir(parents=True, exist_ok=True)
app.mount("/api/benchmarks/files", StaticFiles(directory=Filepaths.BENCHMARKS), name="benchmarks")


@app.get("/api/networks")
def list_networks(view_mode: str = "New Benchmarks") -> list[str]:
    return get_available_networks(view_mode)


@app.get("/api/benchmarks")
def list_benchmarks() -> dict[str, dict[str, list[str]]]:
    """data/benchmarks tree: {commit: {run: [file, ...]}}."""
    tree: dict[str, dict[str, list[str]]] = {}
    for commit_dir in sorted(p for p in Filepaths.BENCHMARKS.iterdir() if p.is_dir()):
        runs: dict[str, list[str]] = {}
        for run in sorted(p for p in commit_dir.iterdir() if p.is_dir()):
            runs[run.name] = sorted(f.name for f in run.iterdir() if f.is_file())
        if runs:
            tree[commit_dir.name] = runs
    return tree


@app.get("/api/network/{name:path}/config")
def network_config(name: str) -> dict:
    run_dir = resolve_run_directory(name)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise HTTPException(404, f"No config found for {name}")
    return json.loads(config_path.read_text())


@app.post("/api/network/{name:path}/archive")
def archive_network(name: str) -> dict:
    run_dir = resolve_run_directory(name)
    commit = run_dir.parent.name
    target = Filepaths.BENCHMARK_ARCHIVE / commit / run_dir.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))
    with contextlib.suppress(OSError):
        run_dir.parent.rmdir()
    state.invalidate(name)
    return {"ok": True}


@app.post("/api/network/{name:path}/rename")
def rename_network(name: str, body: RenameRequest) -> dict:
    run_dir = resolve_run_directory(name)
    new_path = run_dir.parent / body.new_name
    shutil.move(str(run_dir), str(new_path))
    state.invalidate(name)
    return {"name": f"{new_path.parent.name}/{new_path.name}"}


@app.delete("/api/network/{name:path}")
def delete_network(name: str) -> dict:
    run_dir = resolve_run_directory(name)
    shutil.rmtree(run_dir, ignore_errors=True)
    with contextlib.suppress(OSError):
        run_dir.parent.rmdir()
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
