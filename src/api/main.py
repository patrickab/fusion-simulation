"""FastAPI layer wrapping the existing src/ physics + network code for the React frontend."""

import contextlib
import json
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api import benchmark, geometry, network, state
from src.api.schemas import (
    BenchmarkRequest,
    FieldLinesRequest,
    GeometryRequest,
    GridRequest,
    RenameRequest,
    SampleRequest,
)
from src.lib.config import Filepaths
from src.lib.geometry_config import ToroidalCoilConfig
from src.streamlit.network_utils import get_available_networks, move_run_dir, resolve_run_directory

app = FastAPI(title="fusion-simulation API")

app.add_middleware(GZipMiddleware, minimum_size=1024)
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


@app.get("/api/config")
def get_config() -> dict:
    """Frontend-facing constants so Python and TS share one source of truth."""
    from src.engine.model_evaluation import EVAL_CONFIG_COUNT, EVAL_RESOLUTION

    return {
        "eval_config_count": EVAL_CONFIG_COUNT,
        "eval_resolution": EVAL_RESOLUTION,
    }


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
    try:
        run_dir = resolve_run_directory(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise HTTPException(404, f"No config found for {name}")
    return json.loads(config_path.read_text())


@app.get("/api/network/{name:path}/kpis")
def network_kpis(name: str) -> dict:
    """Stored post-training KPIs (kpis.json) — never recomputed at request time."""
    try:
        run_dir = resolve_run_directory(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None
    kpis_path = run_dir / "kpis.json"
    if not kpis_path.exists():
        raise HTTPException(404, f"No KPIs stored for {name}")
    return json.loads(kpis_path.read_text())


@app.post("/api/network/{name:path}/archive")
def archive_network(name: str) -> dict:
    try:
        run_dir = resolve_run_directory(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None
    commit = run_dir.parent.name
    target = Filepaths.BENCHMARK_ARCHIVE / commit / run_dir.name
    move_run_dir(name, target)
    state.invalidate(name)
    return {"ok": True}


@app.post("/api/network/{name:path}/rename")
def rename_network(name: str, body: RenameRequest) -> dict:
    try:
        run_dir = resolve_run_directory(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None
    new_path = run_dir.parent / body.new_name
    move_run_dir(name, new_path)
    state.invalidate(name)
    return {"name": f"{new_path.parent.name}/{new_path.name}"}


@app.delete("/api/network/{name:path}")
def delete_network(name: str) -> dict:
    try:
        run_dir = resolve_run_directory(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None
    shutil.rmtree(run_dir, ignore_errors=True)
    with contextlib.suppress(OSError):
        run_dir.parent.rmdir()
    state.invalidate(name)
    return {"ok": True}


@app.post("/api/network/{name:path}/sample")
def network_sample(name: str, body: SampleRequest) -> dict:
    manager = state.get_manager(name)
    return network.build_sample_response(manager, body.seed, body.sample_size)


@app.post("/api/network/{name:path}/flux")
def network_flux(name: str, body: GridRequest) -> list[dict]:
    manager = state.get_manager(name)
    return network.build_flux_grids(manager, body.seed, body.sample_size, body.resolution)


@app.post("/api/network/{name:path}/residual")
def network_residual(name: str, body: GridRequest) -> list[dict]:
    manager = state.get_manager(name)
    return network.build_residual_grids(manager, body.seed, body.sample_size, body.resolution)


@app.post("/api/network/{name:path}/fieldlines")
def network_fieldlines(name: str, body: FieldLinesRequest) -> dict:
    manager = state.get_manager(name)
    return network.build_field_lines(manager, body.seed, body.sample_size, body.n_lines)


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
        ),
        media_type="text/event-stream",
    )
