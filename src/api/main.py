"""FastAPI layer wrapping the existing src/ physics + network code for the React frontend."""

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
from src.streamlit.network_utils import (
    get_available_networks,
    get_hpo_studies,
    hpo_network_name,
    is_hpo_name,
    move_run_dir,
    move_study_dir,
    renamed_slug,
    resolve_run_directory,
    resolve_study_directory,
    split_hpo_name,
)

app = FastAPI(title="fusion-simulation API")

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


Filepaths.BENCHMARKS.mkdir(parents=True, exist_ok=True)
Filepaths.HPO_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/api/benchmarks/files", StaticFiles(directory=Filepaths.BENCHMARKS), name="benchmarks")


@app.get("/api/networks")
def list_networks(view_mode: str = "New Benchmarks") -> list[str]:
    return get_available_networks(view_mode)


@app.get("/api/hpo")
def list_hpo_studies(archived: bool = False) -> dict[str, list[str]]:
    """List HPO study slugs with their direct trial-network slugs."""
    return get_hpo_studies(archived)


@app.get("/api/config")
def get_config() -> dict:
    """Frontend-facing constants so Python and TS share one source of truth."""
    from src.engine.model_evaluation import (
        EVAL_RESOLUTION,
        N_PLOTS,
        RESIDUAL_COLOR_RANGE,
    )

    return {
        "eval_config_count": N_PLOTS,
        "eval_resolution": EVAL_RESOLUTION,
        "residual_color_range": list(RESIDUAL_COLOR_RANGE),
    }


@app.get("/api/benchmarks")
def list_benchmarks() -> dict[str, dict[str, list[str]]]:
    """Flattened benchmark tree grouped by the commit encoded in each slug."""
    from src.streamlit.network_utils import parse_slug

    tree: dict[str, dict[str, list[str]]] = {}
    runs = sorted(p for p in Filepaths.BENCHMARKS.iterdir() if p.is_dir() and p.name != "_archive")
    for run in runs:
        try:
            _, _, commit = parse_slug(run.name)
        except ValueError:
            continue
        tree.setdefault(commit, {})[run.name] = sorted(f.name for f in run.iterdir() if f.is_file())
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
    # Prefer the composed-field KPIs when a corrector exists, so the reported
    # metrics match the field get_manager() renders for this run.
    stage2_kpis = run_dir / "stage2" / "kpis.json"
    kpis_path = stage2_kpis if stage2_kpis.exists() else run_dir / "kpis.json"
    if not kpis_path.exists():
        raise HTTPException(404, f"No KPIs stored for {name}")
    return json.loads(kpis_path.read_text())


@app.post("/api/network/{name:path}/archive")
def archive_network(name: str) -> dict:
    if is_hpo_name(name):
        try:
            study, _ = split_hpo_name(name)
            move_study_dir(study, Filepaths.HPO_ARCHIVE / study)
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from None
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, f"HPO study not found: {name}") from None
        state.invalidate_prefix(f"hpo/{study}")
        return {"ok": True}
    try:
        run_dir = resolve_run_directory(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None
    target = Filepaths.BENCHMARK_ARCHIVE / run_dir.name
    try:
        move_run_dir(name, target)
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from None
    state.invalidate(name)
    return {"ok": True}


@app.post("/api/network/{name:path}/rename")
def rename_network(name: str, body: RenameRequest) -> dict:
    if is_hpo_name(name):
        try:
            study, run = split_hpo_name(name)
            new_study = renamed_slug(study, body.new_name)
            move_study_dir(study, resolve_study_directory(study).parent / new_study)
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except FileNotFoundError:
            raise HTTPException(404, f"HPO study not found: {name}") from None
        state.invalidate_prefix(f"hpo/{study}")
        return {"name": hpo_network_name(new_study, run) if run else f"hpo/{new_study}"}
    try:
        run_dir = resolve_run_directory(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None
    try:
        new_path = run_dir.parent / renamed_slug(run_dir.name, body.new_name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    try:
        move_run_dir(name, new_path)
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from None
    state.invalidate(name)
    return {"name": new_path.name}


@app.delete("/api/network/{name:path}")
def delete_network(name: str) -> dict:
    if is_hpo_name(name):
        try:
            study, run = split_hpo_name(name)
            target = resolve_study_directory(study) if run is None else resolve_run_directory(name)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, f"HPO target not found: {name}") from None
        shutil.rmtree(target, ignore_errors=True)
        if run is None:
            state.invalidate_prefix(f"hpo/{study}")
        else:
            state.invalidate(name)
        return {"ok": True}
    try:
        run_dir = resolve_run_directory(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Run dir not found: {name}") from None
    shutil.rmtree(run_dir, ignore_errors=True)
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
