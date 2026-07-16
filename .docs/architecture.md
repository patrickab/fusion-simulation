# Architecture

## Overview
Physics-Informed Neural Network (PINN) that solves the Grad–Shafranov (GS) equation for tokamak plasma equilibrium. The network learns the poloidal magnetic flux ψ(R,Z) as a universal function over geometry and plasma state space.

Two UI stacks now sit on top of the same physics/network core:
- **React + Three.js frontend** (`frontend/`) — the active/preferred UI, served by a FastAPI backend.
- **Streamlit app** (`src/streamlit/`) — the original UI, still present but superseded.

## Core Pipeline
```
PlasmaGeometry/PlasmaState params
        │
        ▼
calculate_poloidal_boundary()   → PlasmaBoundary (R-Z coords + derivatives)
        │
        ├──► calculate_fusion_plasma()         → FusionPlasma (3D Cartesian mesh)
        ├──► generate_toroidal_coils_3d()      → list[ToroidalCoil3D]
        └──► Sampler.sample_flux_input()       → FluxInput (batched R,Z,config)
                        │
                        ▼
               FluxPINN.apply()   →  ψ̂(R,Z) (normalized)
                        │
                        ├──► pinn_loss() = L_physics + λ·L_boundary
                        └──► get_b_field() = (B_R, B_Z, B_φ) via AD of ψ
```

## Backend Service Layer (`src/api/`) — FastAPI
Thin HTTP wrapper exposing the `src/engine` + `src/lib` core to the React frontend. Reuses Streamlit helpers (`get_available_networks`, `move_network_files`) for filesystem ops.
- `main.py` — `FastAPI` app + CORS; endpoints under `/api`:
   - `GET  /api/networks` — list flattened single-config checkpoints (view_mode filter)
   - `GET  /api/hpo` — HPO study slugs and their retained trial slugs (live or archived)
  - `GET  /api/network/{name}/{config|kpis}` — HyperParams JSON / stored post-training `kpis.json` (KPIs are never recomputed at request time)
  - `POST /api/network/{name}/{archive|rename|sample|flux|residual|fieldlines}`
  - `DELETE /api/network/{name}`
  - `POST /api/geometry` — 2D boundary + 3D plasma/coil meshes (downsampled via `mesh_stride`)
  - `POST /api/benchmark` — `StreamingResponse` (SSE) emitting one row event per checkpoint
  - `GET  /api/config` — frontend-facing constants (`eval_config_count`, `eval_resolution`) from `model_evaluation.py`
  - `GET  /api/benchmarks` — saved benchmark tree `{commit: {run: [files]}}`; artifacts served statically under `/api/benchmarks/files/` (mount of `data/benchmarks/`)
- `state.py` — in-process `dict[str, NetworkManager]` cache (replaces `st.session_state`); avoids JAX/JIT re-tracing per request.
- `network.py` — sample/flux/residual/fieldlines builders; serializes the shared plasma-aligned grids from `model_evaluation.py`. Field lines are traced **server-side** (`build_field_lines`): B on a 48³ `pv.ImageData` grid (vectors zeroed outside ρ̃>1.05), then VTK RK45 `streamlines_from_source`; response ships polylines decimated 4× for transport, integration exact. Response floats rounded for transport. Grid/bfield endpoints derive seeded configs via a bare domain Sobol stream + `fast_forward` (`_seeded_configs`) instead of a Sampler; only `/sample` builds the full Sampler (interior collocation points).
- `geometry.py` — 2D boundary + 3D `FusionPlasma`/`ToroidalCoil3D` meshes, strided for transport.
- `benchmark.py` — SSE row generator (`_sse_event`), filters networks by git commit.
- `schemas.py` — Pydantic request models (`GeometryRequest`, `SampleRequest`, `GridRequest`, `BFieldRequest`, `RenameRequest`, `BenchmarkRequest`, `CoilConfigIn`).

## React Frontend (`frontend/`) — TypeScript + Vite
Single-page app, three views mirroring the Streamlit pages:
1. **reactor** (`ReactorView.tsx`) — geometry sliders → 2D Plotly cross-section + 3D `PlasmaWireframe`.
2. **network** (`NetworkView.tsx`) — checkpoint picker; flux/residual heatmaps (Plotly), sample scatter, 3D field lines + plasma wireframe (Three.js), config JSON (Shiki-highlighted).
3. **benchmark** (`BenchmarkView.tsx`) — SSE-streamed multi-checkpoint comparison; per-network grid cards; saved-run browser over `GET /api/benchmarks` gated behind a selectbox — exactly one `StoredRun` mounts at a time, since mounting fires a backend residual eval.

Key modules:
- `api.ts` — typed fetch client; module-level `Map` cache (`invalidate()` clears by prefix); `useApi`/`useDebounced` hooks; `benchmarkStream` SSE consumer.
- `three/` — `Reactor3D.tsx` (`PlasmaWireframe`, `FieldLines`), `Scene.tsx` (r3f `Canvas` + `OrbitControls`), `fieldlines.ts` (assembles server-traced polylines into one merged `LineSegments` buffer, |B| vertex colors), `mesh.ts` (sparse wireframe), `colormap.ts` (matplotlib `plasma`).
- `plots.tsx` — Plotly `GridHeatmap`, `SampleScatter`, `CrossSection`; boundary points reordered by poloidal angle.
- `ui.tsx` — design-system primitives (`Panel`, `Section`, `Slider`, `Segmented`, `Toggle`, `Stat`, `Colorbar`, `Popover`, `Spinner`, `JsonBlock`).
- `store.ts` — Zustand store (`view`, `network`, `sidebar`).
- `shiki.ts` — JSON syntax highlighting (andromeeda theme, scientific-notation formatter).
- `styles.css` — dark-only, neutral palette with cyan data accent; hand-written, no CSS framework.

Vite dev server proxies `/api` → `127.0.0.1:8010` (matches `run-webapp.sh`). Both servers launched by `run-webapp.sh` (uvicorn `--reload` + `vite dev`).

## Physics Engine (`src/engine/physics.py`)
- **GS operator** Δ*ψ computed via nested `jax.jvp` forward-mode AD. The R pass returns primal/first/second terms together; the Z pass computes only its second derivative, without materializing a full Hessian.
- **Loss**: hard-BC default is `L_GS + w_flux·hinge(interior mean ψ)` (boundary terms diagnostic-only, gradient-stopped); legacy `soft_bc` adds `λ·(L_Dirichlet + L_Neumann)`.
- **Double vmap**: outer over plasma configs (batch), inner over spatial collocation points.
- **Remat** (`jax.checkpoint`): targeted on the GS residual graph only; the redundant whole-loss outer checkpoint was removed.
- **Axis detection**: `estimate_psi_axis` = `stop_gradient(mean(top-|ψ|))` per batch (sign-agnostic), fed by the primal ψ the operator already computes (no second network pass).
- **Cached boundary fit**: `PlasmaBoundary` stores the 32-harmonic Fourier ridge coefficients at construction; only basis evaluation stays in the differentiated path.
- **Perf trail**: `docs/performance/*.md` records each optimization with its runnable benchmark; train step went 42.7 → 28.4 ms (−33%, RTX 3060, direct-training defaults).

## Neural Network (`src/engine/network.py`)
- **FluxPINN**: MLP with Swish activations. Input: 10 normalized scalars (r, z, r₀, a, κ, δ, p₀, F_axis, α, γ). Output: scalar ψ̂.
- **Sampler**: Sobol quasi-random sequences for collocation points; 50% Sobol + 50% adaptive (focus on high-loss regions). Geometries resampled every 10 epochs.
- **NetworkManager**: wraps model init, training loop, I/O. Saves `network.flax` + `config.json` + `training.csv` + `train.log` per run dir (`data/benchmarks/<commit>/<run>/`), tagged with git commit hash (parent dir).
- **Optimizer**: AdamW with warmup cosine decay schedule.
- **Replay**: `uv run python -m src.engine.network --show <commit/run>` re-renders the Rich Training Metrics table for a stored run from its `training.csv` (accepts a dir path or bare `pinn_<timestamp>` too).

## Geometry (`src/lib/geometry_config.py`, `src/engine/plasma.py`, `src/toroidal_geometry.py`)
- Parametric plasma boundary: R(θ) = R₀ + a·cos(θ + δ·sin θ), Z(θ) = κ·a·sin θ
- 3D torus: poloidal cross-section revolved via `jnp.outer`, result is (n_phi=256, n_theta=256) structured mesh.
- Coils: normal-offset from plasma boundary, swept toroidally; inner/outer/caps per coil.
- Point-in-plasma: O(log N) via sorted angular interpolation of boundary radius.

## Visualization (`src/lib/visualization.py`)
Used by the Streamlit UI and referenced for fixed heatmap scales by the React frontend (zmin/zmax constants).
- **3D**: PyVista `StructuredGrid` + stpyvista for Streamlit embedding.
- **2D**: Plotly `make_subplots` for flux ψ heatmaps and GS residual heatmaps.
- **Field lines**: PyVista `streamlines_from_source` seeded along the midplane.

## Shared Evaluation Grids (`src/engine/model_evaluation.py`)
`evaluate_plasma_grids()` is the common data path for frontend Plotly fields and offline Matplotlib montages. `resolution` is the poloidal sample count and radial resolution is half as large; both renderers therefore evaluate identical plasma-aligned coordinates and display-ready flux/log-residual values. `evaluate_residual_samples()` is the batched, jitted KPI core: per-point `|R_GS|` of shape (n_configs, sample_size) on a deterministic, area-uniform sqrt-rho Sobol sample (16,384 points/config by default), vmapped over configs with a chunked inner point map and ψ computed once per point (reused for the axis estimate). `evaluate_plasma_kpis()` wraps it into whole-plasma/core distribution summaries, edge p95, and boundary leakage. The same KPI mapping is shown in both frontend benchmark views and `plot_plasma_grid_montage()` output.

Single-config benchmark artifacts live directly under `data/benchmarks/<timestamp>_<name>_<commit>/`; every run dir holds `network.flax`, `config.json`, `training.csv`, `train.log`, `kpis.json`, and plots. Invariant: **a run dir exists iff its checkpoint was kept**; `--test`, pruned, failed, aborted, and non-top-k runs delete their own dir on exit. Optuna studies live under `data/hpo/<timestamp>_<study_name>_<commit>/` with logs, ledgers, and direct `pinn_<ts>/` trial dirs. HPO trials resolve as `hpo/<study_slug>/<trial_slug>`. Archived single configs move to `data/benchmarks/_archive/<slug>/`; archiving an HPO study moves its complete directory to `data/hpo/_archive/<study_slug>/`. All data paths derive from `Filepaths.DATA`.

## HPO (`src/engine/optimize_network_optuna.py`, `src/engine/optimize-network-hparams.py`)
- **Optuna** (primary): thin TPE/pruning adapter over `NetworkManager.train`; `SearchSpaceConfig`
  owns search bounds and fixed study settings. Standard and HPO training share one epoch,
  validation, L-BFGS, and saving path. Validation uses a fixed dedicated sampler so it never
  advances training Sobol state. HPO runs in a Textual TUI (`HpoApp`): Tab toggles between the
  study-level Rich dashboard and a sequential per-trial log (one styled network.py
  Training Metrics table per finished trial under its header). Non-tty stdout
  (piped/nohup) automatically
  falls back to the plain Rich Live dashboard; all artifacts are identical either way.
  NetworkManager always tracks metrics/training_log now; Rich Live attaches only in CLI mode.
- **BoTorch** (legacy): Bayesian optimization via qMaxValueEntropy in normalized unit cube.

## Streamlit UI (`src/streamlit/`) — legacy
Three pages (still runnable; React frontend is the active replacement):
1. **reactor-visualisation**: Interactive geometry builder (2D R-Z + 3D PyVista).
2. **network-visualisation**: Load PINN checkpoint → flux heatmaps, GS residual, 3D field lines.
3. **benchmark-visualisation**: Compare multiple checkpoints across randomized plasma configs.
