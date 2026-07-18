# Conventions

## Language & Runtime
- Python 3.11+ (requires-python = ">=3.11", ruff target py312)
- JAX as primary compute framework (GPU/CUDA12); NumPy used for non-JAX paths
- Flax (struct.dataclass) for neural network definition and parameter pytrees
- TypeScript ~5.9 (strict, ES2022) for the frontend; Vite 8 bundler

## Key Dependencies — Python
| Package | Role |
|---|---|
| `jax[cuda12]` | Differentiable compute, vmap, jit, AD |
| `flax` | Neural network layers, struct.dataclass |
| `optax` | Optimizers (AdamW), schedules, huber loss |
| `fastapi` | HTTP API layer (imported but not in pyproject deps; transitive) |
| `uvicorn` | ASGI server for the FastAPI app (launched via `run-webapp.sh`) |
| `pydantic` | Request/response schemas in `src/api/schemas.py` |
| `plotly` | 2D interactive heatmaps and subplots |
| `pyvista` + `stpyvista` | 3D mesh rendering in Streamlit (legacy UI) |
| `streamlit` | Legacy UI framework |
| `optuna` | Primary HPO |
| `textual` | HPO terminal UI (overview/detail toggle) |
| `botorch` + `torch` | Legacy HPO (BoTorch) |
| `scipy` | Sobol sequences (qmc) |
| `ruff` | Linting + formatting (Black-compatible) |

## Key Dependencies — Frontend (`frontend/package.json`)
| Package | Role |
|---|---|
| `react` 19 + `react-dom` | UI runtime |
| `three` + `@react-three/fiber` + `@react-three/drei` | WebGL 3D (react-three-fiber) |
| `plotly.js` + `react-plotly.js` | Custom partial bundle for scatter, carpet and contourcarpet traces |
| `zustand` | Client state store |
| `shiki` | JSON syntax highlighting |
| `vite` 8 + `@vitejs/plugin-react` | Dev server + build |
| `oxlint` | Frontend linter |
| `typescript` 5.9 | Type checking (`tsc -b`) |

## Naming
- Python dataclasses use PascalCase fields for physical concepts (R0, Z_center, PlasmaGeometry)
- Python functions use snake_case; pure JAX functions designed for `jit`/`vmap`
- Streamlit pages use hyphen-separated filenames (e.g., `network-visualisation.py`)
- Frontend React components use PascalCase; TS modules use camelCase (`api.ts`, `fieldlines.ts`)
- FastAPI routes: `/api/<resource>/<name>/<action>` (e.g. `/api/network/{name}/fieldlines`)
- Single-config checkpoints: `data/benchmarks/<YYYY_MM_DD_HH_MM_SS>_<name>_<commit>/`; HPO studies use the same study-slug format and retain `pinn_<timestamp>` trial dirs inside.
- Checkpoint-compatible architecture toggles are lowercase serialized values: `arch="mlp"|"piratenet"`, `rwf=true|false`; defaults stay on the legacy parameter-tree shape.

## Linting / Formatting
- **Python**: Ruff with rule sets E, F, I (isort), B, C4, TCH, SIM, ANN, ARG, RUF. Line length 100. First-party: `src`. Run through uv: `uv run ruff check .` and `uv run ruff format --check .`.
- **Frontend**: `oxlint src` (lint) + `tsc -b` (typecheck). Vite build: `npm --prefix frontend run build`
- Run both before considering Python/frontend work done.

## JAX Patterns
- All JAX functions must be pure (stateless) for `jit` compatibility
- Use `jax.lax.stop_gradient()` for axis detection (prevents gradient feedback)
- `jax.checkpoint` (remat) on PDE residual functions to save VRAM
- Nested `jax.jvp` for 2nd-order coordinate derivatives; retain useful primal/first terms
- `jax.vmap` double-vectorized: outer over plasma configs, inner over spatial points

## Frontend Patterns
- `useApi(key, fn)` cached fetch keyed by request signature; `invalidate(prefix)` clears stale entries
- 3D scene is dark, unlit, Z-up→Y-up group; `OrbitControls` with damped idle drift
- Field-line tracing done server-side (VTK RK45 via PyVista over the 48³ B-grid); client only assembles shipped polylines into a `LineSegments` buffer
- Residual heatmap color scales are fixed across checkpoints; flux ranges remain data-driven
- Dark-only UI: neutral palette, cyan reserved for data/wireframes

## Commit Style
Conventional Commits with scope: `perf(physics):`, `feat(artifacts):`, `refactor(hpo):`, `docs:`, `fix(eval):`. Never commit/push/amend unless the user explicitly asks.

## Operational rules (AGENTS.md)
`uv run`/`uv sync` only, never bare `python`/`pip`; module entry points (`uv run python -m src.engine.network`). Long jobs (training/HPO) always in a tmux window named `claude: <name>`, never bare nohup. Optuna launches in tmux need `--reset-sqlite`/`--resume-sqlite` or they hang on an interactive prompt. Perf work documents each change in `docs/performance/` with a runnable benchmark snippet and measured train-step ms + XLA temp memory (`train_step.lower(...).compile().memory_analysis()`).

## Testing
- `tests/test_early_stopping.py` uses `unittest` for `_PatienceStopper`; run with `uv run python -m unittest discover -s tests`.
- `tests/fixtures/refactor_golden.json` preserves single/composed checkpoint KPI and point references for numerical refactor checks.
- Broader physics correctness is still validated by deterministic GS-residual KPIs and stored benchmark comparisons; frontend validation remains lint/typecheck/build plus visual inspection.
