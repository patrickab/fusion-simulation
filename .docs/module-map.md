# Module Map

```
src/
├── api/                          # FastAPI service layer for the React frontend
│   ├── main.py                   # FastAPI app + routes, including model-stack metadata and artifact actions
│   ├── state.py                  # In-process NetworkManager cache (replaces st.session_state)
│   ├── network.py                # sample/flux/residual and server-traced field-line response builders
│   ├── geometry.py               # 2D boundary + 3D plasma/coil mesh builders (strided)
│   ├── benchmark.py              # SSE benchmark stream (one row event per checkpoint)
│   └── schemas.py                # Pydantic request models (GeometryRequest, SampleRequest, ...)
│
├── engine/
│   ├── network.py              # MLP/PirateNet FluxPINN (+ Fourier/RWF), Sampler, patience stopping, Trainer (optimizer/JIT state, train loop, L-BFGS), frozen FoundationModel and composed neural-corrector fields — no Rich/Plotext/filesystem imports
│   ├── network_manager.py      # NetworkManager facade: composes a Trainer with Rich/Plotext live display, run-dir/checkpoint/metrics.json persistence, and the training CLI (`python -m src.engine.network_manager`)
│   ├── residual_correction.py  # Corrector CLI + shared plain/nested/HPO checkpoint loading → composed NetworkManager. No parallel manager classes.
│   ├── physics.py              # GS operator, loss functions, B-field computation via AD
│   ├── plasma.py               # Parametric boundary → 3D FusionPlasma; point-in-plasma test
│   ├── model_evaluation.py     # Shared grids, Sobol residual KPIs, configurable Matplotlib montages
│   ├── benchmark_report.py    # Ranked LaTeX benchmark report rendered directly with pdflatex
│   ├── optimize_network_optuna.py  # Optuna HPO (primary, with validation-based pruning)
│   └── optimize-network-hparams.py # BoTorch HPO (legacy)
│
├── lib/
│   ├── geometry_config.py      # All dataclasses: coords, plasma/coil geometry, PlasmaConfig
│   ├── network_config.py       # HyperParams (architecture/RWF/training knobs), DomainBounds, FluxInput
│   ├── config.py               # Filepaths, global KPI protocol constants, git-commit identity, BaseModel
│   ├── run_artifacts.py         # Consolidated run.json/metrics.json serialization and projections
│   ├── visualization.py        # PyVista 3D + Plotly 2D render functions
│   ├── utils.py                # Coil → PolyData normalization, plasma → PolyData
│   ├── linalg_utils.py         # Rotation matrices, cylindrical→Cartesian helpers
│   └── logger.py               # Rich + file logger factory
│
├── streamlit/                  # Legacy UI (React frontend is now primary)
│   ├── app.py                  # Page registry and navigation shell
│   ├── utils.py                # Sidebar widgets, field-line geometry helpers, reseed
│   ├── network_utils.py        # Checkpoint discovery, commit extraction, Plotly layout, move_network_files
│   └── pages/
│       ├── reactor-visualisation.py    # Geometry builder: sliders → 2D/3D PyVista render
│       ├── network-visualisation.py    # PINN inspector: flux, residual, 3D field lines
│       └── benchmark-visualisation.py # Multi-checkpoint comparison across plasma configs
│
├── render.py                   # CLI: render plasma + coils to screen or PLY files
└── toroidal_geometry.py        # Coil cross-section offset, JIT kernel, generate_toroidal_coils_3d()

frontend/                       # React + Three.js SPA (active UI)
├── src/
│   ├── api.ts                  # Typed FastAPI client + useApi/useDebounced hooks + SSE benchmarkStream
│   ├── store.ts                # Zustand store (view, network, sidebar)
│   ├── App.tsx                 # View router
│   ├── main.tsx                # createRoot entry
│   ├── ui.tsx                  # Design-system primitives (Panel, Slider, Segmented, Colorbar, Popover, ...)
│   ├── plotly.ts               # Plotly cartesian partial bundle + baseLayout
│   ├── plots.tsx               # GridHeatmap, SampleScatter, CrossSection (boundary angle-reordered)
│   ├── shiki.ts                # JSON syntax highlighter (andromeeda, scientific notation)
│   ├── styles.css              # Dark-only neutral palette, hand-written CSS
│   ├── three/
│   │   ├── Scene.tsx           # r3f Canvas + OrbitControls (Z-up→Y-up group)
│   │   ├── Reactor3D.tsx       # PlasmaWireframe + FieldLines components
│   │   ├── fieldlines.ts       # Assembles server-traced polylines into one LineSegments buffer
│   │   ├── mesh.ts             # sparseWireframe for (n_phi×n_theta) structured grids
│   │   └── colormap.ts         # matplotlib "plasma" colormap + CSS gradient
│   └── views/
│       ├── ReactorView.tsx     # Geometry builder view
│       ├── NetworkView.tsx     # PINN inspector view
│       └── BenchmarkView.tsx  # SSE benchmark comparison view
├── package.json                # React 19, three, @react-three/fiber+drei, plotly, zustand, shiki, vite 8
├── vite.config.ts              # Dev proxy /api → 127.0.0.1:8010
└── tsconfig.json               # strict, ES2022, react-jsx

docs/
├── 01_geometry.md              # Toroidal geometry math (parametric, revolution, coords)
├── 02_pinn_engine.md           # GS equation, loss structure, BCs, profile modeling
├── 03_neural_architecture.md   # References for architecture choices
├── sources.md                  # Literature and external references
├── evaluation/                 # KPI-budget calibration and evaluation protocol evidence
├── hpo/                        # Current PirateNet foundation campaign plan
└── performance/                # One note per perf change with runnable benchmark and measured cost

AGENTS.md                       # Dev basics: uv-only, module entry points + --test smoke tests
                                # (HPO/tmux/benchmark conventions live in the vibe-hpo skill)
run-webapp.sh                   # Launches uvicorn (8010) + vite dev (5173) together
scripts/                        # KPI calibration/re-eval, legacy N6 preset, resumable PirateNet campaign
tests/                          # unittest coverage for patience stopping + numerical refactor fixture
```

## Artifact and evidence layout

```
data/
├── benchmarks/<timestamp>_<name>_<commit>/   # run.json, metrics.json, checkpoint and plots
├── benchmarks/model_selection_benchmark/     # Model-selection learnings and run log
├── benchmarks/_archive/<slug>/               # Supported archive location
├── hpo/<timestamp>_<name>_<commit>/          # Optuna study/campaign DB, ledgers and trial dirs
│   └── _archive/<study_slug>/                 # SQLite, trials.csv and retained trial runs
└── kpi_accuracy/                              # Raw KPI calibration runs
data_legacy/                                   # Ignored pre-consolidation artifact dump
```

## Key data contracts

| Type | Where defined | Role |
|---|---|---|
| `PlasmaGeometry` | geometry_config.py | (R₀, a, κ, δ) — shape params |
| `PlasmaState` | geometry_config.py | (p₀, F_axis, α, γ) — physics params |
| `PlasmaConfig` | geometry_config.py | Geometry + Boundary + State bundle |
| `PlasmaBoundary` | geometry_config.py | R-Z coords + gradients + center |
| `FusionPlasma` | geometry_config.py | 3D (256×256) Cartesian mesh |
| `ToroidalCoil3D` | geometry_config.py | Inner/outer/cap surface arrays |
| `FluxInput` | network_config.py | Batched (B, N) R-Z + config PyTree |
| `HyperParams` | network_config.py | All training hyperparameters |
| `EpochMetrics`/`ValidationMetrics`/`TrainingResult` | network_config.py | Typed Trainer.train() progress event and return value — the observer seam network_manager.py subscribes to |
| `FluxPINN` | network.py | Checkpoint-compatible MLP or PirateNet architecture, optionally Fourier-encoded/RWF |
| `Trainer` | network.py | Model/sampler/optimizer/JIT state + training loop; reports via an optional `TrainingObserver` callback, no Rich/filesystem imports |
| `NetworkManager` | network_manager.py | Training/inference/artifact facade composing a Trainer; accepts an optional `FoundationModel` prior for correction |
| `FoundationModel` | network.py | Frozen dataclass: converged FluxPINN + params; used as prior for neural correction |
| HPO configs | optimize_network_optuna.py | `SearchSpaceConfig` owns model axes; `StudyConfig` owns orchestration and checkpoint policy |
| Pydantic req models | `src/api/schemas.py` | Geometry, sampling, grids, field lines, artifact actions and benchmark requests |
| API response interfaces | `frontend/src/api.ts` | Samples, grids, model stack, geometry, field lines and benchmark SSE contracts |
