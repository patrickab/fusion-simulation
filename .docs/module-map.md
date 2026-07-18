# Module Map

```
src/
├── api/                          # FastAPI service layer for the React frontend
│   ├── main.py                   # FastAPI app + routes (/api/networks, /network/*, /geometry, /benchmark)
│   ├── state.py                  # In-process NetworkManager cache (replaces st.session_state)
│   ├── network.py                # sample/flux/residual/bfield response builders
│   ├── geometry.py               # 2D boundary + 3D plasma/coil mesh builders (strided)
│   ├── benchmark.py              # SSE benchmark stream (one row event per checkpoint)
│   └── schemas.py                # Pydantic request models (GeometryRequest, SampleRequest, ...)
│
├── engine/
│   ├── network.py              # FluxPINN model, Sampler; NetworkManager facade (train/infer/save/load) + FoundationModel (frozen prior for multistage correction); private collaborators: _Field (psi-fn, single or composed psi1+scale·psi2), _MetricsManager (Rich table/progress/training_log), _FileStorageManager (run dir + artifact I/O incl. nested stage2/ layout). Loss seam: compute_loss/train_step take a psi_fn. NetworkManager(config, prior=FoundationModel(...), scale=...) builds a corrector; for_inference classmethod for lean querying.
│   ├── residual_correction.py  # Corrector CLI + shared plain/nested/HPO checkpoint loading → composed NetworkManager. No parallel manager classes.
│   ├── physics.py              # GS operator, loss functions, B-field computation via AD
│   ├── plasma.py               # Parametric boundary → 3D FusionPlasma; point-in-plasma test
│   ├── model_evaluation.py     # Shared grids, Sobol residual KPIs, configurable Matplotlib montages
│   ├── benchmark_report.py    # LaTeX→PDF benchmark report generator (pandoc + pdflatex)
│   ├── optimize_network_optuna.py  # Optuna HPO (primary, with validation-based pruning)
│   └── optimize-network-hparams.py # BoTorch HPO (legacy)
│
├── lib/
│   ├── geometry_config.py      # All dataclasses: coords, plasma/coil geometry, PlasmaConfig
│   ├── network_config.py       # HyperParams, DomainBounds, FluxInput (JAX pytrees)
│   ├── config.py               # Filepaths (data/benchmarks, data/hpo), BaseModel mixin
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
├── training-process.md         # Prose walkthrough of the full training pipeline (setup→HPO)
└── performance/                # One note per perf commit: change + runnable benchmark + measured ms/MiB

AGENTS.md                       # Operational rules: uv run only, tmux windows for long jobs,
                                # training/HPO entry-point flags, --reset-sqlite semantics
model_selection_report/         # LaTeX report (R2 soft-BC baseline → N1/N3/N6 hard-BC) + its 4 checkpoints
todo.md / plot-error.md / latex-benchmark.md   # Untracked handoff notes (see current-focus.md)
run-webapp.sh                   # Launches uvicorn (8010) + vite dev (5173) together
```

## Data layout (gitignored)

```
data/                            # All live artifacts (gitignored)
├── benchmarks/<timestamp>_<name>_<commit>/   # Flattened single-config checkpoint
├── benchmarks/_archive/<slug>/                # Archived single-config checkpoint
└── hpo/<timestamp>_<name>_<commit>/           # Optuna study: study.db, *.json + pinn_<ts>/ trials
    └── _archive/<study_slug>/                  # Archived complete HPO study
data_legacy/                     # Pre-consolidation dump (data/, networks/, toroidal_coils_3d/); gitignored, cruft
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
| `NetworkManager` | network.py | Facade: model + state + sampler; accepts optional `FoundationModel` prior to act as a multistage corrector |
| `FoundationModel` | network.py | Frozen dataclass: converged FluxPINN + params; used as prior for residual-correction stage 2 |
| Pydantic req models | `src/api/schemas.py` | `GeometryRequest`, `SampleRequest`, `GridRequest`, `BFieldRequest`, `RenameRequest`, `BenchmarkRequest`, `CoilConfigIn` |
| API response interfaces | `frontend/src/api.ts` | `SampleResponse`, `Grid2D`, `GeometryResponse`, `FieldLinesResponse`, `SurfaceGrid`, `BenchmarkEvent` (SSE union) |
