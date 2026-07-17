# Current Focus

## Goal
Reduce PINN prediction error for the Grad-Shafranov solver. Hard-BC
architecture won model selection (see `model_selection_report/report.tex`:
soft-BC baseline R2 core median 0.206 → hard-BC N6 6.7e-4, 100-config
protocol). Optimizer and capacity knobs are now exhausted; current levers are
training-step speed (active branch) and the eval/HPO methodology fixes in
`todo.md`.

## Active branch: `vibe-improve-network-performance`
Four perf commits (2026-07-16): remove redundant loss remat, fuse axis+operator
passes, cache boundary Fourier fit, fuse radial JVP. Train step 42.7 → 28.4 ms
(−33%), XLA temp memory 863 → 823 MiB. Each has a note in `docs/performance/`
with the runnable benchmark; follow that format for further perf work.

KPI tracking unified (2026-07-17): one batched+jitted eval core
(`evaluate_residual_samples`) feeds training-time validation, kpis.json and HPO
ranking; budgets calibrated by a 10-checkpoint × 200-config study
(`docs/evaluation/kpi-accuracy-benchmark.md`, harness
`scripts/kpi_accuracy_benchmark.py`). The protocol is two globals in
`src/lib/config.py`: `KPI_POINTS_PER_CONFIG = 8_192` (was 16,384; calibrated
minimum 4,096) and `KPI_EVAL_CONFIGS = 200` (replaced `N_VALIDATION_SIZE=128`
/ HPO `n_validate=20`); all paths score the same `kpi_benchmark_configs`
stream, the dedicated validation Sampler is gone (its boundary-theta draw
leaked into the Fourier fit → ~2e-4 KPI offset for identical domain vectors),
and `val_kpi_median` (full protocol, 0.64 s cached) replaces the
composite-loss `val_loss`. Ranking is protected by the fixed eval seed (common random
numbers) — don't change the seed casually.

## Design decisions that shape the code
1. **Hard Dirichlet BC**: `psi_fn` output × envelope `1 - ρ̃²`,
   `ρ̃ = boundary_normalized_radius(R,Z,boundary)` (`src/engine/plasma.py`).
   ψ=0 at edge by construction. `denormalize_psi` (`network.py`) unifies
   train/infer/API. Soft-BC survives as `--soft-bc` + `model_evaluation
   --legacy` so pre-hard-BC checkpoints evaluate as trained.
2. **Smooth boundary**: `boundary_normalized_radius` fits r(α) as a 32-harmonic
   ridge-LS Fourier series (C^∞); coefficients cached on `PlasmaBoundary`.
   `bnd_leak_max` KPI guards the fit.
3. **ψ sign-agnostic**: `estimate_psi_axis` uses top-|ψ| signed mean (now fed
   by the operator's primal ψ); interior-mean hinge pins new runs to ψ>0 at
   axis; mixed-sign legacy checkpoints still evaluate correctly.
4. **Defaults** (`src/lib/network_config.py`): `huber_delta=1.0`,
   `n_fourier_features=64`, `lbfgs_steps=0`.
5. **HPO parity**: Optuna builds `HyperParams` from `SearchSpaceConfig` and
   calls `NetworkManager.train` (no duplicated loop); dedicated validation
   sampler; Textual TUI, non-tty → plain Rich Live. Trials ranked by fused
   `median + β·p95` of held-out |R_GS|; study end emits `benchmark_report.pdf`
   (`src/engine/benchmark_report.py`, direct pdflatex — pandoc dropped because
   it silently ate `\clearpage`).
6. **Artifact layout**: see architecture.md (flattened
   `data/benchmarks/<ts>_<name>_<commit>/`, studies under `data/hpo/`).

## Remote GPU box
**imperator**, `100.89.120.40` (Tailscale), user `noob`, RTX 3060 12GB.
Password auth only (ssh skill; never hardcode). No deploy key: sync via
`git diff <remote_sha> <local_sha> > patch; scp; git apply`. `uv` at
`~/.local/bin/uv` (non-login PATH). `pdflatex` needed there for study-end
reports — not yet confirmed installed.

## Findings that gate next steps
- **Capacity is not the lever either.** After the 2026-07-14 lr/sigma study
  showed optimizer knobs exhausted (all 6 trials flat from epoch ~1000 while
  lr annealed 100x), the follow-up `arch_wide_or_deep_2400ep` (2026-07-16,
  commit 4a25aff, 26 trials / 12 completed / 14 pruned) answered the
  width/depth question: best 5x200 at 0.00647, with 5x320 and 6x256 within
  1.2% — the whole top-5 is a tie at the same ~1.9e-3 residual floor.
- **The floor is the O-point singularity, not spectral bias** (`todo.md`,
  bottom section): a heavy tail localized at the magnetic axis, where ∇ψ→0
  and the coarse `estimate_psi_axis` injects coherent source error. Planned
  fixes: ε-disk mask/down-weight around the axis, sharper axis estimate,
  Cartesian-window eval to rule out polar-grid aliasing.
- **The old 1.9e-3 "floor" was mostly TF32 noise.** After the precision fix
  (36caa61) the 2026-07-17 full re-evaluation (new 8,192×200 protocol) puts
  the same checkpoints at median ~3.1e-4, p95 ~1.3e-3 (best n6 trial fused
  7.1e-4, was 6.5e-3). All stored kpis.json / study.db values were refreshed
  (`scripts/reevaluate_hpo_kpis.py`); pre-2026-07-17 numbers in reports and
  todo.md are on the old TF32 scale and not comparable. The O-point tail
  shape survives the fix, but absolute targets in todo.md need rebasing.

## Open threads (each has a root-level handoff note)
- **`todo.md` — near-floor HPO plan**, in priority order: (1) relative
  per-config residual in `evaluate_plasma_kpis` so the median measures fit
  quality, not which configs are easy; (2) rebuild warmstart properly; (3)
  keep fixed eval seed; (4) only then GP-BO (BoTorchSampler auto-switch for
  all-continuous spaces, log-objective mandatory) with a replicated
  TPE-vs-UCB/MES/NEI benchmark.
- **Warmstart is dead until rebuilt**: the kpis.json-based ledger scores ~125x
  off the live objective (inverts TPE's prior; its docstring still wrongly
  claims same-scale); the study.db-based fix was only ever uncommitted state
  on imperator and a `git reset --hard` (2026-07-15) destroyed it. Nothing in
  history matches the prior study's budget fields. `WARMSTART_CONFIG_PATHS=[]`,
  every study starts cold. Rebuild = committed `load_prior_study_configs`
  reading `.value` off `study.db` + explicit budget pinning.
- **`plot-error.md` — unresolved**: frontend Plotly residual heatmaps look
  soft/blurred vs the sharp matplotlib montage for identical data. Contour
  levels, resolution caps, and API path all fixed/verified and ruled out;
  remaining suspects are contourcarpet's interpolation vs gouraud pcolormesh,
  stale bundle, plasma-vs-magma colormap, CSS raster downscaling.
- **`latex-benchmark.md`**: report generator rewritten (pdflatex, per-config
  pages, settings/KPI grids) and verified on a 6-config study; still to check:
  live end-of-study invocation, pdflatex on imperator, >6-config pagination,
  Fourier-feature label formatting.
- `src/streamlit/` still imported by the FastAPI layer
  (`src.streamlit.network_utils`) — candidate for relocation to `src/lib/`.
- Pre-2026-07-12 checkpoints trained on a Laplacian-only loss (two since-fixed
  source-term bugs) — invalid for ranking; re-evaluate under current
  `model_evaluation` before trusting old KPIs.
