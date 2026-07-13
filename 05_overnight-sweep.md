# 05 — Overnight Sweep & Literature Iteration

Campaign to decide **legacy soft-BC (bb503b0-style) vs. new hard-BC** architecture
under the *corrected* GS physics, then iterate with literature methods if the old
model is surpassed.

**Ground rule (from the user): ignore KPI numbers. Judge by visually inspecting the
residual/flux plots and by reading the training logs.** KPIs are recorded only as a
coarse tiebreaker, never as the decision.

Budget ceiling: **2500 epochs + 100 L-BFGS steps per run** (hard). All sweep runs use
2400 epochs, 0 L-BFGS (L-BFGS reserved for a polish pass on the winner).

---

## Reference baseline (captured before backward-compat removal)

`pinn_2026_04_23_02_43_bb503b0_0.flax`, re-evaluated under current corrected-physics
eval: **core_med = 0.0868** (matches its historically recorded 0.087). Artifacts in
`data/benchmarks/bb503b0/pinn_2026_04_23_02_43_bb503b0_0/`.

Caveat baked into the whole campaign: bb503b0 was trained while two physics bugs
(axis-sign assumption + flux_depth sign loss) silently disabled the real GS source
terms — so its 0.087 was earned under a *Laplacian-only* loss. Every sweep run below
trains under the corrected loss, so a like-for-like score match is not expected; the
question is which architecture best solves the *real* GS equation.

---

## Sweep design (`scripts/run_sweep.py`, `scripts/run_overnight_sweep.sh`)

R-family anchors to bb503b0's exact recorded hyperparameters + the mandatory
collapse-guard, varying one axis at a time. N-family gives the hard-BC architecture
an equal 2400-epoch budget.

| Run | Family   | What it isolates |
|-----|----------|------------------|
| R1  | soft-BC  | bb503b0 reproduction, 600 epochs, corrected physics only |
| R2  | soft-BC  | budget alone (2400 epochs) |
| R3  | soft-BC  | resampling: was bb503b0's aggressive σ=0.17 a bug-compensation artifact? (→0.05) |
| R4  | soft-BC  | Huber PDE loss (δ=1.0) vs MSE |
| R5  | soft-BC  | Random Fourier features (64) |
| N1  | hard-BC  | envelope instead of soft penalties, 2400 epochs |
| N2  | hard-BC  | + Fourier(64, σ=1) + Huber(1.0) |
| N3  | hard-BC  | bb503b0's tuned hyperparameters, hard-BC envelope |
| R6  | soft-BC  | **bigger net 5×200** (vs 4×128 — the bb503b0 shape was never tuned for size) |
| N4  | hard-BC  | bigger net 5×200 |
| R7  | soft-BC  | **larger batch** (128 vs 32) |
| N5  | hard-BC  | larger batch (128 vs 64) |

Run order is front-loaded so a short night still answers the core questions:
`R1 R2 N1 R6 N4 R7 N5 R3 R4 R5 N2 N3`. Each run is followed by
`model_evaluation.py --plot-quantity {residual,flux}`.

### Analysis rubric (per user guidance)

- **Do not call a run "converged" from the plot alone — read the log first.** If the
  final logged epoch still shows loss falling steeply *and* the LR hasn't annealed to
  `learning_rate_min`, the run was cut mid-descent. Epoch budget is capped at 2500, so
  the lever there is a **bigger/different architecture**, not just more epochs.
- Architecture size is a prime suspect: the 4×128 shape was inherited from bb503b0, not
  optimized. 5×200 (R6/N4) directly tests this.
- Larger batch (R7/N5) tests whether noisier small-batch gradients were limiting.

---

## Execution

- **Host:** imperator (RTX 3060 12GB), tmux `fusion-simulation` window `sweep`.
- **Synced git HEAD on imperator:** `9ba7647` (local working-tree edits scp'd on top;
  run artifacts therefore land under `data/benchmarks/9ba7647/…`).
- **Launched:** 2026-07-13 00:26 CEST. R1 confirmed training (GPU 100%, loss 724→239
  over first 20 epochs).
- Driver log: `logs/sweep_driver.log` (markers: `RUN DONE …`, `ALL DONE`, `JOB FAILED`).

---

## Observations

All 12 runs completed 2026-07-13 07:29 CEST (~7h). Baseline for comparison:
**bb503b0 core_loss_median = 0.0868** (corrected-physics re-eval), residual plot shows
a uniform purple wash (mean |R_GS| ≈ 0.089) across all 8 eval configs, flux plot shows
a sane peaked ψ profile (peak ≈ 40).

| Run | loss@600/1200/1800/2400 | LR@end | core_med | Verdict |
|-----|--------------------------|--------|----------|---------|
| R1  | 2.39 (600ep, fully annealed) | 2.6e-6 | 0.302 | soft-BC repro at bb503b0's own 600-epoch budget is **much worse** than the archived bb503b0 checkpoint (0.30 vs 0.087) — confirms bb503b0's recorded score really was earned under the old Laplacian-only (buggy) loss; the real GS source term is a harder problem. |
| R2  | 2.85→0.65→0.42→0.29 | 2.6e-6 (annealed) | 0.195 | 4x more epochs helps (0.30→0.20) but LR is already at floor and loss is still falling ~30%/600ep at the end — **compute-bound, not LR-bound**. Soft-BC alone can't close the gap in-budget. |
| R3  | 1.80→0.56→0.39→0.29 | 2.6e-6 | 0.182 | Lower resampling σ (0.05) gives a small edge over R2 — bb503b0's aggressive σ=0.17 was *not* helping under corrected physics. |
| R4  | 1.68→0.46→0.30→0.25 | 2.6e-6 | 0.233 | Huber loss underperforms MSE here — soft-BC penalty terms don't benefit from robust loss the way hard-BC residuals might. |
| R5  | 7.77→0.73→0.58→0.57 | 2.6e-6 | 0.182 | Fourier features help vs R2 but plateau early; edge_p95 blows up to 0.71 — spectral bias fix doesn't help the boundary-penalty formulation. |
| R6  | 0.77→0.51→0.19→0.12 | 2.6e-6 | **0.144** | **Bigger net (5×200) is the strongest lever in the soft-BC family** — beats R2/R3/R4/R5 outright, and still falling ~35%/600ep at cutoff → underfit, not overfit; would improve further with more epochs/depth. |
| R7  | 42.1→5.7→2.4→1.8 | 2.6e-6 | 0.302 | Batch 128 (vs 32) is actively harmful in-budget — 4x fewer gradient steps at fixed epoch count outweighs any noise-reduction benefit. Confirms batch size is not a free lunch here. |
| N1  | 0.012→0.0034→0.0016→0.0015 | 5.0e-5 (HARD's floor, **not near zero**) | 0.0052 | Hard-BC **crushes every soft-BC run by 1–2 orders of magnitude**, converges almost immediately (flattens by epoch 1200). But note: LR floor is HARD's own 5e-5, much shallower anneal than bb503b0's 2.6e-6 — room to dig deeper. |
| N4  | 0.013→0.0034→0.0013→0.0014 | 5.0e-5 | 0.0039 | Hard-BC + 5×200 ≈ N1, marginal gain — architecture matters far less once hard-BC removes the boundary-penalty optimization difficulty. |
| N5  | 0.037→0.011→0.0066→0.0054 | 5.0e-5 | 0.0169 | Batch 128 hurts hard-BC too (3x worse than N1), same fewer-steps effect as R7. |
| N2  | 0.0023→0.0011→0.00075→0.00075 | 5.0e-5 | 0.0065 | Fourier+Huber on hard-BC ≈ N1, no clear win. |
| N3  | 0.0020→0.00042→0.00032→0.00030 | **2.6e-6** (bb503b0's deep anneal) | **0.0025** | **Best of all 12 runs.** hard-BC + bb503b0's *exact* tuned schedule/batch/boundary-weight. ~2x better than N1 — isolates that **anneal depth (LR floor), not just hard-BC itself**, is doing real work, exactly the "LR may still be falling" risk flagged — except here it's the opposite: shallow-floor runs (N1/N4/N5/N2) stopped annealing too early relative to what bb503b0's schedule proves is achievable. |

**Visual confirmation (not just KPIs):** N3's residual plot is essentially uniform
near-black across all 8 eval configs (mean |R_GS| ≈ 0.0048) — a qualitatively different
picture from bb503b0's uniform purple wash (mean 0.089). N3's flux plot shows a clean,
physically sane peaked ψ profile (peak ≈ 55, comparable shape to bb503b0's peak ≈ 40).
This is a real, visually obvious improvement, not a KPI artifact.

## Decision (round 1)

**Hard-BC (envelope) architecture decisively beats soft-BC (penalty) under the
corrected GS physics** — by 1–2 orders of magnitude, confirmed visually via residual
plots, not just KPI numbers. bb503b0's soft-BC approach cannot close this gap even
with 4x its original epoch budget or a bigger network.

Winning config so far: **N3** = bb503b0's exact tuned hyperparameters (batch=32,
weight_boundary=20, σ_resample=0.17, LR 1.84e-4→2.6e-6) with `soft_bc=False`.

**New model (N3) has surpassed the legacy baseline** (core_med 0.0025 vs 0.0868, and
visually confirmed via plots) → per the original task authorization this unlocks:
commit to `vibe-improve-network-performance`, literature search for further gains.

Two open threads before committing:
1. **Round 2** (launching now): N6 = N3 + 5×200 (does architecture still help once
   hard-BC + deep anneal are in place?), N7 = N3 + 100 L-BFGS polish steps (unused
   budget headroom — max is 2500 epochs + 100 L-BFGS, N3 uses 0).
2. Literature pass for further hard-BC-specific training methods (next step).
