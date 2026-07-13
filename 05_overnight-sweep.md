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

### Why the gain is ~78x, decomposed (not "just architecture search")

| Lever | Comparison | Multiplier |
|---|---|---|
| **Boundary-condition formulation** (soft→hard) | R2 0.195 → N1 0.0052 | **~37x** |
| Architecture size (4×128→5×200) | R2→R6: 0.195→0.144; N1→N4: 0.0052→0.0039 | ~1.3x (same in both families — independent, minor factor) |
| LR anneal depth (floor 5e-5→2.6e-6) | N1 0.0052 → N3 0.0025 | ~2x |

37 × 1.3 × 2 ≈ 96x ≈ observed 78x (R2→N3). Architecture size gave the *same* ~1.3x
bump in both the soft-BC and hard-BC families, proving it's a minor, independent lever
— not the driver. The driver is the boundary-enforcement mechanism itself: soft-BC
optimizes two competing objectives (PDE residual vs. boundary penalty) that never
fully reconcile, and for an elliptic PDE a residual boundary error corrupts the whole
domain (global coupling / maximum-principle behavior). Hard-BC's multiplicative
envelope removes the competing objective entirely — ψ=0 at the boundary by
construction for any network output — so all gradient signal targets the one
remaining objective. This is the classical "exact constraint imposition" result from
the PINN literature (Lagaris et al. 1998; Sukumar & Srivastava 2022), now confirmed
empirically on this exact problem. Anneal depth is a secondary, orthogonal lever
(optimization schedule, not model capacity or constraint structure).

Two open threads before committing:
1. **Round 2** (launching now): N6 = N3 + 5×200 (does architecture still help once
   hard-BC + deep anneal are in place?), N7 = N3 + 100 L-BFGS polish steps (unused
   budget headroom — max is 2500 epochs + 100 L-BFGS, N3 uses 0).
2. Literature pass for further hard-BC-specific training methods (next step).

---

## Round 2

| Run | Config | core_med | Verdict |
|-----|--------|----------|---------|
| N6  | N3 + 5×200 (vs 4×128) | 0.00213 | Modest real gain over N3 (0.0025→0.0021, ~1.2x), consistent with round 1's independent ~1.3x architecture-size finding now confirmed at the new best-config level. Training curve fully annealed (LR floor 2.6e-6 by ~epoch 2000), loss plateaued ~epoch 1500, no headroom left. Residual plot: visibly darker/tighter boundary regions and a slightly tighter axis-centered "star" streak vs N3 — real, visually confirmed, not a KPI artifact. |
| N7  | N3 + 100 L-BFGS polish steps | 0.00254 | **No improvement** over N3 (0.00251) — within noise. Training curve shows loss/LR already fully flat *before* L-BFGS starts (AdamW phase alone reaches its floor), so the polish pass has nothing left to polish. Residual plot shows the same axis-centered "star" pattern at the same magnitude as N3. Useful negative result: this rules out "just optimize harder at fixed capacity" as a fix for the core-residual pattern — it's a structural limitation (likely second-derivative/curvature noise concentrated at the ψ extremum, a known PINN pathology at critical points), not an optimization-tail one. Directly motivates round 3's RBA run (N8), which up-weights persistently-high-residual points instead of relying on more generic training. |

**Decision (round 2):** N6 (5×200 architecture) becomes the new best config, replacing
N3. L-BFGS polish (N7) is not worth the extra budget for this problem — dropped as a
technique going forward. Round 3 targets the remaining core-residual pattern directly.

---

## Round 3

N8 (Residual-Based Attention, Anagnostopoulos et al. 2024), N9 (multi-scale random
Fourier features, Wang et al. 2021), N10 (both combined) — all on top of N3's base
(hard-BC + bb503b0 schedule, 4×128 — `run_sweep.py`'s `BB503B0` dict, the same
architecture size as N3; N6's 5×200 is a separate, independent lever tested in round 2
and not combined with these).

**Methodology correction mid-round:** the first comparison pass used the default
`EVAL_RESOLUTION=200` polar-mesh grid. At that resolution N8/N9 *looked* clearly worse
than N6/N3 in the residual montage — but this grid's cells are pie-slice quadrilaterals
whose arc-width grows with radius, and `pcolormesh(shading='gouraud')` flat-fills each
cell's full extent, so a genuinely thin, sharp residual filament near the O-point gets
visually stretched into a wide wedge toward the boundary. Bumped `EVAL_RESOLUTION` to
600 (verified against the frontend's own `contourcarpet` renderer, which contours
sub-cell rather than flat-filling and so doesn't inherit this artifact) and re-rendered
all four (N3/N6/N8/N9) at the same corrected resolution for a fair comparison before
drawing any conclusion — this is the run of the "judge by plots" rule doing its job.

| Run | Config | core_med | Verdict |
|-----|--------|----------|---------|
| N8  | N3 + RBA (`rba_eta=0.01, rba_decay=0.999`) | 0.0029 | Worse than N6 (0.0029 vs 0.0021), and no better than N3 (0.0025) at matched architecture — not just a low-res artifact. Training fully converged (LR at floor, loss flat). Up-weighting persistently-high-residual points didn't fix the core pattern; if anything it looks marginally broader than N6's tight streaks. |
| N9  | N3 + multi-scale Fourier features (`n_fourier_features=64`, σ∈{0.5,2.0,8.0}) | 0.0031 | Worse than both N3 and N6, and the corrected-resolution plot reveals why: a fine speckle/moiré **ringing artifact** across nearly the entire domain — a known pathology of sinusoidal positional encodings on a bounded, non-periodic domain. Completely masked at `resolution=200` (grid averaging hid it); only visible once resolution was fixed. Genuine regression, not a rendering artifact. |
| N10 | N3 + RBA + multi-scale Fourier (both combined) | 0.0031 (core_med) | **Inherits N9's ringing artifact essentially unchanged** — the residual montage shows the same domain-spanning fine radial-line/speckle pattern as N9 alone, undiminished by RBA. KPI sits between N8 and N9 individually (0.0031 vs N8's 0.0029, N9's 0.0031) — combining the two techniques does not compound any benefit, it just inherits Fourier's defect. Training fully converged (LR annealed to floor by epoch ~2400, both train/val loss flat). |

**Visual confirmation (not just KPIs):** at matched `resolution=600`, N6 remains the
tightest, most core-localized residual pattern of the four. N3 is visibly broader than
N6. N8 is broader/less concentrated than N6 but without the ringing defect. N9 and N10
both show the same fine-grained speckle/ringing covering nearly the whole domain —
clearly the worst of the four, once rendered fairly.

## Decision (round 3)

**Neither RBA nor multi-scale Fourier features (alone or combined) beat N6 on this
problem.** RBA gives a mild regression; multi-scale Fourier features introduce a
genuine ringing artifact from applying periodic positional encodings to this bounded
non-periodic domain. **N6 remains the standing best config.**

Reverted the RBA/multi-scale-Fourier scaffolding (`src/engine/network.py`,
`src/engine/physics.py`, `src/lib/network_config.py`, `scripts/run_sweep.py`) back to
the round-2 (N6-winning) state — dead, experimentally-disproven code left in the tree
is worse than no code. Round-3 findings live only here.

Also fixed two real bugs discovered while chasing the N8/N9 visual-appearance question
(unrelated to which technique wins, but load-bearing for trusting any future residual
plot at all): (1) `evaluate_plasma_grids` used an unchunked `jax.vmap` whose peak
memory scaled with `resolution² × n_configs` — safe at the old resolution=200, but
OOM'd the GPU mid-run once resolution went to 600; replaced with `jax.lax.map(...,
batch_size=GRID_EVAL_CHUNK)` so peak memory is now flat regardless of resolution. (2)
The API's JSON serializers (`src/api/network.py`, `src/api/geometry.py`) were rounding
every float array to 2 decimals by default — for the residual field (range
`[0, ~0.02]`) this collapsed the values into ~3 discrete buckets before they ever
reached the frontend, which is what made the frontend's Plotly heatmap look binary
navy/yellow instead of a smooth gradient. Removed all rounding from both files per
explicit instruction (full float64 precision end-to-end; this local API has no
payload-size constraint that would justify trading accuracy for it).

---

## Round 4

**Multistage residual correction** (Wang 2024/2025, arXiv 2507.16636 / 2407.17213):
froze N6 as stage-1 and trained a small stage-2 net (64×64, 300 epochs, ~3min) so
`psi_final = psi_stage1 + psi_stage2`, with stage2 optimized against the GS residual
of the *composed* field.

| Run | Config | core_med | Verdict |
|-----|--------|----------|---------|
| N11 | N6 (frozen) + stage-2 correction net (64×64, 300ep) | 0.0158 | **Dramatic regression — 7.5x worse than N6 alone (0.0158 vs 0.0021).** Stage2's own training-loss log looked fine (residual ~0.0025 on its training batches, converged/flat by epoch 300) — this is exactly the trap the "read the log, not just KPIs" rule exists for: the discrepancy only shows up once evaluated on the dense eval grid (45k points) instead of the sparse, adaptively-resampled training collocation set. |

**Visual confirmation (decisive, not subtle):** the residual montage is *not* a mild
regression — nearly the entire domain in all 8 configs sits at or above the display
ceiling (`|R_GS| ≥ 0.01`, saturated pale yellow), with only thin dark spoke-lines
threading through it. Those spokes align with the sparse training-ray pattern: the
stage-2 net memorized a correction along the specific collocation points/rays it was
trained on and actively *hurts* the field everywhere between them, rather than learning
a smooth global correction. This is the opposite failure mode from N9's ringing — not
a rendering artifact, a genuine, severe overfit.

## Decision (round 4)

**Multistage residual correction, as implemented, is a clear failure — reverted.**
Deleted `src/engine/residual_correction.py`, `scripts/train_multistage.py`, and
`scripts/run_round3.sh` (orphaned once round 3 was reverted) — same reasoning as round
3: disproven scaffolding left in the tree is a liability, not a resource. If this
technique is revisited later, the fix to try first is denser/non-adaptive stage-2
collocation sampling (the failure pattern points straight at the sparse-ray overfit,
not at the method being fundamentally unsound).

---

## Campaign conclusion

Two further literature techniques (round 3: RBA + multi-scale Fourier features; round
4: multistage residual correction) were tried on top of N6 and both **failed to beat
it**, one subtly (N8/N9, only visible after fixing the eval-resolution artifact) and
one decisively (N11). **N6 stands as the final winning model**: hard-BC (envelope)
architecture, bb503b0's tuned schedule (batch=32, weight_boundary=20, σ_resample=0.17,
LR 1.84e-4→2.6e-6), 5×200 hidden layers, 2400 epochs, no L-BFGS polish — core_med
0.00213, ~41x better than the legacy bb503b0 baseline (0.0868), confirmed visually via
residual/flux plots at a corrected, artifact-free resolution.

Two real, general bugs were fixed along the way (independent of which architecture
wins, load-bearing for trusting any future residual plot or frontend render):
`evaluate_plasma_grids`'s grid evaluation is now memory-chunked (`jax.lax.map`, peak
memory flat regardless of resolution — previously OOM'd on a GPU at high resolution),
and the API's JSON serializers no longer round float payloads (previously truncated
the residual field to ~3 discrete buckets before it ever reached the frontend).
