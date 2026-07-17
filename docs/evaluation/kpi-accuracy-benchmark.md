# KPI Accuracy Benchmark: calibrating (n_points, n_configs)

**Question.** The post-training / HPO benchmark evaluated `|R_GS|` statistics on
16,384 Sobol points per plasma config (`DEFAULT_KPI_SAMPLE_SIZE`). Is that
budget necessary, and how many plasma configs are needed for stable
means/quantiles? Calibrated budgets let one unified KPI path serve training-time
tracking (cheap), post-training eval and HPO ranking (accurate) without paying
the maximum budget everywhere.

**Harness.** `scripts/kpi_accuracy_benchmark.py`, all evaluation through the
unified `evaluate_residual_samples` (`src/engine/model_evaluation.py`). Raw
results in `data/kpi_accuracy/run{1,2,3}.json`; tables below reproduce with
`--analyze`. Environment: RTX 3060 12 GB, JAX GPU, f32 with
`jax_default_matmul_precision="highest"`.

**Checkpoints.** 10 stored networks spanning median `|R_GS|` 1.8e-3 … 0.19 and
five architectures (4×128 with/without Fourier features, 5×200, 6×256, 5×320),
including the ~1.85e-3 near-tie cluster from the width/depth study — the hard
case for ranking stability.

**Statistics tracked.** Pooled `loss_median`, `loss_mean`, `loss_p95`,
`loss_p05`, `core_loss_median`, `edge_loss_p95` (as in `evaluate_plasma_kpis`),
plus the fused HPO ranking score `median + 0.3·p95`.

## Run 1 — points per config

Fixed 200 plasma configs (the CLI's domain-Sobol stream, seed `BASE_SEED+123`);
each budget evaluated with 4 independent Sobol scramblings and compared against
the mean of 4 seeds at 16,384 points. Numbers are the **worst relative error
over all 10 networks and 4 seeds**; `rank_flips` counts seeds whose fused-score
ordering of the 10 networks differs from the reference ordering.

| n_points | median | mean | p95 | p05 | core_med | edge_p95 | rank flips | t/eval |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 113.3% | 61.0% | 42.9% | 153.6% | 97.5% | 79.8% | 2/4 | 1.1 s |
| 256 | 24.0% | 15.4% | 8.3% | 30.4% | 15.5% | 22.2% | 0/4 | 1.2 s |
| 512 | 6.8% | 8.1% | 5.6% | 7.3% | 12.4% | 5.9% | 0/4 | 1.2 s |
| 1024 | 3.3% | 4.0% | 3.3% | 2.5% | 6.2% | 2.4% | 0/4 | 1.3 s |
| 2048 | 2.0% | 3.7% | 1.5% | 2.2% | 3.4% | 1.0% | 0/4 | 1.4 s |
| **4096** | **1.2%** | **2.7%** | **1.2%** | **1.0%** | **1.7%** | **0.6%** | **0/4** | **1.6 s** |
| 8192 | 0.7% | 1.4% | 0.6% | 0.7% | 1.0% | 0.3% | 0/4 | 2.0 s |
| 16384 (ref self-noise) | 0.2% | 0.5% | 0.2% | 0.5% | 0.3% | 0.1% | — | 2.5 s |

Two structural findings:

1. **Error is inversely related to network quality.** The soft-BC checkpoint
   (median 0.19) is accurate to 0.2% already at 128 points; the near-floor
   5×200/5×320/6×256 cluster (median ≈1.85e-3) carries the worst-case error at
   every budget. Their residual mass is concentrated in the heavy O-point tail
   (median 1.9e-3 vs p95 1.4e-2), so quantile estimators are noisiest exactly
   for the checkpoints HPO must distinguish. Budgets must be sized for good
   networks, and will only get more demanding as training improves.
2. **Common random numbers protect the ranking.** All candidates share one
   Sobol draw per seed, so sampling error is highly correlated across networks
   and the fused-score ordering survives per-stat noise well above the ~1.2%
   cluster separation (zero flips from 256 points up). This only holds while
   every candidate is evaluated on the *same* points — the fixed KPI seed is
   load-bearing, and cross-study comparisons (different seeds/eval code) need
   the absolute-accuracy budgets below, not the ranking-stability ones.

**Choice: `n_points* = 4096`.** Every statistic is within 2.7% worst-case
(median ≤1.2%, at the cluster separation), 4× cheaper than 16,384. 2048 would
still rank correctly but lets core-median error grow past 3%.

## Run 2 — number of plasma configs

Fixed 4096 points; config count swept with 4 independent domain-Sobol draws per
budget, compared against the run-1 reference (200 configs at 16,384 points).
Same worst-over-networks-and-draws convention.

| n_configs | median | mean | p95 | p05 | core_med | edge_p95 | rank flips |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 10.7% | 19.4% | 26.3% | 20.8% | 14.6% | 31.5% | 0/4 |
| 25 | 9.9% | 5.7% | 10.0% | 11.0% | 10.1% | 13.9% | 0/4 |
| 50 | 4.5% | 5.4% | 7.5% | 8.3% | 5.9% | 11.0% | 0/4 |
| **100** | **2.6%** | **4.8%** | **4.2%** | **4.3%** | **3.1%** | **7.1%** | **0/4** |
| 200 | 2.8% | 3.3% | 3.8% | 3.9% | 2.8% | 5.9% | 0/4 |

Findings:

1. **100 configs is the plateau.** Going 100 → 200 no longer halves the error
   (median 2.6% → 2.8%): at 4096 points the per-config point noise and the
   reference's own config draw dominate, so extra configs buy nothing until
   points are increased too. 50 configs and below leave visible (>5%) noise in
   the tail statistics, which sit in few high-residual configs.
2. **Ranking again survives everything** — including 12-config draws — for the
   same common-random-numbers reason: every candidate sees the same configs.
   HPO's `n_validate = 20` is therefore defensible for *within-study* pruning
   and ranking (fixed validation set shared by all trials), but 20-config
   absolute KPIs carry ~10-26% error and must not be compared across studies
   or against `kpis.json` from the 100-config eval.

**Choice: `n_configs* = 100`** (the existing `KPI_CONFIG_COUNT`), with the
budget freed by 16,384 → 4096 points redirected to nothing — the full
benchmark eval drops ~4× in cost.

## Run 3 — joint stability check

Stability of the chosen baseline under **fully independent joint draws** (new
point scrambling *and* new config set per draw — the harshest test, unlike the
fixed-seed production eval). Numbers are the worst per-network relative range
(max−min)/mean across draws; baseline uses 8 draws, escalations 4.

| variant | median | mean | p95 | p05 | core_med | edge_p95 | ranking stable |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 4096 × 100 | 5.1% | 4.5% | 9.0% | 6.8% | 5.1% | 11.1% | yes (8/8) |
| double points 8192 × 100 | 4.0% | 1.8% | 8.5% | 4.8% | 3.0% | 8.3% | yes (4/4) |
| double configs 4096 × 200 | 2.8% | 5.1% | 5.1% | 5.2% | 2.5% | 6.4% | yes (4/4) |

The 10-network ordering (including the ~1.2%-separated near-tie cluster) never
changed in any of the 16 draws, so the baseline needs **no** increase for
ranking purposes. For absolute statistics the residual spread is
config-dominated: doubling configs roughly halves the tail ranges while
doubling points barely moves them. If a future use case needs tighter
cross-study tail numbers, go to 200 configs — not more points.

## Calibrated budgets

| Use | configs | points | accuracy (worst net) | cost/checkpoint |
|---|---:|---:|---|---:|
| Model benchmark: `kpis.json`, CLI eval, HPO final ranking | 100 | 4,096 | median ≤ ~1.2% (points) + config spread ±2.5%; ranking exact | ~1.6 s |
| Training-time tracking + HPO pruning | fixed validation set (20 HPO / 128 direct) | 1,024 | ±3.3% point noise, but fixed-seed CRN makes the curve smooth and trials comparable | ≤ ~0.3 s per eval |
| previously | 100–128 | 16,384 | median ≤ ~0.5% (points), same config spread | ~2.5 s (was minutes pre-unification) |

Rationale:

- **16,384 points was excessive.** Its only gain over 4,096 is sub-point-noise
  (0.5% vs 1.2% on the median) — an order of magnitude below the config-draw
  spread (±2.5%) that dominates either way, and irrelevant to ranking, which is
  seed-pinned (common random numbers) and never flipped at 4,096 in 16
  independent draws.
- **Training-time tracking doesn't need benchmark accuracy.** It needs a smooth,
  trial-comparable curve of the *same metric* the benchmark reports. 1,024
  points on the fixed validation configs gives that at ≤0.3 s per evaluation
  (every 50 epochs ≈ 20+ s of training), so unified KPI tracking costs ~1% of
  training time.
- **Keep the evaluation seed fixed.** All ranking-stability results rely on
  every candidate seeing the same points and configs. Changing the seed (or
  comparing across studies evaluated with different seeds) degrades comparisons
  to the absolute-accuracy bounds above.

## Follow-ups applied from these findings

- `DEFAULT_KPI_SAMPLE_SIZE`: 16,384 → 4,096 (`src/engine/model_evaluation.py`).
- End-of-training `_benchmark_network` now evaluates on the same domain-Sobol
  config stream as the CLI (`BASE_SEED+123`, of which the validation configs
  are a prefix), so training-time tracking, `kpis.json` and CLI re-evaluation
  agree up to config count.
- Training validation (`val` in `training.csv`, HPO pruning signal) now reports
  median `|R_GS|` from the unified KPI path at 1,024 points instead of the
  composite training loss — the tracked curve, the post-training KPIs and the
  HPO objective are the same quantity at different budgets.
