# Model Selection Benchmark — Learnings

Four curated checkpoints tracing the winning lineage of the overnight sweep
(full rationale/all 15 runs: `05_overnight-sweep.md` at repo root).

| Dir | Milestone | core_loss_median | Why it's here |
|---|---|---|---|
| `..._soft-bc-reproduction` | R1: bb503b0 config, corrected physics | 0.302 | Baseline — legacy soft-BC approach under the *real* GS source term (bb503b0's original 0.087 was earned under a buggy Laplacian-only loss). |
| `..._hard-bc-unoptimized` | N1: hard-BC envelope, untuned schedule | 0.0052 | Isolates the single biggest lever: soft→hard BC alone is ~37x, before any other tuning. |
| `..._hard-bc-tuned-schedule` | N3: hard-BC + bb503b0's tuned LR/batch/σ schedule | 0.0025 | Anneal depth is a real, separate ~2x lever on top of hard-BC. |
| `..._hard-bc-final` | N6: N3 + 5×200 architecture | **0.00213** | **Final winner.** Architecture size is a minor, independent ~1.3x lever, confirmed at this config too. |

**Core finding:** boundary-condition formulation (soft penalty vs. hard envelope)
dominates the gain, not architecture or epoch budget. Two follow-up literature
techniques (RBA + multi-scale Fourier features; multistage residual correction)
were tried on top of N6 and both failed to beat it — see `05_overnight-sweep.md`
rounds 3–4. N6 stands as the final model, ~41x better than the legacy baseline.
