# Current Focus

## Goal

Build and freeze one strong PirateNet foundation checkpoint before spending more compute on
multistage residual correction. The current campaign asks whether PirateNet + Random Weight
Factorization can beat the best reevaluated hard-BC MLP under one fixed evaluator, then freezes
the winning architecture and optimizer settings for later corrector work.

## Active Work

The active branch is `vibe-improve-network-performance`. Recent commits (2026-07-17/18) added
RWF, PirateNet, multistage correction, foundation/corrector UI metadata, a unified
`NetworkManager` facade, and patience-based stopping with a focused unit test. The working tree
also contains the new campaign specification (`docs/hpo/hpo-plan-piratenet.md`), resumable driver
(`scripts/run_piratenet_foundation_campaign.py`), and local HPO-driver changes; these campaign
assets are not yet committed.

The staged campaign is:

1. Completed: batch 32 beat batch 64 at 600 epochs (fused `9.31e-3` vs `3.74e-2`) and is frozen.
2. Completed: MSE beat Huber by 15.6% fused with better p95; `huber_delta=0.0` is frozen.
3. Completed: `(200,200)` beat `(128,128)` and `(128,128,128)` by 33% (fused `5.17e-3` vs
   `7.76e-3`/`7.77e-3`); depth added nothing at this budget. `hidden_dims=(200,200)` is frozen.
4. Active: L-BFGS polish A/B (2026-07-19 amendment) — winner config + `lbfgs_steps=300`,
   decided pairwise against the unpolished winner, then fixed for all later runs.
5. Run eight LR/adaptive-sampling anchors (P0-P7; P7 added 2026-07-19 with a high `3e-5` LR
   floor after the `(200,200)` trace showed LR starvation below `~2.5e-5`).
6. Run clean 12-trial broad and 12-trial narrowed 600-epoch Optuna studies (broad
   `learning_rate_min` bound raised to `5e-5`), then confirmation seeds 43/44.

The campaign driver now resumes by campaign name rather than commit (mid-campaign commits
previously forked a fresh campaign directory) and recovers completed runs from `result.json`
if the orchestrator dies before recording them. A `(200,200)` run costs ~44 min on the
RTX 3060 (vs ~24 min for `(128,128)`), so the remaining ~33 runs are roughly a day of GPU time.

The initial batch-32 screen at peak LR `1e-3` suffered a persistent optimizer regression after
epoch 430 and was stopped after epoch 600. Controlled screens now use peak LR `1e-4` and prefer
batch 32 when stable and within 10% of batch 64.

## Evaluation Contract

- Objective: `loss_median + 0.3 * loss_p95`, minimized.
- Protocol: fixed 200 plasma configurations x 8,192 area-uniform Sobol points.
- Common training/evaluation seed: 42 for screens and HPO; do not change it casually because
  common random numbers are load-bearing for close rankings.
- Admit only current-protocol, 600-epoch observations matching the frozen architecture, batch,
  and loss invariants to Optuna warmstart.
- Keep hard BC, no Fourier features, no L-BFGS, fixed collocation/training budgets, and fixed
  weight decay throughout this foundation campaign.

## Evidence

- Best reevaluated 5x200 MLP: fused objective about `7.06e-4`; this is the external acceptance
  baseline, not PirateNet warmstart data.
- The controlled 600-epoch batch screen selected batch 32: fused objective `9.312e-3` versus
  `3.740e-2` for batch 64, with p95 `1.696e-2` versus `6.696e-2`. Batch 32 remains fixed
  throughout the exploratory hyperparameter study.
- PirateNet + RWF incumbent `(128,128)`, no Fourier features, batch 64, Huber, 2,400 epochs:
  median `1.123e-3`, p95 `3.903e-3`, fused `2.294e-3`. It justifies a controlled campaign but is
  not yet competitive with the MLP baseline.
- The stored composed corrector fixture improves that incumbent median from `1.123e-3` to
  `8.493e-4` at scale `0.01`, showing correction is viable after the foundation is selected.
- The July KPI unification and full-f32 precision fix remain prerequisites: old `val_loss`,
  pre-reevaluation reports, and stale Optuna values are not comparable with current results.
- Physics train-step optimizations reduced the direct-training benchmark from 42.7 to 28.4 ms
  on the RTX 3060; runnable evidence remains in `docs/performance/`.

## Open Questions

- MSE is mildly favored by Fourier-free history, but no controlled PirateNet comparison exists.
- PirateNet capacity and the useful LR/adaptive-sampling neighborhood remain unresolved.
- Generic legacy warmstart paths can mix stale objectives or widen distributions. The campaign
  must use its clean studies, strict invariant checks, and source provenance rather than the old
  July databases.
- End-of-study PDF generation requires `pdflatex`; availability on the remote GPU box is still
  unconfirmed.
