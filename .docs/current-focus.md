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

1. Completed: batch 32 beat batch 64 at 600 epochs and is frozen for the remaining study.
2. Active: screen MSE versus Huber at 600 epochs using batch 32.
3. Compare PirateNet capacities `(128,128)`, `(200,200)`, and `(128,128,128)` at 600 epochs.
4. Run seven LR/adaptive-sampling anchors around the selected architecture.
5. Run clean 12-trial broad and 12-trial narrowed 600-epoch Optuna studies, then confirmation seeds 43/44.

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
