# Agent instructions

HPO, benchmark, run-inspection, and Optuna conventions live in the `vibe-hpo`
skill. This file only covers the developer-facing basics that every task needs.

## Environment

Always use `uv run`/`uv sync`, never bare `python`/`pip`. Run package entry
points as modules:

```bash
uv run python -m src.engine.network --test
uv run python -m src.engine.optimize_network_optuna --test
```

`--test` runs a minimal-budget smoke test (rapid iteration, no checkpoints /
benchmark dirs); use it as the cheapest self-check before any real run. Both
modules expose many override flags — see `--help` for the full surface (e.g.
`--epochs`, `--arch`, `--hidden-dims`, `--lr` on the network; `--reset-sqlite`
/ `--resume-sqlite`, `--retrain` / `--no-retrain` on the study). Note
`--n-train` must be divisible by `--batch-size`.

Run dedicated drivers through uv: `uv run python scripts/<driver>.py`.
