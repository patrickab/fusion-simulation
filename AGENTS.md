# Agent instructions

## Environment

Always use `uv run`/`uv sync`, never bare `python`/`pip`. Run package entry
points as modules, for example:

```bash
uv run python -m src.engine.network --test
uv run python -m src.engine.optimize_network_optuna --test
```

Run dedicated drivers through uv: `uv run python scripts/<driver>.py`.

## Inspecting a run

Retained runs use the consolidated artifact contract: `run.json`, `metrics.json`,
`network.flax`, and plots. Runtime readers do not support the old split files.
Set the direct run directory once, for example:

```bash
RUN=data/benchmarks/<timestamp>_<name>_<commit>
```

Inspect the complete run record, including configuration, outcome, resources,
and KPIs:

```bash
uv run python -m json.tool "$RUN/run.json"
```

Print only the configuration or KPIs through the same projections used by the
application:

```bash
uv run python -c 'import json, sys; from pathlib import Path; from src.lib.run_artifacts import load_config; print(json.dumps(load_config(Path(sys.argv[1])), indent=2))' "$RUN"
uv run python -c 'import json, sys; from pathlib import Path; from src.lib.run_artifacts import load_kpis; print(json.dumps(load_kpis(Path(sys.argv[1])), indent=2))' "$RUN"
```

Replay the Rich training table from `metrics.json`:

```bash
uv run python -m src.engine.network --show "$RUN"
```

For campaign comparisons, calculate the standard fused objective from stored
KPIs without rerunning evaluation:

```bash
uv run python -c 'import sys; from pathlib import Path; from src.lib.run_artifacts import load_kpis; k = load_kpis(Path(sys.argv[1])); print(k["loss_median"] + 0.3 * k["loss_p95"])' "$RUN"
```

## Long-running jobs

Never launch training, sweeps, HPO, or benchmarks with bare
`nohup ... & disown`. Always use tmux so the user can attach and inspect output.

- Reuse an existing tmux session when possible; otherwise create one with
  `tmux new -d -s work`.
- Create a window named `claude: <name>`:
  ```bash
  tmux new-window -t <session> -n "claude: <name>" "<command>; read"
  ```
- The command must stream useful progress to the pane. Do not disable its live
  display or redirect output away from tmux; if the program cannot render live
  progress, pipe stdout and stderr through `tee` to a named log file.
- After launch, capture the pane once to verify that progress is visible, and
  report both `tmux attach-session -t <session>` and the exact window name so
  the user can select it with `tmux select-window -t '<session>:<window>'`.
- Keep the trailing `; read` so output remains visible after completion.
- Report the session/window name and the command used.

## Benchmark procedure

1. State the comparison and success metric before launching work. Change only
   the variables under test; keep the evaluator, seeds, budgets, and static
   configuration fixed.
2. Run the cheapest available smoke test or self-check first.
3. Give every new search space or protocol a dedicated study/run name. Never
   resume a database after changing its distributions, objective, model
   configuration, budget, or evaluator.
4. Launch long work through uv inside tmux. For the direct Optuna CLI, pass
   `--reset-sqlite` or `--resume-sqlite` explicitly when a database exists.
   Programmatic drivers must set `restart=True` for a clean run or
   `restart=False` for a compatible resume.
5. Preserve enough provenance to reproduce and compare results: git commit,
   configuration, seed, objective, evaluator protocol, budget, and source run.
   Warmstart only from observations matching all of them and fitting the
   current parameter distributions.
6. Verify completed artifacts and compare the declared metrics before drawing
   conclusions. Do not mix legacy or incompatible results into rankings.

Put experiment-specific parameters, phases, and acceptance rules in a driver
under `scripts/` and its protocol document under `docs/`, not in this file.

## Optuna conventions

- `SearchSpaceConfig` owns model/training parameters: scalar values are pinned,
  lists are discrete choices, and `Range` values are continuous axes.
- `StudyConfig` owns orchestration such as trial count, pruning, ranking,
  persistence, and optional foundation-model configuration. Do not pass its
  fields into `HyperParams`.
- Warmstart trials inform the sampler but do not consume the local trial budget.
- Patience-stopped trials are completed and rankable; exceptions are failures.
- `checkpoint_policy="none"` keeps no trial directories, `"top_k"` saves every
  completed trial after evaluation and ranks the requested top set, and
  `"all"` saves every trial during training.
- Pruned, failed, and aborted trials must discard incomplete run directories.
- Study artifacts live under `data/hpo/<timestamp>_<study_name>_<commit>/`.
