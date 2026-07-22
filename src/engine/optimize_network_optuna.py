"""Optuna HPO driver for the PINN solver.

Searches over network architecture (width x depth), learning rate schedule,
weight decay, and adaptive-sampling sigma. Each trial trains a network and
is ranked by a fused validation score: median + beta*p95 of |R_GS| over a
held-out set. Hyperband pruning uses the per-epoch val_kpi_median (median
|R_GS| at KPI_POINTS_PER_CONFIG x the fixed validation configs) — the same
quantity as the fused score's median term, evaluated at the global protocol
budget (docs/evaluation/kpi-accuracy-benchmark.md).

  SearchSpaceConfig    StudyConfig
       │                   │
       └────────┬──────────┘
                │
    ┌───────────▼────────────┐
    │  warmstart (optional)  │
    │  benchmark study.db    ├──┐
    │  configured run paths  ├──┤
    └────────────────────────┘  │ inject as completed trials
                                │
    ┌───────────────────────────▼──────────────────────┐
    │  Optuna study  (TPE or GP sampler + Hyperband    │
    │  pruner; GP+logEI when the space is all-Range)   │
    │                                                  │
    │  per trial:                                      │
    │    sample HyperParams ◄── SearchSpaceConfig      │
    │    train NetworkManager ──► val_kpi_median ───────┼──► prune?
    │    evaluate |R_GS| stats                         │
    │    score = median + β·p95                        │
    └───────────────────────┬──────────────────────────┘
                            │
    ┌───────────────────────▼──────────────────────────┐
    │  data/hpo/<timestamp>_<name>_<commit>/           │
    │    study.db  trials.csv  top_trials.json         │
    │    benchmark_report.pdf  top-k checkpoints       │
    └──────────────────────────────────────────────────┘

Entry point: main(). Launches HpoApp (TUI) when stdout is a tty, plain
Live dashboard otherwise. Configuration: SearchSpaceConfig, StudyConfig.
"""

import argparse
import ast
import csv
import json
import logging
import os
import shutil
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import jax
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.pruners import HyperbandPruner, NopPruner
from optuna.samplers import BaseSampler, GPSampler, TPESampler

from src.engine.model_evaluation import evaluate_validation_loss_stats
from src.engine.network import (
    EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_ROLLING_WINDOW,
    FoundationModel,
)
from src.engine.network_manager import NetworkManager
from src.engine.residual_correction import load_foundation
from src.lib.config import (
    KPI_EVAL_CONFIGS,
    KPI_POINTS_PER_CONFIG,
    NEURAL_CORRECTOR_DIR,
    Filepaths,
    current_commit,
)
from src.lib.network_config import Architecture, HyperParams
from src.lib.optuna_tui import HpoApp, OptunaProgressDisplay, logger, resolve_reset_choice
from src.lib.run_artifacts import (
    format_duration,
    load_config,
    load_kpis,
    load_run,
    update_run_result,
)

# Budget fields change what the loss *means* (sampling/training amount), so a mismatch
# invalidates a warmstart candidate (with total epochs, see budget_mismatch); nothing else does.
BUDGET_FIELDS = ("n_rz_inner_samples", "n_rz_boundary_samples", "n_train", "lbfgs_steps")


@dataclass(frozen=True)
class Range:
    """Continuous, optionally log-scaled search axis."""

    low: float
    high: float
    log: bool = False


@dataclass
class SearchSpaceConfig:
    """
    Search space for HyperParams, one field per tunable axis.

        - Scalar = pinned
        - list   = discrete choices
        - Range  = continuous
        - None   = not pinned here; the HyperParams default applies

    Fields mirror HyperParams (same grouping) except for corrector_scale.
    """

    # Optimizer & loss
    learning_rate_max: float | Range
    learning_rate_min: float | Range
    weight_decay: float | Range
    sigma_residual_adaptive_sampling: float | Range

    # Architecture choices each study must decide explicitly.
    hidden_dims: tuple[int, ...] | list[tuple[int, ...]]
    rwf: bool
    soft_bc: bool

    # Architecture defaults
    arch: Architecture = Architecture.mlp
    n_fourier_features: int | None = None
    weight_boundary_condition: float | Range | None = None  # soft-BC studies only
    weight_flux_scale: float | Range | None = None  # collapse-guard hinge weight
    huber_delta: float | None = None  # None/0.0 = MSE
    batch_size: int = 32

    # Training budget
    n_train: int = 1024  # size training set
    warmup_epochs: int = 400
    decay_epochs: int = 2000
    n_rz_inner_samples: int = 512
    n_rz_boundary_samples: int = 256
    lbfgs_steps: int = 0

    # --- Neural-corrector studies only; never fed to HyperParams ---
    corrector_scale: float | list[float] | Range = 0.01

    def get_static_params(self) -> dict[str, Any]:
        """Fields pinned to a single value -- fed straight into HyperParams(**...).

        None means "not pinned": the field is omitted so the HyperParams
        default applies (huber_delta's default is itself None, so omitting it
        is equivalent to pinning MSE).
        """
        return {
            f.name: value
            for f in fields(self)
            if f.name != "corrector_scale"
            and not isinstance(value := getattr(self, f.name), list | Range)
            and value is not None
        }

    def get_suggestable_params(self) -> dict[str, list | Range]:
        """Fields Optuna searches: name -> its list of choices or its Range."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "corrector_scale" and isinstance(getattr(self, f.name), list | Range)
        }

    def budget_mismatch(self, hparams: HyperParams) -> dict[str, Any]:
        """Offending BUDGET_FIELDS + total epochs that make hparams incomparable to this
        study's budget, or {} if comparable. Only budget invalidates a historical result.
        """
        mismatches = {
            name: getattr(self, name)
            for name in BUDGET_FIELDS
            if getattr(self, name) != getattr(hparams, name)
        }
        total_epochs = self.warmup_epochs + self.decay_epochs
        if total_epochs != hparams.warmup_epochs + hparams.decay_epochs:
            mismatches["total_epochs"] = total_epochs
        return mismatches


@dataclass
class StudyConfig:
    """Orchestration knobs for the Optuna study itself -- never fed to HyperParams."""

    search_space: SearchSpaceConfig
    study_name: str = "arch_wide_or_deep_2400ep"
    n_trials: int = 20
    top_k: int = 3
    n_startup_trials: int = 10
    n_validate: int = KPI_EVAL_CONFIGS
    min_epochs: int | None = None  # Hyperband pruning floor; None = total_epochs // 8
    prune_trials: bool = True
    checkpoint_policy: Literal["none", "top_k", "all"] = "all"
    score_beta: float = 0.3  # p95 tail weight in ``loss_median + score_beta * loss_p95``

    # Previous experiments can be read back and injected as historical evidence.
    warmstart_experiment_db: Path | None = None
    warmstart_config_paths: list[Path] = field(default_factory=list)
    retrain: bool = False  # retrain budget-mismatched warmstart configs

    foundation_path: str | Path | None = None  # frozen foundation for corrector studies

    @property
    def total_epochs(self) -> int:
        """Per-trial epoch budget, always the search space's schedule."""
        return self.search_space.warmup_epochs + self.search_space.decay_epochs


def load_configs(paths: list[Path]) -> list[tuple[Path, HyperParams]]:
    """Load configs from disk."""
    return [(path, HyperParams.from_dict(load_config(path.parent))) for path in paths]


def _write_objective_metadata(
    study_path: Path,
    *,
    score_beta: float,
    n_validate: int,
    points_per_config: int,
    overwrite: bool = False,
) -> None:
    """Persist the protocol needed to interpret every objective value."""
    path = study_path / "objective.json"
    metadata = {
        "formula": "loss_median + score_beta * loss_p95",
        "score_beta": score_beta,
        "n_validate": n_validate,
        "points_per_config": points_per_config,
        "early_stopping": {
            "patience": EARLY_STOPPING_PATIENCE,
            "min_relative_improvement": EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT,
            "rolling_window": EARLY_STOPPING_ROLLING_WINDOW,
        },
        "evaluator_commit": current_commit(),
    }
    if path.exists() and not overwrite and json.loads(path.read_text()) != metadata:
        raise ValueError(f"Study objective protocol does not match {path}")
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def _validate_foundation_identity(optuna_study: optuna.Study, foundation_dir: Path | None) -> None:
    """Reject resuming a study with a different foundation checkpoint."""
    expected = str(foundation_dir.resolve()) if foundation_dir is not None else None
    stored = optuna_study.user_attrs.get("foundation_path")
    if stored is not None and stored != expected:
        raise ValueError(
            f"Study {optuna_study.study_name!r} is bound to foundation {stored!r}, "
            f"but this run requested {expected!r}. Use a new study name."
        )
    if stored is None:
        optuna_study.set_user_attr("foundation_path", expected)


def study_dir(study_name: str, commit: str | None = None) -> Path:
    """
    Storage root for one Optuna study under data/hpo/.

        data/hpo/<timestamp>_<study_name>_<commit>/

    An existing directory for the same name and commit is reused (resume);
    otherwise a fresh timestamped one is created.
    """

    def _timestamp() -> str:
        return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    commit = commit or current_commit()
    # ponytail: >1 match (two runs of the same name+commit) picks the newest;
    # that's the one a bare re-run means to resume.
    existing = sorted(Filepaths.HPO_ROOT.glob(f"*_{study_name}_{commit}"))
    path = (
        existing[-1] if existing else Filepaths.HPO_ROOT / f"{_timestamp()}_{study_name}_{commit}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_experiment_db(db_dir: Path | None) -> optuna.Study | None:
    """Open the benchmark Optuna study living under ``db_dir/study.db``.

    The study name is auto-discovered: the non-legacy study (i.e. not
    "warmstart_experiments") with the most trials. Returns None if ``db_dir``
    is None, the db is missing, or it holds no eligible study.
    """
    if db_dir is None:
        return None
    storage_path = db_dir / "study.db"
    if not storage_path.exists():
        return None
    storage = f"sqlite:///{storage_path}"
    eligible = [
        s
        for s in optuna.get_all_study_summaries(storage)
        if s.study_name != "warmstart_experiments" and s.n_trials > 0
    ]
    if not eligible:
        return None
    chosen = max(eligible, key=lambda s: s.n_trials)
    return optuna.load_study(study_name=chosen.study_name, storage=storage)


def _reevaluate_config(path: Path, hp: HyperParams, budget: dict[str, Any]) -> HyperParams:
    """Retrain config at the study's current budget, destructively replacing its run artifacts."""
    corrected = HyperParams(**{**hp.to_dict(), **budget})
    stale = sorted(budget)
    logger.warning(f"{path.parent}: stale budget vs. current search space ({stale}) -- retraining")
    manager = NetworkManager(corrected)
    manager.train(save_to_disk=True)
    shutil.rmtree(path.parent)
    shutil.move(str(manager.run_dir()), str(path.parent))
    return corrected


def optuna_warmstart(study: StudyConfig) -> optuna.Study | None:
    """Populate the legacy ``warmstart_experiments`` ledger from configured paths.

    No-op returning None if the list is empty or no benchmark db is configured.
    Budget-mismatched entries are retrained or skipped depending on ``study.retrain``.
    """
    if study.warmstart_experiment_db is None or not study.warmstart_config_paths:
        return None

    db = optuna.create_study(
        study_name="warmstart_experiments",
        storage=f"sqlite:///{study.warmstart_experiment_db / 'study.db'}",
        load_if_exists=True,
        direction="minimize",
    )
    existing_runs = {
        trial.user_attrs.get("run")
        for trial in db.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    }

    for path, hp in load_configs(study.warmstart_config_paths):
        run = str(path.parent)
        if run in existing_runs:
            continue
        mismatched = study.search_space.budget_mismatch(hp)

        if mismatched:
            if not study.retrain:
                continue  # skip: never enters the ledger, never informs TPE
            hp = _reevaluate_config(path, hp, mismatched)

        kpis = load_kpis(path.parent)
        loss = kpis["loss_median"] + study.score_beta * kpis["loss_p95"]

        db.add_trial(
            optuna.trial.create_trial(
                value=loss,
                user_attrs={"config": hp.to_dict(), "run": run},
            )
        )
        existing_runs.add(run)
    return db


def _get_experiment_db_configs(
    db: optuna.Study | None, db_dir: Path | None = None, score_beta: float = 0.3
) -> list[tuple[HyperParams, float, float | None, float | None]]:
    """Configs and scores recorded in a warmstart database, empty if none exists.

    Ledger trials carry a precomputed score; raw benchmark trials are
    rescored from their kpis so both rank on the same scale.
    """
    if db is None:
        return []
    candidates = []
    seen_runs = set()
    for t in db.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        if t.value is None:
            continue
        if (run := t.user_attrs.get("run")) is not None:
            if run in seen_runs:
                continue
            seen_runs.add(run)
        if db_dir is not None and "run" in t.user_attrs:
            run_dir = db_dir / t.user_attrs["run"]
            hp = HyperParams.from_dict(load_config(run_dir))
            kpis = load_kpis(run_dir)
            fused = kpis["loss_median"] + score_beta * kpis["loss_p95"]
            candidates.append((hp, fused, kpis["loss_median"], kpis["loss_p95"]))
        elif "config" in t.user_attrs:
            hp = HyperParams.from_dict(t.user_attrs["config"])
            run_dir = Path(t.user_attrs["run"]) if "run" in t.user_attrs else None
            if run_dir is not None and (run_dir / "run.json").exists():
                kpis = load_kpis(run_dir)
                fused = kpis["loss_median"] + score_beta * kpis["loss_p95"]
                candidates.append((hp, fused, kpis["loss_median"], kpis["loss_p95"]))
            else:
                candidates.append((hp, t.value, None, None))
    return candidates


# Types Optuna can store as a categorical choice natively.
_STORAGE_SAFE = (type(None), bool, int, float, str)


def _suggest(trial: optuna.Trial, name: str, spec: list | Range) -> Any:  # noqa: ANN401
    """Ask Optuna for one value, matching the spec's shape to the right suggest_* call."""
    if isinstance(spec, Range):
        return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    if all(isinstance(choice, _STORAGE_SAFE) for choice in spec):
        return trial.suggest_categorical(name, spec)
    # e.g. hidden_dims candidates are tuples -- not SQLite-safe for suggest_categorical
    # directly, so search over string labels and map back to the real value.
    labels = {str(choice): choice for choice in spec}
    # An injected historical trial may carry a label outside this space's current choices;
    # suggest_categorical would reject it (FrozenTrial checks membership in the choices
    # passed here), so decode the fixed label directly. `name in trial.params` holds only
    # for such completed trials, never a live one mid-objective().
    if name in trial.params and trial.params[name] not in labels:
        return ast.literal_eval(trial.params[name])
    label = trial.suggest_categorical(name, list(labels))
    return labels[label]


def build_hyperparams(trial: optuna.Trial, search_space: SearchSpaceConfig) -> HyperParams:
    """Sample every searchable axis and merge with the pinned ones."""
    suggested = {
        name: _suggest(trial, name, spec)
        for name, spec in search_space.get_suggestable_params().items()
    }
    return HyperParams(**search_space.get_static_params(), **suggested)


def _param_value(spec: list | Range, value: Any) -> Any:  # noqa: ANN401
    """Value in Optuna's storage format: tuple choices are str()-encoded."""
    if isinstance(spec, Range) or all(isinstance(choice, _STORAGE_SAFE) for choice in spec):
        return value
    return str(value)


def _distribution_for(spec: list | Range, value: Any) -> optuna.distributions.BaseDistribution:  # noqa: ANN401
    """Per-injection distribution, widened to contain an out-of-range historical value
    so it still informs TPE instead of being dropped for falling outside the space."""
    if isinstance(spec, Range):
        low, high = min(spec.low, value), max(spec.high, value)
        return FloatDistribution(low, high, log=spec.log)
    if all(isinstance(choice, _STORAGE_SAFE) for choice in spec):
        choices = list(spec) if value in spec else [*spec, value]
        return CategoricalDistribution(choices)
    labels = [str(choice) for choice in spec]
    return CategoricalDistribution(labels if value in labels else [*labels, value])


def _inject_historical_trials(
    optuna_study: optuna.Study,
    search_space: SearchSpaceConfig,
    candidates: list[tuple[HyperParams, float, float | None, float | None]],
) -> None:
    """Inject historical pairs as completed Optuna trials to seed TPE.

    Out-of-range values are widened rather than filtered, so every candidate informs the sampler.
    """
    suggestable = search_space.get_suggestable_params()
    seen = set()
    for hp, loss, _median, _p95 in candidates:
        searched = {
            name: _param_value(spec, getattr(hp, name)) for name, spec in suggestable.items()
        }
        key = tuple(sorted(searched.items()))
        if key in seen:
            continue
        seen.add(key)
        distributions = {
            name: _distribution_for(suggestable[name], value) for name, value in searched.items()
        }
        optuna_study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=loss,
                params=searched,
                distributions=distributions,
                user_attrs={"warmstart": True},
            )
        )


def _discard_study_artifacts(hpo_dir: Path, foundation_dir: Path | None) -> None:
    """Remove discarded trial artifacts without touching other studies' warmstarts."""
    if foundation_dir is not None:
        shutil.rmtree(
            foundation_dir / NEURAL_CORRECTOR_DIR / hpo_dir.name,
            ignore_errors=True,
        )
    for run_dir in hpo_dir.glob("pinn_*"):
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
    for artifact in ("trials.csv", "top_trials.json", "benchmark_report.pdf"):
        (hpo_dir / artifact).unlink(missing_ok=True)
    shutil.rmtree(hpo_dir / "top_k", ignore_errors=True)


def _reset_study(study_name: str, storage_path: Path) -> None:
    """Delete one study through Optuna's public API, preserving sibling ledgers."""
    if not storage_path.exists():
        return
    with suppress(KeyError):
        optuna.delete_study(study_name=study_name, storage=f"sqlite:///{storage_path}")


def check_capacity(study: StudyConfig, foundation: FoundationModel | None = None) -> None:
    """Verify the search space's most resource-demanding config trains without crashing."""
    search_space = study.search_space
    suggestable = search_space.get_suggestable_params()
    picked = {
        # Each spec's most demanding end: Range.high, or a list's first entry
        # (lists like _ARCHITECTURES are hand-ordered biggest-first).
        name: spec.high if isinstance(spec, Range) else spec[0]
        for name, spec in suggestable.items()
    }
    hp = HyperParams(**search_space.get_static_params(), **picked)
    manager = NetworkManager(hp, n_validation_size=study.n_validate, prior=foundation)
    manager.train_epoch(0)
    jax.clear_caches()
    logger.info(f"Capacity check passed: {len(hp.hidden_dims)}x{hp.hidden_dims[0]}")


def objective(
    trial: optuna.Trial,
    study: StudyConfig,
    display: OptunaProgressDisplay,
    hpo_benchmark_dir: Path,
    foundation: FoundationModel | None = None,
    foundation_dir: Path | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> float:
    """Train one network and return its fused ranking score (median + beta*p95 of |R_GS|).

    Per-epoch Hyperband pruning uses val_kpi_median (median |R_GS| at the global
    protocol: KPI_POINTS_PER_CONFIG x fixed validation configs), not the composite
    training loss.  Pruning and ranking therefore measure the same quantity.
    """
    hp = build_hyperparams(trial, study.search_space)
    trial.set_user_attr("config", hp.to_dict())
    trial.set_user_attr("training_seed", 42)
    corrector_scale = 1.0
    if foundation is not None:
        scale_spec = study.search_space.corrector_scale
        corrector_scale = (
            _suggest(trial, "corrector_scale", scale_spec)
            if isinstance(scale_spec, list | Range)
            else scale_spec
        )
        trial.set_user_attr("corrector_scale", corrector_scale)
    display_params = {
        "depth": len(hp.hidden_dims),
        "width": hp.hidden_dims[0],
        "lr_max": hp.learning_rate_max,
        "lr_min": hp.learning_rate_min,
        "wd": hp.weight_decay,
        "sig": hp.sigma_residual_adaptive_sampling,
        "eps": corrector_scale if foundation is not None else None,
    }
    manager = NetworkManager(
        hp,
        n_validation_size=study.n_validate,
        test_mode=study.checkpoint_policy == "none",
        output_dir=(foundation_dir / NEURAL_CORRECTOR_DIR / hpo_benchmark_dir.name)
        if foundation_dir is not None
        else hpo_benchmark_dir,
        prior=foundation,
        scale=corrector_scale,
    )
    total_epochs = hp.warmup_epochs + hp.decay_epochs
    display.start_trial(trial.number + 1, display_params, total_epochs)
    display.current_manager = manager
    manager.metrics_row_sink = display.add_metrics_row
    last_epoch = 0
    completed = False

    def report(epoch: int, val_kpi_median: float | None) -> None:
        # val_kpi_median is median |R_GS| at KPI_POINTS_PER_CONFIG from NetworkManager.
        nonlocal last_epoch
        if cancel_requested is not None and cancel_requested():
            raise KeyboardInterrupt
        last_epoch = epoch
        display.update_epoch(epoch, val_kpi_median)
        if val_kpi_median is not None:
            trial.report(val_kpi_median, epoch)
            if study.prune_trials and trial.should_prune():
                raise optuna.TrialPruned

    try:
        manager.train(
            save_to_disk=study.checkpoint_policy == "all",
            validation_callback=report,
            show_progress=False,
        )
        summary = manager.training_summary or {}
        trial.set_user_attr("stop_reason", summary.get("stop_reason", "unknown"))
        trial.set_user_attr("trained_epochs", summary.get("trained_epochs", total_epochs))
        trial.set_user_attr("best_validation_epoch", summary.get("best_validation_epoch"))
        # Ranking metric: fused median + beta*p95 of |R_GS| over validation
        # configs (see evaluate_validation_loss_stats/score). Median alone ignores
        # worst-case error; max is a high-variance single-point estimator.
        # Stats are computed once and shown to the user alongside the fused score.
        val_median, val_p95 = evaluate_validation_loss_stats(manager)
        val_score = val_median + study.score_beta * val_p95
        trial.set_user_attr("loss_median", val_median)
        trial.set_user_attr("loss_p95", val_p95)
        if study.checkpoint_policy == "top_k":
            manager.to_disk()
        run_dir = manager.run_dir()
        if (run_dir / "run.json").exists():
            update_run_result(run_dir, objective_value=val_score)
        display.update(
            trial.number + 1,
            display_params,
            val_score,
            "done",
            summary.get("trained_epochs", total_epochs),
            median=val_median,
            p95=val_p95,
        )
        completed = True
        return val_score
    except optuna.TrialPruned:
        trial.set_user_attr("trained_epochs", last_epoch)
        trial.set_user_attr("stop_reason", "pruned")
        display.update(trial.number + 1, display_params, None, "pruned", last_epoch)
        raise
    except Exception as exc:
        trial.set_user_attr("trained_epochs", last_epoch)
        trial.set_user_attr("stop_reason", "failed")
        trial.set_user_attr("failure_type", type(exc).__name__)
        trial.set_user_attr("failure_message", str(exc))
        display.update(trial.number + 1, display_params, None, "failed", last_epoch)
        raise
    finally:
        if manager.artifact_stem is not None:
            run = manager.run_dir()
            trial.set_user_attr(
                "run", str(run) if foundation_dir is not None else manager.artifact_stem
            )
        if completed:
            manager.discard_unsaved_run()
        else:
            manager.discard_run()
        jax.clear_caches()


def _write_trials_csv(study: optuna.Study, hpo_benchmark_dir: Path) -> None:
    """Atomically export finalized study trials in a compact comparison table."""
    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    trials = [trial for trial in study.get_trials(deepcopy=False) if trial.state in terminal_states]
    param_names = sorted({name for trial in trials for name in trial.params})
    fields = [
        "trial",
        "state",
        "objective",
        "loss_median",
        "loss_p95",
        "trained_epochs",
        "stop_reason",
        "duration",
        "seed",
        "warmstart",
        "run",
        "source_study",
        "source_trial",
        "source_run",
        "failure_type",
        "failure_message",
        *[f"param_{name}" for name in param_names],
    ]
    path = hpo_benchmark_dir / "trials.csv"
    temporary = path.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            attrs = trial.user_attrs
            run_name = attrs.get("run", "")
            run_dir = _trial_run_dir(hpo_benchmark_dir, run_name) if run_name else None
            run = (
                load_run(run_dir) if run_dir is not None and (run_dir / "run.json").exists() else {}
            )
            result = run.get("result", {})
            kpis = result.get("kpis", {})
            row = {
                "trial": trial.number,
                "state": trial.state.name,
                "objective": _csv_float(trial.value),
                "loss_median": _csv_float(kpis.get("loss_median", attrs.get("loss_median"))),
                "loss_p95": _csv_float(kpis.get("loss_p95", attrs.get("loss_p95"))),
                "trained_epochs": attrs.get("trained_epochs", result.get("trained_epochs", "")),
                "stop_reason": attrs.get("stop_reason", result.get("stop_reason", "")),
                "duration": format_duration(trial.duration.total_seconds())
                if trial.duration is not None
                else "",
                "seed": attrs.get("training_seed", run.get("seed", "")),
                "warmstart": attrs.get("warmstart", False),
                "run": run_name,
                "source_study": attrs.get("source_study", ""),
                "source_trial": attrs.get("source_trial", ""),
                "source_run": attrs.get("source_run", ""),
                "failure_type": attrs.get("failure_type", ""),
                "failure_message": attrs.get("failure_message", ""),
                **{f"param_{name}": _csv_value(trial.params.get(name)) for name in param_names},
            }
            writer.writerow(row)
    temporary.replace(path)


def _csv_float(value: object) -> str:
    return f"{float(value):.4e}" if value is not None else ""


def _trial_run_dir(hpo_benchmark_dir: Path, run: object) -> Path:
    """Resolve a local HPO stem or an absolute corrector artifact path."""
    path = Path(str(run))
    return path if path.is_absolute() else hpo_benchmark_dir / path


def _csv_value(value: object) -> object:
    if isinstance(value, float):
        return _csv_float(value)
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, separators=(",", ":"))
    return "" if value is None else value


def _save_top_configs(
    results: list[tuple[HyperParams, float]],
    study: optuna.Study,
    hpo_benchmark_dir: Path,
) -> None:
    ranked_trials = sorted(
        (
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and not trial.user_attrs.get("warmstart", False)
        ),
        key=lambda trial: trial.value,
    )[: len(results)]
    output_file = hpo_benchmark_dir / "top_trials.json"
    output_file.write_text(
        json.dumps(
            {
                "study_name": study.study_name,
                "n_trials": len(study.trials),
                "n_completed": sum(
                    t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
                ),
                "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                "best_loss": results[0][1] if results else None,
                "top_k": [
                    {
                        "rank": rank,
                        "loss": loss,
                        "config": hp.to_dict(),
                        **(
                            {
                                "corrector_scale": ranked_trials[rank - 1].user_attrs[
                                    "corrector_scale"
                                ]
                            }
                            if "corrector_scale" in ranked_trials[rank - 1].user_attrs
                            else {}
                        ),
                    }
                    for rank, (hp, loss) in enumerate(results, 1)
                ],
            },
            indent=2,
        )
    )
    logger.info(f"Top configurations saved to {output_file}")


def _bundle_top_artifacts(trials: list[optuna.trial.FrozenTrial], hpo_benchmark_dir: Path) -> None:
    """Copy lightweight artifacts for the ranked trials into top_k/."""
    bundle_dir = hpo_benchmark_dir / "top_k"
    shutil.rmtree(bundle_dir, ignore_errors=True)
    bundle_dir.mkdir()
    manifest = []
    study_dir = hpo_benchmark_dir.resolve()
    for rank, trial in enumerate(trials, 1):
        run = trial.user_attrs.get("run")
        if run is None:
            continue
        source = _trial_run_dir(hpo_benchmark_dir, run).resolve()
        if not source.is_relative_to(study_dir) and NEURAL_CORRECTOR_DIR not in source.parts:
            raise ValueError(f"Trial {trial.number} has invalid run path {run!r}")
        if not (source / "network.flax").exists():
            continue
        destination = bundle_dir / f"{rank:02d}_{source.name}"
        destination.mkdir()
        artifacts = []
        for artifact in source.iterdir():
            if artifact.is_file() and artifact.name != "network.flax":
                shutil.copy2(artifact, destination / artifact.name)
                artifacts.append(artifact.name)
        manifest.append(
            {
                "rank": rank,
                "trial": trial.number,
                "value": trial.value,
                "source": run,
                "artifacts": sorted(artifacts),
            }
        )
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _generate_benchmark_report(
    trials: list[optuna.trial.FrozenTrial], hpo_benchmark_dir: Path
) -> None:
    """Render the LaTeX benchmark PDF for the ranked trials.

    Failures do not propagate -- a broken report must not fail a finished study.
    """
    output_path = hpo_benchmark_dir / "benchmark_report.pdf"
    output_path.unlink(missing_ok=True)
    run_dirs = [
        _trial_run_dir(hpo_benchmark_dir, t.user_attrs["run"])
        for t in trials
        if (
            _trial_run_dir(hpo_benchmark_dir, t.user_attrs.get("run", "")) / "network.flax"
        ).exists()
    ]
    if not run_dirs:
        return
    try:
        from src.engine.benchmark_report import generate_report

        generate_report(run_dirs, output_path=output_path)
    except Exception as e:
        logger.warning(f"Benchmark report generation failed: {e}")


def build_sampler(study: StudyConfig) -> BaseSampler:
    """GPSampler (GP regression, logEI acquisition by default) when every suggestable
    axis is a continuous Range; TPE otherwise (categorical axes like hidden_dims)."""
    search_space = study.search_space
    axes = list(search_space.get_suggestable_params().values())
    # Corrector studies additionally suggest corrector_scale (see objective()).
    if study.foundation_path is not None and isinstance(search_space.corrector_scale, list | Range):
        axes.append(search_space.corrector_scale)
    if axes and all(isinstance(spec, Range) for spec in axes):
        return GPSampler(seed=42, n_startup_trials=study.n_startup_trials)
    return TPESampler(seed=42, n_startup_trials=study.n_startup_trials)


def run_optimization(
    study: StudyConfig,
    *,
    display: OptunaProgressDisplay,
    restart: bool = False,
    trial_callback: Callable[[optuna.Study, optuna.trial.FrozenTrial], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> list[tuple[HyperParams, float]]:
    """Run or resume a study and return its best configurations."""
    if study.foundation_path is not None and (
        study.warmstart_experiment_db is not None or study.warmstart_config_paths
    ):
        raise ValueError("Corrector studies cannot configure warmstart sources")
    # Freeze the commit once for the whole run: a commit made mid-study would
    # otherwise move study.db / optuna.log / trials.csv to a different path
    # and split the study across two commit dirs.
    commit = current_commit()
    hpo_benchmark_dir = study_dir(study.study_name, commit)
    foundation = None
    foundation_dir = None
    if study.foundation_path is not None:
        foundation, _, foundation_dir = load_foundation(
            study.foundation_path,
            soft_bc=study.search_space.soft_bc,
        )
    file_handler = logging.FileHandler(
        hpo_benchmark_dir / "optuna.log", mode="w" if restart else "a"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    # Warmstart candidates from two sources, both optional:
    #   * the benchmark db -- raw trials read back from each run dir's
    #     run.json config + KPIs, fused score recomputed.
    #   * the legacy configured-path ledger -- trials carry their own
    #     config + pre-fused value; optuna_warmstart() may retrain budget-stale
    #     configs, so it runs before check_capacity.
    benchmark_study = load_experiment_db(study.warmstart_experiment_db)
    candidates = _get_experiment_db_configs(
        benchmark_study, study.warmstart_experiment_db, study.score_beta
    )
    legacy_ledger = optuna_warmstart(study)
    candidates += _get_experiment_db_configs(legacy_ledger, None, study.score_beta)
    check_capacity(study, foundation)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_path = hpo_benchmark_dir / "study.db"
    if restart:
        _reset_study(study.study_name, storage_path)
        _discard_study_artifacts(hpo_benchmark_dir, foundation_dir)
        _write_objective_metadata(
            hpo_benchmark_dir,
            score_beta=study.score_beta,
            n_validate=study.n_validate,
            points_per_config=KPI_POINTS_PER_CONFIG,
            overwrite=True,
        )

    def create_study(load_if_exists: bool) -> optuna.Study:
        return optuna.create_study(
            study_name=study.study_name,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=load_if_exists,
            direction="minimize",
            sampler=build_sampler(study),
            pruner=(
                HyperbandPruner(
                    min_resource=study.min_epochs or study.total_epochs // 8,
                    max_resource=study.total_epochs,
                    reduction_factor=3,
                )
                if study.prune_trials
                else NopPruner()
            ),
        )

    discarded_failed_only_study = False
    try:
        optuna_study = create_study(load_if_exists=not restart)
        _validate_foundation_identity(optuna_study, foundation_dir)
        # Reap zombie RUNNING trials left by a crashed/interrupted prior run --
        # they'd otherwise block the budget forever (Optuna never auto-stales
        # them for sqlite).
        for t in optuna_study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
            logger.info(f"Reaping zombie trial {t.number} (RUNNING from a prior run)")
            optuna_study.tell(trial=t.number, values=None, state=optuna.trial.TrialState.FAIL)

        # If the study has no real results (only failed zombies / warmstart from
        # an interrupted prior run), discard it and start fresh. This keeps
        # abort→restart cycles clean: no accumulating failed trials, no
        # ever-incrementing trial numbers. Warmstart comes from the benchmark db,
        # not this dir, so nothing is lost.
        has_real_results = any(
            t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
            and not t.user_attrs.get("warmstart", False)
            for t in optuna_study.trials
        )
        if not restart and not has_real_results and optuna_study.trials:
            logger.info("Study has no completed trials -- discarding for a fresh start")
            del optuna_study
            _reset_study(study.study_name, storage_path)
            _discard_study_artifacts(hpo_benchmark_dir, foundation_dir)
            optuna_study = create_study(load_if_exists=False)
            _validate_foundation_identity(optuna_study, foundation_dir)
            discarded_failed_only_study = True

        if not restart:
            _write_objective_metadata(
                hpo_benchmark_dir,
                score_beta=study.score_beta,
                n_validate=study.n_validate,
                points_per_config=KPI_POINTS_PER_CONFIG,
                overwrite=discarded_failed_only_study,
            )

        # Inject warmstart candidates once (dedup by the warmstart user_attr, so
        # a resume of an already-warmstarted study skips re-injecting).
        already_warmstarted = any(t.user_attrs.get("warmstart", False) for t in optuna_study.trials)
        if candidates and not already_warmstarted:
            _inject_historical_trials(optuna_study, study.search_space, candidates)
        # Always seed the display with warmstart candidates -- on a resume the
        # injection is skipped (already in the db) but the TUI was just started
        # and its trials table is empty.
        if candidates:
            display.add_warmstart_trials(candidates)

        # Only completed/pruned trials of *this* study count against the budget
        # -- failed trials (incl. reaped zombies) produced no result, and
        # warmstart trials inform TPE but don't consume n_trials.
        prior_trials = sum(
            t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
            and not t.user_attrs.get("warmstart", False)
            for t in optuna_study.trials
        )
        warmstart_trials = sum(t.user_attrs.get("warmstart", False) for t in optuna_study.trials)
        display._prior_trials = prior_trials
        display._warmstart_trials = warmstart_trials
        callbacks = [lambda current, _trial: _write_trials_csv(current, hpo_benchmark_dir)]
        if trial_callback is not None:
            callbacks.append(trial_callback)

        optuna_study.optimize(
            lambda trial: objective(
                trial,
                study,
                display,
                hpo_benchmark_dir,
                foundation,
                foundation_dir,
                cancel_requested,
            ),
            n_trials=max(0, study.n_trials - prior_trials),
            catch=(Exception,),
            callbacks=callbacks,
        )

        complete = sorted(
            (
                t
                for t in optuna_study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None
                and not t.user_attrs.get("warmstart", False)
            ),
            key=lambda trial: trial.value,
        )[: study.top_k]
        results = [
            (build_hyperparams(trial, study.search_space), trial.value) for trial in complete
        ]
        for rank, (hp, loss) in enumerate(results, 1):
            architecture = f"{len(hp.hidden_dims)}x{hp.hidden_dims[0]}"
            logger.info(
                f"Rank {rank}: loss={loss:.6f}, architecture={architecture}, "
                f"lr={hp.learning_rate_max:.2e}"
            )
        _save_top_configs(results, optuna_study, hpo_benchmark_dir)
        _write_trials_csv(optuna_study, hpo_benchmark_dir)
        if study.checkpoint_policy != "none":
            _bundle_top_artifacts(complete, hpo_benchmark_dir)
            _generate_benchmark_report(complete, hpo_benchmark_dir)
        return results
    except KeyboardInterrupt:
        logger.info("Interrupted by user -- completed trial artifacts preserved")
        raise


def resolve_retrain_choice(study: StudyConfig) -> bool:
    """Whether to retrain budget-mismatched warmstart configs.

    Returns False immediately if no configured paths have a mismatched budget.
    Raises on non-tty when mismatches exist -- the destructive choice cannot be silently guessed.
    """
    # Retrain only applies to the legacy configured-path ledger (the
    # benchmark-db path injects already-trained trials as-is). Nothing to
    # reconcile if that path is off -- return before touching the fail-fast
    # load_configs, whose paths need not exist then.
    if study.warmstart_experiment_db is None or not study.warmstart_config_paths:
        return False
    mismatched = [
        path
        for path, hp in load_configs(study.warmstart_config_paths)
        if study.search_space.budget_mismatch(hp)
    ]
    if not mismatched:
        return False
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Warmstart configs with a mismatched training budget were found: "
            f"{[str(p.parent) for p in mismatched]}. Pass --retrain or --no-retrain "
            "explicitly -- non-interactive launches can't be prompted."
        )
    print("\nWarmstart configs with a mismatched training budget were found:")
    for path in mismatched:
        print(f"  {path.parent}")
    return input("Retrain them at this study's budget? (y/n): ").strip().lower().startswith("y")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for PINN")
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--reset-sqlite",
        action="store_true",
        help="Wipe this study's existing study.db before starting",
    )
    reset_group.add_argument(
        "--resume-sqlite",
        action="store_true",
        help="Resume this study's existing study.db without prompting",
    )
    retrain_group = parser.add_mutually_exclusive_group()
    retrain_group.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain warmstart configs whose training budget mismatches this study",
    )
    retrain_group.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip (don't inject) warmstart configs whose training budget mismatches this study",
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--full-trials", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    args = parser.parse_args()

    # CLI default study: LR/wd/sigma neighborhood of the incumbent, arch searched.
    search_space = SearchSpaceConfig(
        learning_rate_max=Range(5e-4, 1e-3, log=True),
        learning_rate_min=1e-5,
        weight_decay=Range(1e-9, 1e-8, log=True),
        sigma_residual_adaptive_sampling=Range(0.02, 0.05),
        hidden_dims=[(320,) * 5, (256,) * 6, (200,) * 5],
        rwf=False,
        soft_bc=False,
    )
    study = StudyConfig(search_space=search_space)
    study.prune_trials = not args.full_trials
    study.checkpoint_policy = "all" if args.save_all else "top_k"

    if args.test:
        # Quick run for smoke-testing
        search_space.batch_size = 32
        search_space.n_train = 64
        search_space.n_rz_boundary_samples = 16
        search_space.n_rz_inner_samples = 64
        search_space.hidden_dims = [(32, 32), (64, 64)]
        search_space.warmup_epochs = 50
        search_space.decay_epochs = 50
        study.n_validate = 16
        study.n_trials = 10
        study.min_epochs = 100  # = total_epochs: a single Hyperband rung
        study.n_startup_trials = 2
        study.checkpoint_policy = "none"
        study.study_name = f"{study.study_name}_test"

    if args.test:
        reset = True  # tests always start clean; never prompt
    elif args.reset_sqlite:
        reset = True
    elif args.resume_sqlite:
        reset = False
    else:
        commit = current_commit()

        # check if study database already exists, if yes prompt the user about reset or continue
        reset = resolve_reset_choice(
            study_dir(study.study_name, commit) / "study.db",
            study.study_name,
            commit,
        )

    if args.test:
        study.retrain = False  # synthetic test search space; nothing to reconcile
    elif args.retrain:
        study.retrain = True
    elif args.no_retrain:
        study.retrain = False
    else:
        # neither flag -> prompt, but only if a warmstart config's budget actually mismatches
        study.retrain = resolve_retrain_choice(study)

    # TUI whenever a terminal exists; piped/nohup output gets the plain Live dashboard,
    # which degrades to sequential text on non-ttys. All logging/benchmark artifacts
    # (study.db, optuna.log, trials.csv, checkpoints) are identical either way.
    if sys.stdout.isatty():
        app = HpoApp(study, reset=reset)
        app.run()
        if app.crash_traceback is not None:
            # Textual's alternate screen is gone now; print to the real terminal
            # and exit non-zero, matching the non-tty path so driver watchers fire.
            print(app.crash_traceback, file=sys.stderr)
            sys.exit(1)
    else:
        with OptunaProgressDisplay(study) as display:
            run_optimization(study, display=display, restart=reset)


if __name__ == "__main__":
    # Run-by-path loads this file twice (here as __main__, again as the qualified
    # module when HpoApp imports it), duplicating Range et al. Call the canonical
    # main so cross-module isinstance() checks see one copy of each class.
    from src.engine.optimize_network_optuna import main

    main()
