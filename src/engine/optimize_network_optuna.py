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
    │  WARMSTART_CONFIG_PATHS├──┤
    └────────────────────────┘  │ inject as completed trials
                                │
    ┌───────────────────────────▼──────────────────────┐
    │  Optuna study  (TPE sampler + Hyperband pruner)  │
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
    │    study.db  trials.json  top_trials.json        │
    │    benchmark_report.pdf  top-k checkpoints       │
    └──────────────────────────────────────────────────┘

Entry point: main(). Launches HpoApp (TUI) when stdout is a tty, plain
Live dashboard otherwise. Configuration: SearchSpaceConfig, StudyConfig.
"""

import argparse
import ast
import json
import logging
import os
import shutil
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import jax
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.pruners import HyperbandPruner, NopPruner
from optuna.samplers import TPESampler

from src.engine.model_evaluation import evaluate_validation_loss_stats
from src.engine.network import FoundationModel, NetworkManager
from src.engine.residual_correction import load_foundation
from src.lib.config import KPI_EVAL_CONFIGS, KPI_POINTS_PER_CONFIG, Filepaths, current_commit
from src.lib.network_config import HyperParams
from src.lib.optuna_tui import HpoApp, OptunaProgressDisplay, logger, resolve_reset_choice

HPO_ROOT = Filepaths.DATA / "hpo"

# Architectures (width * depth) to search over. Hand-ordered biggest-first:
# check_capacity() smoke-trains the first entry as the worst case.
_ARCHITECTURES = [
    (320,) * 5,
    (256,) * 6,
    (200,) * 5,  # incumbent
]

WARMSTART_CONFIG_PATHS: list[Path] = [
    #    Filepaths.BENCHMARKS
    #    / "model_selection_benchmark"
    #    / "pinn_2026_07_13_01_18_41_hard-bc-unoptimized"
    #    / "config.json",
    #    Filepaths.BENCHMARKS
    #    / "model_selection_benchmark"
    #    / "pinn_2026_07_13_06_59_27_hard-bc-tuned-schedule"
    #    / "config.json",
    #    Filepaths.BENCHMARKS
    #    / "model_selection_benchmark"
    #    / "pinn_2026_07_13_08_32_59_hard-bc-final"
    #    / "config.json",
]


# Benchmark run dir whose Optuna study.db warms up a fresh study. Its completed
# trials are read back from each trial's run dir (config.json + kpis.json) and
# injected as historical evidence for TPE -- no retraining. None disables it.
# The study name is auto-discovered (the non-legacy study with the most trials).
WARMSTART_EXPERIMENT_DB: Path | None = (
    # None
    Path("data/hpo/2026_07_16_00_16_02_arch_wide_or_deep_2400ep_4a25aff/")
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

    Fields mirror HyperParams except for the corrector's stage2_scale.
    """

    hidden_dims: tuple[int, ...] | list[tuple[int, ...]] = field(
        default_factory=_ARCHITECTURES.copy
    )
    learning_rate_max: Range = Range(5e-4, 1e-3, log=True)
    learning_rate_min: float = 1e-5
    weight_decay: Range = Range(1e-9, 1e-8, log=True)
    sigma_residual_adaptive_sampling: Range = Range(0.02, 0.05)
    weight_boundary_condition: float = 10.0
    weight_flux_scale: float = 10.0
    soft_bc: bool = False
    n_rz_inner_samples: int = 512
    n_rz_boundary_samples: int = 256
    batch_size: int = 32
    n_train: int = 1024
    warmup_epochs: int = 400
    decay_epochs: int = 2000
    huber_delta: float = 0.0
    n_fourier_features: int = 0
    lbfgs_steps: int = 0
    stage2_scale: float | list[float] | Range = 0.01

    def get_static_params(self) -> dict[str, Any]:
        """Fields pinned to a single value -- fed straight into HyperParams(**...)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "stage2_scale" and not isinstance(getattr(self, f.name), list | Range)
        }

    def get_suggestable_params(self) -> dict[str, list | Range]:
        """Fields Optuna searches: name -> its list of choices or its Range."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "stage2_scale" and isinstance(getattr(self, f.name), list | Range)
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


_DEFAULT_TOTAL_EPOCHS = SearchSpaceConfig.warmup_epochs + SearchSpaceConfig.decay_epochs


@dataclass
class StudyConfig:
    """Orchestration knobs for the Optuna study itself -- never fed to HyperParams."""

    study_name: str = "arch_wide_or_deep_2400ep"
    n_trials: int = 20
    top_k: int = 3
    n_startup_trials: int = 10
    n_validate: int = KPI_EVAL_CONFIGS
    min_epochs: int = _DEFAULT_TOTAL_EPOCHS // 8
    total_epochs: int = _DEFAULT_TOTAL_EPOCHS
    prune_trials: bool = True
    # "none": no checkpoints/benchmark dirs (--test). "top_k": completed trials save
    # post-hoc. "all": every trial saves during training and retains its train.log.
    checkpoint_policy: Literal["none", "top_k", "all"] = "top_k"
    # Retrain (at this study's budget) or skip warmstart configs with a mismatched budget.
    # Resolved in main(); lives on `study` so it survives the HpoApp path unchanged.
    retrain: bool = False
    # Weight on the p95 tail in the fused ranking score
    # ``loss_median + score_beta * loss_p95`` (both terms are |R_GS|). ~0.2-0.5
    # trades typical vs. worst-case error; p95 ~10x median keeps the tail modest.
    score_beta: float = 0.3
    # Inject the WARMSTART_EXPERIMENT_DB benchmark trials as historical evidence.
    # Disabled in --test: a smoke test's tiny budget is incomparable, and injected
    # trials would consume the small n_trials budget.
    warmstart: bool = True
    # Frozen stage-1 checkpoint used by every trial in a corrector study.
    stage1_run: str | None = None


def _record_foundation(study_path: Path, stage1_run: str | None, source: Path | None) -> None:
    """Persist the foundation identity and a self-contained checkpoint copy."""
    path = study_path / "foundation.json"
    if path.exists():
        saved = json.loads(path.read_text()).get("stage1_run")
        if saved != stage1_run:
            raise ValueError(
                f"Study foundation is {saved!r}, but this run requested {stage1_run!r}"
            )
    if source is not None:
        destination = study_path / "_foundation"
        destination.mkdir(exist_ok=True)
        if source.resolve() != destination.resolve():
            for artifact in ("config.json", "network.flax"):
                shutil.copy2(source / artifact, destination / artifact)
    if not path.exists() and stage1_run is not None:
        path.write_text(json.dumps({"stage1_run": stage1_run}, indent=2) + "\n")


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
        "evaluator_commit": current_commit(),
    }
    if path.exists() and not overwrite and json.loads(path.read_text()) != metadata:
        raise ValueError(f"Study objective protocol does not match {path}")
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def study_dir(study_name: str, commit: str | None = None) -> Path:
    """
    Storage root for one Optuna study under data/hpo/.

        data/hpo/<timestamp>_<study_name>_<commit>/

    An existing directory for the same name and commit is reused (resume);
    otherwise a fresh timestamped one is created.
    """
    commit = commit or current_commit()
    # ponytail: >1 match (two runs of the same name+commit) picks the newest;
    # that's the one a bare re-run means to resume.
    existing = sorted(HPO_ROOT.glob(f"*_{study_name}_{commit}"))
    path = existing[-1] if existing else HPO_ROOT / f"{_now()}_{study_name}_{commit}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def load_configs(paths: list[Path] = WARMSTART_CONFIG_PATHS) -> list[tuple[Path, HyperParams]]:
    # Fail-fast by design: a wrong/missing WARMSTART_CONFIG_PATHS entry SHALL crash --
    # the caller supplies correct paths; a silently skipped warmstart config is worse.
    return [(path, HyperParams.from_json(str(path))) for path in paths]


def load_experiment_db(db_dir: Path | None = WARMSTART_EXPERIMENT_DB) -> optuna.Study | None:
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


def optuna_warmstart(
    search_space: SearchSpaceConfig, retrain: bool, score_beta: float = 0.3
) -> optuna.Study | None:
    """Populate the legacy ``warmstart_experiments`` ledger from WARMSTART_CONFIG_PATHS.

    No-op returning None if the list is empty or no benchmark db is configured.
    Budget-mismatched entries are retrained or skipped depending on ``retrain``.
    """
    if WARMSTART_EXPERIMENT_DB is None or not WARMSTART_CONFIG_PATHS:
        return None

    db = optuna.create_study(
        study_name="warmstart_experiments",
        storage=f"sqlite:///{WARMSTART_EXPERIMENT_DB / 'study.db'}",
        load_if_exists=True,
        direction="minimize",
    )

    for path, hp in load_configs(WARMSTART_CONFIG_PATHS):
        mismatched = search_space.budget_mismatch(hp)

        if mismatched:
            if not retrain:
                continue  # skip: never enters the ledger, never informs TPE
            hp = _reevaluate_config(path, hp, mismatched)

        kpis = json.loads((path.parent / "kpis.json").read_text())
        loss = kpis["loss_median"] + score_beta * kpis["loss_p95"]

        db.add_trial(
            optuna.trial.create_trial(
                value=loss,
                user_attrs={"config": hp.to_dict(), "run": str(path.parent)},
            )
        )
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
    for t in db.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        if t.value is None:
            continue
        if "config" in t.user_attrs:
            candidates.append((HyperParams.from_dict(t.user_attrs["config"]), t.value, None, None))
        elif db_dir is not None and "run" in t.user_attrs:
            run_dir = db_dir / t.user_attrs["run"]
            hp = HyperParams.from_json(str(run_dir / "config.json"))
            kpis = json.loads((run_dir / "kpis.json").read_text())
            fused = kpis["loss_median"] + score_beta * kpis["loss_p95"]
            candidates.append((hp, fused, kpis["loss_median"], kpis["loss_p95"]))
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


def check_capacity(
    search_space: SearchSpaceConfig,
    study: StudyConfig,
    foundation: FoundationModel | None = None,
) -> None:
    """Verify the search space's most resource-demanding config trains without crashing."""
    suggestable = search_space.get_suggestable_params()
    picked = {
        # Each spec's most demanding end: Range.high, or a list's first entry
        # (lists like _ARCHITECTURE_GRID are hand-ordered biggest-first).
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
    search_space: SearchSpaceConfig,
    study: StudyConfig,
    display: OptunaProgressDisplay,
    hpo_benchmark_dir: Path,
    foundation: FoundationModel | None = None,
) -> float:
    """Train one network and return its fused ranking score (median + beta*p95 of |R_GS|).

    Per-epoch Hyperband pruning uses val_kpi_median (median |R_GS| at the global
    protocol: KPI_POINTS_PER_CONFIG x fixed validation configs), not the composite
    training loss.  Pruning and ranking therefore measure the same quantity.
    """
    hp = build_hyperparams(trial, search_space)
    stage2_scale = 1.0
    if foundation is not None:
        scale_spec = search_space.stage2_scale
        stage2_scale = (
            _suggest(trial, "stage2_scale", scale_spec)
            if isinstance(scale_spec, list | Range)
            else scale_spec
        )
        trial.set_user_attr("stage2_scale", stage2_scale)
    display_params = {
        "depth": len(hp.hidden_dims),
        "width": hp.hidden_dims[0],
        "lr_max": hp.learning_rate_max,
        "lr_min": hp.learning_rate_min,
        "wd": hp.weight_decay,
        "sig": hp.sigma_residual_adaptive_sampling,
        "eps": stage2_scale if foundation is not None else None,
    }
    manager = NetworkManager(
        hp,
        n_validation_size=study.n_validate,
        test_mode=study.checkpoint_policy == "none",
        output_dir=hpo_benchmark_dir,
        prior=foundation,
        scale=stage2_scale,
    )
    total_epochs = hp.warmup_epochs + hp.decay_epochs
    display.start_trial(trial.number + 1, display_params, total_epochs)
    display.current_manager = manager
    manager.metrics_row_sink = display.add_metrics_row
    last_epoch = 0

    def report(epoch: int, val_kpi_median: float | None) -> None:
        # val_kpi_median is median |R_GS| at KPI_POINTS_PER_CONFIG from NetworkManager.
        nonlocal last_epoch
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
        # Ranking metric: fused median + beta*p95 of |R_GS| over validation
        # configs (see evaluate_validation_loss_stats/score). Median alone ignores
        # worst-case error; max is a high-variance single-point estimator.
        # Stats are computed once and shown to the user alongside the fused score.
        val_median, val_p95 = evaluate_validation_loss_stats(manager)
        val_score = val_median + study.score_beta * val_p95
        if study.checkpoint_policy == "top_k":
            manager.to_disk()
        display.update(
            trial.number + 1,
            display_params,
            val_score,
            "done",
            total_epochs,
            median=val_median,
            p95=val_p95,
        )
        return val_score
    except optuna.TrialPruned:
        display.update(trial.number + 1, display_params, None, "pruned", last_epoch)
        raise
    except Exception:
        display.update(trial.number + 1, display_params, None, "failed", last_epoch)
        raise
    finally:
        if manager.artifact_stem is not None:
            trial.set_user_attr("run", manager.artifact_stem)
        # Only this trial's own run dir is touched; failed trials are also
        # dropped from trials.json (study.db keeps them for the sampler).
        manager.discard_unsaved_run()
        _write_trials_json(trial.study, hpo_benchmark_dir)
        jax.clear_caches()


def _write_trials_json(study: optuna.Study, hpo_benchmark_dir: Path) -> None:
    """Ledger of which runs/checkpoints belong to this study. Rewritten after every
    trial so an aborted study still leaves an on-disk record."""
    (hpo_benchmark_dir / "trials.json").write_text(
        json.dumps(
            [
                {
                    "trial": t.number,
                    "state": t.state.name,
                    "value": t.value,
                    **{key: value for key, value in t.user_attrs.items() if key != "checkpoint"},
                }
                for t in study.get_trials(deepcopy=False)
                if t.state != optuna.trial.TrialState.FAIL
            ],
            indent=2,
        )
    )


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
                            {"stage2_scale": ranked_trials[rank - 1].user_attrs["stage2_scale"]}
                            if "stage2_scale" in ranked_trials[rank - 1].user_attrs
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
        source = (hpo_benchmark_dir / run).resolve()
        if source.parent != study_dir:
            raise ValueError(f"Trial {trial.number} has invalid run path {run!r}")
        if not (source / "network.flax").exists():
            continue
        destination = bundle_dir / f"{rank:02d}_{run}"
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
        hpo_benchmark_dir / t.user_attrs["run"]
        for t in trials
        if (hpo_benchmark_dir / t.user_attrs.get("run", "") / "network.flax").exists()
    ]
    if not run_dirs:
        return
    try:
        from src.engine.benchmark_report import generate_report

        generate_report(run_dirs, output_path=output_path)
    except Exception as e:
        logger.warning(f"Benchmark report generation failed: {e}")


def run_optimization(
    search_space: SearchSpaceConfig | None = None,
    study: StudyConfig | None = None,
    *,
    display: OptunaProgressDisplay,
    restart: bool = False,
) -> list[tuple[HyperParams, float]]:
    """Run or resume a study and return its best configurations."""
    search_space = search_space or SearchSpaceConfig()
    study = study or StudyConfig()
    if study.stage1_run is not None and study.warmstart:
        raise ValueError("Corrector studies require StudyConfig(warmstart=False)")
    # Freeze the commit once for the whole run: a commit made mid-study would
    # otherwise move study.db / optuna.log / trials.json to a different path
    # and split the study across two commit dirs.
    commit = current_commit()
    hpo_benchmark_dir = study_dir(study.study_name, commit)
    if not restart:
        _write_objective_metadata(
            hpo_benchmark_dir,
            score_beta=study.score_beta,
            n_validate=study.n_validate,
            points_per_config=KPI_POINTS_PER_CONFIG,
        )
    foundation = None
    foundation_source = None
    if study.stage1_run is not None:
        local_foundation = hpo_benchmark_dir / "_foundation"
        local_complete = all(
            (local_foundation / artifact).exists() for artifact in ("config.json", "network.flax")
        )
        foundation, _, foundation_source = load_foundation(
            local_foundation if local_complete else study.stage1_run,
            soft_bc=search_space.soft_bc,
        )
    _record_foundation(hpo_benchmark_dir, study.stage1_run, foundation_source)
    file_handler = logging.FileHandler(
        hpo_benchmark_dir / "optuna.log", mode="w" if restart else "a"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    # Warmstart candidates from two sources, both optional:
    #   * the benchmark db -- raw trials read back from each run dir's
    #     config.json + kpis.json, fused score recomputed.
    #   * the legacy WARMSTART_CONFIG_PATHS ledger -- trials carry their own
    #     config + pre-fused value; optuna_warmstart() may retrain budget-stale
    #     configs, so it runs before check_capacity.
    benchmark_study = load_experiment_db(WARMSTART_EXPERIMENT_DB) if study.warmstart else None
    candidates = _get_experiment_db_configs(
        benchmark_study, WARMSTART_EXPERIMENT_DB, study.score_beta
    )
    legacy_ledger = optuna_warmstart(
        search_space, retrain=study.retrain, score_beta=study.score_beta
    )
    candidates += _get_experiment_db_configs(legacy_ledger, None, study.score_beta)
    check_capacity(search_space, study, foundation)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_path = hpo_benchmark_dir / "study.db"
    if restart:
        storage_path.unlink(missing_ok=True)
        for run_dir in hpo_benchmark_dir.glob("pinn_*"):
            if run_dir.is_dir():
                shutil.rmtree(run_dir)
        for artifact in ("trials.json", "top_trials.json", "benchmark_report.pdf"):
            (hpo_benchmark_dir / artifact).unlink(missing_ok=True)
        shutil.rmtree(hpo_benchmark_dir / "top_k", ignore_errors=True)
        _write_objective_metadata(
            hpo_benchmark_dir,
            score_beta=study.score_beta,
            n_validate=study.n_validate,
            points_per_config=KPI_POINTS_PER_CONFIG,
            overwrite=True,
        )

    try:
        optuna_study = optuna.create_study(
            study_name=study.study_name,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=not restart,
            direction="minimize",
            sampler=TPESampler(seed=42, n_startup_trials=study.n_startup_trials),
            pruner=(
                HyperbandPruner(
                    min_resource=study.min_epochs,
                    max_resource=study.total_epochs,
                    reduction_factor=3,
                )
                if study.prune_trials
                else NopPruner()
            ),
        )
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
            storage_path.unlink()
            optuna_study = optuna.create_study(
                study_name=study.study_name,
                storage=f"sqlite:///{storage_path}",
                direction="minimize",
                sampler=TPESampler(seed=42, n_startup_trials=study.n_startup_trials),
                pruner=(
                    HyperbandPruner(
                        min_resource=study.min_epochs,
                        max_resource=study.total_epochs,
                        reduction_factor=3,
                    )
                    if study.prune_trials
                    else NopPruner()
                ),
            )

        # Inject warmstart candidates once (dedup by the warmstart user_attr, so
        # a resume of an already-warmstarted study skips re-injecting).
        already_warmstarted = any(t.user_attrs.get("warmstart", False) for t in optuna_study.trials)
        if candidates and not already_warmstarted:
            _inject_historical_trials(optuna_study, search_space, candidates)
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
        optuna_study.optimize(
            lambda trial: objective(
                trial, search_space, study, display, hpo_benchmark_dir, foundation
            ),
            n_trials=max(0, study.n_trials - prior_trials),
            catch=(Exception,),
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
        results = [(build_hyperparams(trial, search_space), trial.value) for trial in complete]
        for rank, (hp, loss) in enumerate(results, 1):
            architecture = f"{len(hp.hidden_dims)}x{hp.hidden_dims[0]}"
            logger.info(
                f"Rank {rank}: loss={loss:.6f}, architecture={architecture}, "
                f"lr={hp.learning_rate_max:.2e}"
            )
        _save_top_configs(results, optuna_study, hpo_benchmark_dir)
        _write_trials_json(optuna_study, hpo_benchmark_dir)
        if study.checkpoint_policy != "none":
            _bundle_top_artifacts(complete, hpo_benchmark_dir)
            _generate_benchmark_report(complete, hpo_benchmark_dir)
        return results
    except KeyboardInterrupt:
        logger.info("Interrupted by user -- completed trial artifacts preserved")
        raise


def resolve_retrain_choice(search_space: SearchSpaceConfig) -> bool:
    """Whether to retrain budget-mismatched warmstart configs.

    Returns False immediately if there are no mismatches or warmstart is disabled.
    Raises on non-tty when mismatches exist -- the destructive choice cannot be silently guessed.
    """
    # Retrain only applies to the legacy WARMSTART_CONFIG_PATHS path (the
    # benchmark-db path injects already-trained trials as-is). Nothing to
    # reconcile if that path is off -- return before touching the fail-fast
    # load_configs, whose paths need not exist then.
    if WARMSTART_EXPERIMENT_DB is None or not WARMSTART_CONFIG_PATHS:
        return False
    mismatched = [
        path
        for path, hp in load_configs(WARMSTART_CONFIG_PATHS)
        if search_space.budget_mismatch(hp)
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

    search_space = SearchSpaceConfig()
    study = StudyConfig()
    study.prune_trials = not args.full_trials
    study.checkpoint_policy = "all" if args.save_all else "top_k"

    if args.test:
        # Quick run for smoke-testing
        search_space.batch_size = 8
        search_space.n_train = 64
        search_space.n_rz_boundary_samples = 16
        search_space.n_rz_inner_samples = 64
        search_space.hidden_dims = [(32,), (128,), (32, 32), (128, 128)]
        search_space.warmup_epochs = 5
        search_space.decay_epochs = 25
        study.n_validate = 16
        study.n_trials = 3
        study.total_epochs = 30
        study.min_epochs = 5
        study.n_startup_trials = 2
        study.checkpoint_policy = "none"
        study.study_name = f"{study.study_name}_test"
        study.warmstart = False

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
        study.retrain = resolve_retrain_choice(search_space)

    # TUI whenever a terminal exists; piped/nohup output gets the plain Live dashboard,
    # which degrades to sequential text on non-ttys. All logging/benchmark artifacts
    # (study.db, optuna.log, trials.json, train.log, checkpoints) are identical either way.
    if sys.stdout.isatty():
        app = HpoApp(search_space, study, restart=reset)
        app.run()
        if app.crash_traceback is not None:
            # Textual's alternate screen is gone now; print to the real terminal
            # and exit non-zero, matching the non-tty path so driver watchers fire.
            print(app.crash_traceback, file=sys.stderr)
            sys.exit(1)
    else:
        with OptunaProgressDisplay(study) as display:
            run_optimization(search_space, study, display=display, restart=reset)


if __name__ == "__main__":
    # Run-by-path loads this file twice (here as __main__, again as the qualified
    # module when HpoApp imports it), duplicating Range et al. Call the canonical
    # main so cross-module isinstance() checks see one copy of each class.
    from src.engine.optimize_network_optuna import main

    main()
