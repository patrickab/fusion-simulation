"""Optuna hyperparameter search using the standard network training path."""

import argparse
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
from optuna.pruners import HyperbandPruner, NopPruner
from optuna.samplers import TPESampler

from src.engine.network import NetworkManager
from src.lib.config import Filepaths, current_commit
from src.lib.network_config import HyperParams
from src.lib.optuna_tui import HpoApp, OptunaProgressDisplay, logger, resolve_reset_choice

HPO_ROOT = Filepaths.DATA / "hpo"

# Architectures (width * depth) to search over.
_ARCHITECTURE_GRID = [
    (200,) * 5,
    (160,) * 5,
    (128,) * 5,
    (200,) * 4,
    (160,) * 4,
    (128,) * 4,
]

# Hand-curated list of previous experiments for warm-start of experiment database.
WARMSTART_CONFIG_PATHS: list[Path] = [
    Filepaths.BENCHMARKS
    / "model_selection_benchmark"
    / "pinn_2026_07_13_01_18_41_hard-bc-unoptimized"
    / "config.json",
    Filepaths.BENCHMARKS
    / "model_selection_benchmark"
    / "pinn_2026_07_13_06_59_27_hard-bc-tuned-schedule"
    / "config.json",
    Filepaths.BENCHMARKS
    / "model_selection_benchmark"
    / "pinn_2026_07_13_08_32_59_hard-bc-final"
    / "config.json",
]

# Optuna storage URL for persistent experiment ledger, e.g.
# "sqlite:///data/hpo/experiments.db". "" -> in-memory only (still works for a
# single optuna_warmstart() call, just doesn't persist between processes).
WARMSTART_EXPERIMENT_DATABASE: str | None = None


@dataclass(frozen=True)
class Range:
    """Continuous, optionally log-scaled search axis."""

    low: float
    high: float
    log: bool = False


@dataclass
class SearchSpaceConfig:
    """
    Search-space for HyperParams.

    Dynamically & declaratively defines Optuna optimization run.
        - Scalar = pinned
        - list   = discrete choices,
        - Range  = continuous.

    To define a new searchable axis define a range or list of values.
    Field names mirror HyperParams.
    """

    hidden_dims: tuple[int, ...] | list[tuple[int, ...]] = field(
        default_factory=_ARCHITECTURE_GRID.copy
    )
    learning_rate_max: Range = Range(1e-3, 3e-3, log=True)
    learning_rate_min: Range = Range(1e-6, 2.4e-4, log=True)
    weight_decay: Range = Range(1e-8, 1e-4, log=True)
    sigma_residual_adaptive_sampling: Range = Range(0.01, 0.05)
    weight_boundary_condition: float = 10.0
    weight_flux_scale: float = 10.0
    soft_bc: bool = True
    n_rz_inner_samples: int = 512
    n_rz_boundary_samples: int = 128
    batch_size: int = 64
    n_train: int = 1024
    warmup_epochs: int = 100
    decay_epochs: int = 500
    huber_delta: float = 1.0
    n_fourier_features: int = 64
    lbfgs_steps: int = 0

    def get_static_params(self) -> dict[str, Any]:
        """Fields pinned to a single value -- fed straight into HyperParams(**...)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not isinstance(getattr(self, f.name), list | Range)
        }

    def get_suggestable_params(self) -> dict[str, list | Range]:
        """Fields Optuna searches: name -> its list of choices or its Range."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), list | Range)
        }

    def is_valid_config(self, hparams: HyperParams) -> bool:
        """Whether hp's values fall within every axis this search space searches."""
        for name, spec in self.get_suggestable_params().items():
            value = getattr(hparams, name)
            if isinstance(spec, Range):
                if not (spec.low <= value <= spec.high):
                    return False
            elif value not in spec:
                return False
        return True

    def is_static_parameter_mismatch(self, hparams: HyperParams) -> dict[str, Any]:
        """
        Check if hyperparameters disagree with static parameters.

        Used for warmstarting experiments to ensure fair comparison.
        """
        return {k: v for k, v in self.get_static_params().items() if getattr(hparams, k) != v}


_DEFAULT_TOTAL_EPOCHS = SearchSpaceConfig.warmup_epochs + SearchSpaceConfig.decay_epochs


@dataclass
class StudyConfig:
    """Orchestration knobs for the Optuna study itself -- never fed to HyperParams."""

    study_name: str = "pinn_hpo"
    n_trials: int = 5
    top_k: int = 3
    n_startup_trials: int = 10
    n_validate: int = 20
    min_epochs: int = _DEFAULT_TOTAL_EPOCHS // 2
    total_epochs: int = _DEFAULT_TOTAL_EPOCHS
    prune_trials: bool = True
    # "none": no checkpoints/benchmark dirs (--test). "top_k": only trials that rank in
    # the current top_k get saved (post-hoc). "all": every trial saves during training.
    checkpoint_policy: Literal["none", "top_k", "all"] = "top_k"


def study_dir(study_name: str, commit: str | None = None) -> Path:
    """
    Storage location for one optuna study:

        data / hpo / <timestamp>_<study_name>_<commit> /
            {study.db, optuna.log, top_trials.json, trials.json, pinn_<ts>/...}

    The timestamp prefix keeps studies alphanumerically sortable and gives each
    a stable id/name. An existing dir for the same name+commit is reused (so a
    study resumes into it); otherwise a fresh timestamped one is minted.
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
    return [(path, HyperParams.from_json(str(path))) for path in paths]


def load_experiment_db(
    search_space: SearchSpaceConfig, storage: str | None = WARMSTART_EXPERIMENT_DATABASE
) -> optuna.Study | None:
    if storage is None:
        return None

    db = optuna.create_study(
        study_name="warmstart_experiments",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
    )

    for trial in db.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        config = trial.user_attrs.get("config")
        hparams = config and HyperParams.from_dict(config)
        if hparams and (mismatches := search_space.is_static_parameter_mismatch(hparams)):
            raise ValueError(f"{trial.user_attrs.get('run')}: stale ledger entry {mismatches}")

    return db


def _reevaluate_config(path: Path, hp: HyperParams, static: dict[str, Any]) -> HyperParams:
    """
    Retrains config under search_space's static values
    Destructively replaces previous artifacts (network/plots/kpis)
    """
    corrected = HyperParams(**{**hp.to_dict(), **static})
    stale = sorted(static)
    logger.warning(f"{path.parent}: stale vs. current search space ({stale}) -- retraining")
    manager = NetworkManager(corrected)
    manager.train(save_to_disk=True)
    shutil.rmtree(path.parent)
    shutil.move(str(manager.run_dir()), str(path.parent))
    return corrected


def optuna_warmstart(search_space: SearchSpaceConfig) -> optuna.Study | None:
    """Populate the experiment database

    WARMSTART_CONFIG_PATHS: reuse kpis.json, if compliant with SearchSpaceConfig, else re-evaluate
    WARMSTART_EXPERIMENT_DATABASE: no-op (returns None) if unset, else reads from its path."""
    db = load_experiment_db(search_space, WARMSTART_EXPERIMENT_DATABASE)

    if db is None:
        return None

    for path, hp in load_configs(WARMSTART_CONFIG_PATHS):
        mismatched = search_space.is_static_parameter_mismatch(hp)

        if mismatched:
            hp = _reevaluate_config(path, hp, mismatched)

        loss = json.loads((path.parent / "kpis.json").read_text())["loss_median"]

        db.add_trial(
            optuna.trial.create_trial(
                value=loss,
                user_attrs={"config": hp.to_dict(), "run": str(path.parent)},
            )
        )
    return db


def _get_experiment_db_configs(db: optuna.Study | None) -> list[HyperParams]:
    """Every config recorded in a warmstart experiment database, or [] if there is none."""
    if db is None:
        return []
    return [
        HyperParams.from_dict(t.user_attrs["config"])
        for t in db.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if "config" in t.user_attrs
    ]


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
    label = trial.suggest_categorical(name, [str(choice) for choice in spec])
    return {str(choice): choice for choice in spec}[label]


def build_hyperparams(trial: optuna.Trial, search_space: SearchSpaceConfig) -> HyperParams:
    """Sample every searchable axis and merge with the pinned ones."""
    suggested = {
        name: _suggest(trial, name, spec)
        for name, spec in search_space.get_suggestable_params().items()
    }
    return HyperParams(**search_space.get_static_params(), **suggested)


def check_capacity(search_space: SearchSpaceConfig, study: StudyConfig) -> None:
    """Verify the search space's most resource-demanding config trains without crashing."""
    suggestable = search_space.get_suggestable_params()
    picked = {
        # Each spec's most demanding end: Range.high, or a list's first entry
        # (lists like _ARCHITECTURE_GRID are hand-ordered biggest-first).
        name: spec.high if isinstance(spec, Range) else spec[0]
        for name, spec in suggestable.items()
    }
    hp = HyperParams(**search_space.get_static_params(), **picked)
    manager = NetworkManager(hp, n_validation_size=study.n_validate)
    manager.train_epoch(0)
    jax.clear_caches()
    logger.info(f"Capacity check passed: {len(hp.hidden_dims)}x{hp.hidden_dims[0]}")


def objective(
    trial: optuna.Trial,
    search_space: SearchSpaceConfig,
    study: StudyConfig,
    display: OptunaProgressDisplay,
    hpo_benchmark_dir: Path,
) -> float:
    """Train one normally configured network and report validation loss to Optuna."""
    hp = build_hyperparams(trial, search_space)
    display_params = {
        "depth": len(hp.hidden_dims),
        "width": hp.hidden_dims[0],
        "lr_max": hp.learning_rate_max,
        "lr_min": hp.learning_rate_min,
        "wd": hp.weight_decay,
        "sig": hp.sigma_residual_adaptive_sampling,
    }
    manager = NetworkManager(
        hp,
        n_validation_size=study.n_validate,
        test_mode=study.checkpoint_policy == "none",
        output_dir=hpo_benchmark_dir,
    )
    total_epochs = hp.warmup_epochs + hp.decay_epochs
    display.start_trial(trial.number + 1, display_params, total_epochs)
    display.current_manager = manager
    manager.metrics_row_sink = display.add_metrics_row
    last_epoch = 0

    def report(epoch: int, val_loss: float | None) -> None:
        nonlocal last_epoch
        last_epoch = epoch
        display.update_epoch(epoch, val_loss)
        if val_loss is not None:
            trial.report(val_loss, epoch)
            if study.prune_trials and trial.should_prune():
                raise optuna.TrialPruned

    try:
        val_loss = manager.train(
            save_to_disk=study.checkpoint_policy == "all",
            validation_callback=report,
            show_progress=False,
        )
        if study.checkpoint_policy == "top_k":
            completed = sorted(
                t.value
                for t in trial.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
                if t.value is not None
            )
            if len(completed) < study.top_k or val_loss < completed[study.top_k - 1]:
                trial.set_user_attr("checkpoint", manager.to_disk())
        elif study.checkpoint_policy == "all":  # train() already saved; stem is the name
            trial.set_user_attr("checkpoint", manager.artifact_stem)
        display.update(trial.number + 1, display_params, val_loss, "done", total_epochs)
        return val_loss
    except optuna.TrialPruned:
        display.update(trial.number + 1, display_params, None, "pruned", last_epoch)
        raise
    except Exception:
        display.update(trial.number + 1, display_params, None, "failed", last_epoch)
        raise
    finally:
        if manager.artifact_stem is not None:
            trial.set_user_attr("run", manager.artifact_stem)
        # Pruned/failed/aborted/non-top-k trials leave nothing behind;
        # study.db, optuna.log and trials.json are the record. Only this
        # trial's own run dir is touched — sibling runs in the same study dir
        # (other trials) are never deleted, and the study dir is only removed
        # once empty (never, while study.db/optuna.log live in it).
        manager.discard_unsaved_run()
        _write_trials_json(trial.study, hpo_benchmark_dir)
        jax.clear_caches()


def _write_trials_json(study: optuna.Study, hpo_benchmark_dir: Path) -> None:
    """Ledger of which runs/checkpoints belong to this study. Rewritten after every
    trial so an aborted study still leaves an on-disk record."""
    (hpo_benchmark_dir / "trials.json").write_text(
        json.dumps(
            [
                {"trial": t.number, "state": t.state.name, "value": t.value, **t.user_attrs}
                for t in study.get_trials(deepcopy=False)
            ],
            indent=2,
        )
    )


def _save_top_configs(
    results: list[tuple[HyperParams, float]], study: optuna.Study, hpo_benchmark_dir: Path
) -> None:
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
                    {"rank": rank, "loss": loss, "config": hp.to_dict()}
                    for rank, (hp, loss) in enumerate(results, 1)
                ],
            },
            indent=2,
        )
    )
    logger.info(f"Top configurations saved to {output_file}")


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
    # Freeze the commit once for the whole run: a commit made mid-study would
    # otherwise move study.db / optuna.log / trials.json to a different path
    # and split the study across two commit dirs.
    commit = current_commit()
    hpo_benchmark_dir = study_dir(study.study_name, commit)
    file_handler = logging.FileHandler(hpo_benchmark_dir / "optuna.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    # One ledger read/write for both check_capacity and the enqueue loop below --
    # optuna_warmstart() may retrain stale configs, no reason to pay that twice.
    candidates = _get_experiment_db_configs(optuna_warmstart(search_space))
    check_capacity(search_space, study)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_path = hpo_benchmark_dir / "study.db"
    if restart and storage_path.exists():
        storage_path.unlink()

    # Concurrent create_study calls on one sqlite file race optuna's
    # check-then-insert and corrupt the db with duplicate study rows
    #
    # Crashed TUIs stay open by design, so a relaunch while one is alive is the common case.
    # Guard with an atomically-acquired lock (O_EXCL: no check-then-write gap).
    lock = hpo_benchmark_dir / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pid = lock.read_text().strip()
        if pid.isdigit() and Path(f"/proc/{pid}").exists():
            raise RuntimeError(
                f"Study '{study.study_name}' is already running (pid {pid}). "
                "Concurrent runs corrupt the study db -- quit the other process "
                "first (crashed TUIs stay open; press q there)."
            ) from None
        # Lock file names a pid that's no longer alive -- stale lock from a
        # crashed process, safe to reclaim.
        lock.unlink(missing_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))

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
                    reduction_factor=2,
                )
                if study.prune_trials
                else NopPruner()
            ),
        )
        if not optuna_study.trials:
            # Enqueue every valid, distinct candidate -- each one still costs a full
            # retrain (enqueue_trial can't inject a free historical result), but there's
            # no reason to cap how many known-good starting points TPE gets to try first.
            seen = set()
            for hp in candidates:
                if not search_space.is_valid_config(hp):
                    continue
                warmstart = {
                    name: getattr(hp, name) for name in search_space.get_suggestable_params()
                }
                key = tuple(sorted(warmstart.items()))
                if key in seen:
                    continue
                seen.add(key)
                optuna_study.enqueue_trial(warmstart)

        prior_trials = sum(t.state != optuna.trial.TrialState.WAITING for t in optuna_study.trials)
        display._prior_trials = prior_trials
        optuna_study.optimize(
            lambda trial: objective(trial, search_space, study, display, hpo_benchmark_dir),
            n_trials=max(0, study.n_trials - prior_trials),
            catch=(Exception,),
        )

        complete = sorted(
            (
                t
                for t in optuna_study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
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
        return results
    finally:
        lock.unlink(missing_ok=True)


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
        study.n_trials = 5
        study.total_epochs = 30
        study.min_epochs = 5
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

    # TUI whenever a terminal exists; piped/nohup output gets the plain Live dashboard,
    # which degrades to sequential text on non-ttys. All logging/benchmark artifacts
    # (study.db, optuna.log, trials.json, train.log, checkpoints) are identical either way.
    if sys.stdout.isatty():
        HpoApp(search_space, study, restart=reset).run()
    else:
        with OptunaProgressDisplay(study) as display:
            run_optimization(search_space, study, display=display, restart=reset)


if __name__ == "__main__":
    main()
