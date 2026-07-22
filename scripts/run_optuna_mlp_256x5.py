"""HPO refinement for the 5x256 MLP, warm-started from its capacity probe."""

from pathlib import Path
import sys

from src.engine.optimize_network_optuna import (
    Range,
    SearchSpaceConfig,
    StudyConfig,
    run_optimization,
    study_dir,
)
from src.lib.config import current_commit
from src.lib.optuna_tui import HpoApp, OptunaProgressDisplay

STUDY_NAME = "mlp_256x5"
WARMSTART_RUN = Path("data/benchmarks/2026_07_20_10_49_18_mlp_256x5_probe_6d2fd3b")


def main() -> None:
    search_space = SearchSpaceConfig(
        hidden_dims=(256,) * 5,
        learning_rate_max=Range(1e-3, 6e-3, log=True),
        learning_rate_min=Range(2e-6, 8e-6, log=True),
        weight_decay=Range(1e-9, 1e-8, log=True),
        sigma_residual_adaptive_sampling=Range(0.01, 0.1),
        batch_size=32,
        soft_bc=False,
        rwf=False,
        arch="mlp",
        n_rz_inner_samples=512,
        n_rz_boundary_samples=256,
        n_train=1024,
        warmup_epochs=400,
        decay_epochs=2000,
    )
    path = study_dir(STUDY_NAME, current_commit())
    study = StudyConfig(
        search_space=search_space,
        study_name=STUDY_NAME,
        warmstart_experiment_db=path,
        warmstart_config_paths=[WARMSTART_RUN / "run.json"],
        min_epochs=800,
    )
    if sys.stdout.isatty():
        app = HpoApp(study, restart=False)
        app.run()
        if app.crash_traceback is not None:
            print(app.crash_traceback, file=sys.stderr)
            sys.exit(1)
    else:
        with OptunaProgressDisplay(study) as display:
            run_optimization(study, display=display)


if __name__ == "__main__":
    main()
