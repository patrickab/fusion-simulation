"""HPO refinement seeded by the prior study's four best trial configs."""

from pathlib import Path
import sys

from src.engine.optimize_network_optuna import (
    Range,
    SearchSpaceConfig,
    StudyConfig,
    run_optimization,
)
from src.lib.optuna_tui import HpoApp, OptunaProgressDisplay

STUDY_NAME = "mlp_256x5_refine"
FULL_EPOCHS = 2400
WARMUP_EPOCHS = 120
DECAY_EPOCHS = FULL_EPOCHS - WARMUP_EPOCHS
WARMSTART_DB = Path("data/hpo/2026_07_22_21_08_15_mlp_256x5_refine_a2b9b73")


def main() -> None:
    search_space = SearchSpaceConfig(
        hidden_dims=(256,) * 5,
        learning_rate_max=Range(5e-4, 5e-3, log=True),
        learning_rate_min=Range(5e-7, 5e-6, log=True),
        weight_decay=Range(1e-9, 1e-8, log=True),
        sigma_residual_adaptive_sampling=Range(0.01, 0.1),
        batch_size=32,
        soft_bc=False,
        rwf=False,
        arch="mlp",
        n_rz_inner_samples=512,
        n_rz_boundary_samples=256,
        n_train=1024,
        warmup_epochs=WARMUP_EPOCHS,
        decay_epochs=DECAY_EPOCHS,
    )

    study = StudyConfig(
        search_space=search_space,
        study_name=STUDY_NAME,
        warmstart_experiment_db=WARMSTART_DB,
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
