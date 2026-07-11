"""Optuna hyperparameter search using the standard network training path."""

import argparse
from collections import deque
import json
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import optuna
from optuna.pruners import HyperbandPruner, NopPruner
from optuna.samplers import TPESampler
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.engine.network import NetworkManager
from src.lib.logger import get_logger
from src.lib.network_config import HyperParams

logger = get_logger(name="OptunaHPO", log_dir="logs/hpo")
console = Console(width=160)


@dataclass
class SearchSpaceConfig:
    """Search bounds and fixed experiment settings."""

    depth_range: tuple[int, int] = (4, 5)
    width_choices: tuple[int, ...] = (128, 160, 200)
    lr_max_range: tuple[float, float] = (1e-3, 3e-3)
    lr_min_ratio_range: tuple[float, float] = (0.001, 0.08)
    weight_decay_range: tuple[float, float] = (1e-8, 1e-4)
    sigma_residual_range: tuple[float, float] = (0.01, 0.05)

    weight_boundary_condition: float = 10.0
    n_rz_inner: int = 512
    n_rz_boundary: int = 128
    batch_size: int = 64
    n_train: int = 1024
    total_epochs: int = 600
    warmup_ratio: float = 0.167
    n_validate: int = 128

    min_epochs: int = 50
    n_trials: int = 50
    top_k: int = 3
    n_startup_trials: int = 10

    huber_delta: float = 1.0
    n_fourier_features: int = 64
    lbfgs_steps: int = 0
    study_name: str = "pinn_hpo"
    prune_trials: bool = True
    save_all: bool = False


class OptunaProgressDisplay:
    """Rich live dashboard showing optimization progress and top configs."""

    def __init__(self, config: SearchSpaceConfig, prior_trials: int = 0) -> None:
        self.config = config
        # Keep a rolling window: an ever-growing table eventually exceeds the terminal height
        # and desynchronizes Rich's cursor-based redraw when the terminal scrolls.
        self._trials_data: deque[dict[str, Any]] = deque(maxlen=15)
        self._best_configs: list[tuple[dict[str, Any], float]] = []
        self._best_loss, self._start_time = float("inf"), datetime.now()
        self._counts = {"pruned": 0, "failed": 0, "done": 0}
        self._trials_processed, self._prior_trials = 0, prior_trials
        self._current_trial_info: dict[str, Any] = {}

        self._progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("({task.percentage:.0f}%)"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task = self._progress.add_task("Progress:", total=self.config.n_trials)
        self._epoch_task = self._progress.add_task(
            "[magenta]Epochs:  ", total=self.config.total_epochs, visible=False
        )
        self._live = Live(
            self._build_layout(),
            refresh_per_second=4,
            console=console,
            vertical_overflow="visible",
        )

    def _get_trials_table(self) -> Table:
        table = Table(title="Previous Trials", show_header=True, header_style="bold cyan")
        for column in [
            "Trial",
            "Depth",
            "Width",
            "Max LR",
            "Min LR",
            "Weight Decay",
            "Sig Adapt Sampling",
            "Val Loss",
            "Status",
        ]:
            table.add_column(column, justify="left" if column == "Status" else "right")
        for data in self._trials_data:
            table.add_row(
                *[
                    str(data.get(key, "?"))
                    for key in (
                        "trial",
                        "depth",
                        "width",
                        "lr_max",
                        "lr_min",
                        "wd",
                        "sig",
                        "loss",
                        "status",
                    )
                ]
            )
        return table

    def _get_best_table(self) -> Table:
        table = Table(
            title=f"Top {self.config.top_k} Configs",
            show_header=True,
            header_style="bold green",
        )
        for column in [
            "Rank",
            "Depth",
            "Width",
            "Max LR",
            "Min LR",
            "Weight Decay",
            "Sig Adapt Sampling",
            "Val Loss",
        ]:
            table.add_column(column, justify="right")
        for rank, (params, loss) in enumerate(self._best_configs, 1):
            table.add_row(
                str(rank),
                str(params.get("depth", "?")),
                str(params.get("width", "?")),
                f"{params.get('lr_max', 0):.2e}",
                f"{params.get('lr_min', 0):.2e}",
                f"{params.get('wd', 0):.2e}",
                f"{params.get('sig', 0):.3f}",
                f"{loss:.4f}",
            )
        return table

    def _build_layout(self) -> Panel:
        elapsed_str = str(datetime.now() - self._start_time).split(".")[0]
        summary = (
            f"[bold]Best Val Loss:[/] {self._best_loss:.4f}  |  "
            f"[bold]Elapsed:[/] {elapsed_str}\n"
            f"Session: {self._counts['done']} done  |  {self._counts['pruned']} pruned  |  "
            f"{self._counts['failed']} failed"
            + (f"  |  Prior: {self._prior_trials}" if self._prior_trials else "")
        )

        current_table = Table(show_header=False, box=box.SIMPLE)
        if self._current_trial_info:
            current = self._current_trial_info
            params = current["params"]
            val_loss = current.get("val_loss")
            title = f"Current Trial: {current['trial']}"
            rows = [
                ("Architecture:", f"{params.get('depth')}x{params.get('width')}"),
                ("Max LR:", f"{params.get('lr_max', 0):.2e}"),
                ("Min LR:", f"{params.get('lr_min', 0):.2e}"),
                ("Weight Decay:", f"{params.get('wd', 0):.2e}"),
                ("Sigma Res:", f"{params.get('sig', 0):.3f}"),
                (
                    "Recent Val Loss:",
                    f"[bold cyan]{val_loss:.4f}[/bold cyan]"
                    if val_loss is not None
                    else "[bold cyan]--[/bold cyan]",
                ),
            ]
        else:
            title = "Current Trial: ---"
            rows = [
                (key, "---" if "Loss" not in key else "[bold cyan]---[/bold cyan]")
                for key in (
                    "Architecture:",
                    "Max LR:",
                    "Min LR:",
                    "Weight Decay:",
                    "Sigma Res:",
                    "Recent Val Loss:",
                )
            ]
        current_table.title = title
        for key, value in rows:
            current_table.add_row(key, value)

        return Panel(
            Group(
                self._progress,
                summary,
                self._get_best_table(),
                Panel(current_table, border_style="magenta"),
                self._get_trials_table(),
            ),
            title="[bold cyan]PINN HPO Optimization[/bold cyan]",
            border_style="cyan",
        )

    def start_trial(self, trial_num: int, params: dict[str, Any], total_epochs: int) -> None:
        self._current_trial_info = {
            "trial": trial_num,
            "params": params,
            "epoch": 0,
            "val_loss": None,
        }
        self._progress.remove_task(self._epoch_task)
        self._epoch_task = self._progress.add_task(
            "[magenta]Epochs:  ", total=total_epochs, visible=True
        )
        self._live.update(self._build_layout())

    def update_epoch(self, epoch: int, val_loss: float | None = None) -> None:
        if not self._current_trial_info:
            return
        self._current_trial_info["epoch"] = epoch
        if val_loss is not None:
            self._current_trial_info["val_loss"] = val_loss
        self._progress.update(self._epoch_task, completed=epoch)
        self._live.update(self._build_layout())

    def would_qualify_for_top_k(self, loss: float) -> bool:
        return len(self._best_configs) < self.config.top_k or loss < self._best_configs[-1][1]

    def update(
        self,
        trial_num: int,
        params: dict[str, Any],
        loss: float | None,
        status: str,
        epoch: int | None = None,
    ) -> None:
        self._trials_processed += 1
        self._progress.update(self._task, completed=self._trials_processed)
        self._progress.update(self._epoch_task, visible=False)
        self._current_trial_info = {}
        if status in self._counts:
            self._counts[status] += 1

        if loss is not None:
            self._best_configs.append((params.copy(), loss))
            self._best_configs.sort(key=lambda item: item[1])
            self._best_configs = self._best_configs[: self.config.top_k]
            self._best_loss = self._best_configs[0][1]

        color = {"pruned": "yellow", "failed": "red", "done": "green"}.get(status, "white")
        status_text = f"[{color}]{status}" + (f" @ {epoch}" if epoch is not None else "") + "[/]"
        self._trials_data.append(
            {
                "trial": trial_num,
                "depth": params.get("depth", "?"),
                "width": params.get("width", "?"),
                "lr_max": f"{params.get('lr_max', 0):.2e}",
                "lr_min": f"{params.get('lr_min', 0):.2e}",
                "wd": f"{params.get('wd', 0):.2e}",
                "sig": f"{params.get('sig', 0):.3f}",
                "loss": f"{loss:.4f}" if loss is not None else "--",
                "status": status_text,
            }
        )
        self._live.update(self._build_layout())

    def __enter__(self) -> "OptunaProgressDisplay":
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: ANN401
        self._live.__exit__(*args)


def build_hyperparams(trial: optuna.Trial, config: SearchSpaceConfig) -> HyperParams:
    """Sample trainable hyperparameters and apply the study's fixed settings."""
    depth = trial.suggest_int("depth", *config.depth_range)
    width = trial.suggest_categorical("width", config.width_choices)
    lr_max = trial.suggest_float("lr_max", *config.lr_max_range, log=True)
    lr_min_ratio = trial.suggest_float("lr_min_ratio", *config.lr_min_ratio_range, log=True)

    warmup_epochs = int(config.total_epochs * config.warmup_ratio)
    return HyperParams(
        hidden_dims=(width,) * depth,
        learning_rate_max=lr_max,
        learning_rate_min=lr_max * lr_min_ratio,
        weight_decay=trial.suggest_float(
            "weight_decay", *config.weight_decay_range, log=True
        ),
        weight_boundary_condition=config.weight_boundary_condition,
        sigma_residual_adaptive_sampling=trial.suggest_float(
            "sigma_residual", *config.sigma_residual_range
        ),
        n_rz_inner_samples=config.n_rz_inner,
        n_rz_boundary_samples=config.n_rz_boundary,
        warmup_epochs=warmup_epochs,
        decay_epochs=config.total_epochs - warmup_epochs,
        batch_size=config.batch_size,
        n_train=config.n_train,
        huber_delta=config.huber_delta,
        n_fourier_features=config.n_fourier_features,
        lbfgs_steps=config.lbfgs_steps,
    )


def check_capacity(config: SearchSpaceConfig) -> None:
    """Verify that the largest architecture fits before starting the study."""
    hp = HyperParams(
        hidden_dims=(max(config.width_choices),) * config.depth_range[1],
        learning_rate_max=1e-3,
        n_rz_inner_samples=config.n_rz_inner,
        n_rz_boundary_samples=config.n_rz_boundary,
        batch_size=config.batch_size,
        warmup_epochs=10,
        decay_epochs=10,
        huber_delta=config.huber_delta,
        n_fourier_features=config.n_fourier_features,
    )
    manager = NetworkManager(hp, n_validation_size=config.n_validate)
    manager.train_epoch(0)
    jax.clear_caches()
    logger.info(f"Capacity check passed: {config.depth_range[1]}x{max(config.width_choices)}")


def objective(
    trial: optuna.Trial,
    config: SearchSpaceConfig,
    display: OptunaProgressDisplay,
) -> float:
    """Train one normally configured network and report validation loss to Optuna."""
    hp = build_hyperparams(trial, config)
    display_params = {
        "depth": len(hp.hidden_dims),
        "width": hp.hidden_dims[0],
        "lr_max": hp.learning_rate_max,
        "lr_min": hp.learning_rate_min,
        "wd": hp.weight_decay,
        "sig": hp.sigma_residual_adaptive_sampling,
    }
    manager = NetworkManager(hp, n_validation_size=config.n_validate)
    total_epochs = hp.warmup_epochs + hp.decay_epochs
    display.start_trial(trial.number + 1, display_params, total_epochs)
    last_epoch = 0

    def report(epoch: int, val_loss: float | None) -> None:
        nonlocal last_epoch
        last_epoch = epoch
        display.update_epoch(epoch, val_loss)
        if val_loss is not None:
            trial.report(val_loss, epoch)
            if config.prune_trials and trial.should_prune():
                raise optuna.TrialPruned

    try:
        val_loss = manager.train(
            save_to_disk=config.save_all,
            validation_callback=report,
            show_progress=False,
        )
        if not config.save_all:
            completed = sorted(
                t.value
                for t in trial.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
                if t.value is not None
            )
            if len(completed) < config.top_k or val_loss < completed[config.top_k - 1]:
                manager.to_disk()
        display.update(trial.number + 1, display_params, val_loss, "done", total_epochs)
        return val_loss
    except optuna.TrialPruned:
        display.update(trial.number + 1, display_params, None, "pruned", last_epoch)
        raise
    except Exception:
        display.update(trial.number + 1, display_params, None, "failed", last_epoch)
        raise
    finally:
        jax.clear_caches()


def get_warmstart_config(config: SearchSpaceConfig) -> dict[str, Any] | None:
    """Return the best known configuration when it belongs to the search space."""
    warmstart = {
        "depth": 4,
        "width": 128,
        "lr_max": 0.002,
        "lr_min_ratio": 0.025,
        "weight_decay": 1e-7,
        "sigma_residual": 0.05,
    }
    if config.depth_range[0] <= warmstart["depth"] <= config.depth_range[1] and warmstart[
        "width"
    ] in config.width_choices:
        return warmstart
    return None


def _save_top_configs(results: list[tuple[HyperParams, float]], study: optuna.Study) -> None:
    output_dir = Path("logs/hpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"top_{study.study_name}_{datetime.now():%Y%m%d_%H%M}.json"
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
    config: SearchSpaceConfig | None = None,
    restart: bool = False,
) -> list[tuple[HyperParams, float]]:
    """Run or resume a study and return its best configurations."""
    config = config or SearchSpaceConfig()
    check_capacity(config)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_path = Path("logs/hpo") / f"{config.study_name}.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    if restart and storage_path.exists():
        storage_path.unlink()

    study = optuna.create_study(
        study_name=config.study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=not restart,
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=config.n_startup_trials),
        pruner=(
            HyperbandPruner(
                min_resource=config.min_epochs,
                max_resource=config.total_epochs,
                reduction_factor=2,
            )
            if config.prune_trials
            else NopPruner()
        ),
    )
    if not study.trials:
        warmstart = get_warmstart_config(config)
        if warmstart is not None:
            study.enqueue_trial(warmstart)

    prior_trials = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
    with OptunaProgressDisplay(config, prior_trials=prior_trials) as display:
        study.optimize(
            lambda trial: objective(trial, config, display),
            n_trials=max(0, config.n_trials - prior_trials),
            catch=(Exception,),
        )

    complete = sorted(
        (
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ),
        key=lambda trial: trial.value,
    )[: config.top_k]
    results = [(build_hyperparams(trial, config), trial.value) for trial in complete]
    for rank, (hp, loss) in enumerate(results, 1):
        architecture = f"{len(hp.hidden_dims)}x{hp.hidden_dims[0]}"
        logger.info(
            f"Rank {rank}: loss={loss:.6f}, architecture={architecture}, "
            f"lr={hp.learning_rate_max:.2e}"
        )
    _save_top_configs(results, study)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for PINN")
    parser.add_argument("--restart-experiment", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--variant", choices=("mse", "huber_ff64_lbfgs"))
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--full-trials", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    args = parser.parse_args()

    config = SearchSpaceConfig()
    if args.variant == "mse":
        config.huber_delta = 0.0
        config.n_fourier_features = 0
        config.study_name = "pinn_hpo_mse"
    elif args.variant == "huber_ff64_lbfgs":
        config.lbfgs_steps = 300
        config.study_name = "pinn_hpo_huber_ff64_lbfgs"
    if args.n_trials is not None:
        config.n_trials = args.n_trials
    config.prune_trials = not args.full_trials
    config.save_all = args.save_all

    if args.test:
        config.batch_size = 8
        config.n_train = 64
        config.n_rz_boundary = 16
        config.n_rz_inner = 64
        config.depth_range = (1, 2)
        config.width_choices = (32, 128)
        config.n_validate = 16
        config.n_trials = 5
        config.total_epochs = 30
        config.min_epochs = 5
        config.n_startup_trials = 2
        args.restart_experiment = True

    run_optimization(config, restart=args.restart_experiment)


if __name__ == "__main__":
    main()
