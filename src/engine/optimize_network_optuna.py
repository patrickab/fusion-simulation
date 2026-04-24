"""Optuna-based hyperparameter optimization with physics-based validation.

Uses a fixed validation set of Sobol-sampled plasma configs to evaluate generalization.
Validation loss uses forward pass only (no gradient computation) via make_psi_fn,
computed every val_eval_frequency epochs for pruning decisions.
"""

import argparse
import json
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from scipy.stats import qmc

from src.engine.network import NetworkManager
from src.lib.logger import get_logger
from src.lib.network_config import DomainBounds, HyperParams

logger = get_logger(name="OptunaHPO", log_dir="logs/hpo")
console = Console(width=160)


@dataclass
class SearchSpaceConfig:
    """Hyperparameter search bounds for Optuna HPO."""

    depth_range: tuple[int, int] = (4, 5)
    width_choices: tuple[int, ...] = (128, 160, 200)
    lr_max_range: tuple[float, float] = (1e-3, 3e-3)
    lr_min_ratio_range: tuple[float, float] = (0.001, 0.08)
    weight_decay_range: tuple[float, float] = (1e-8, 1e-4)
    sigma_residual_range: tuple[float, float] = (0.01, 0.05)

    weight_boundary_condition: float = 10.0
    n_rz_inner: int = 512
    total_epochs: int = 600
    warmup_ratio: float = 0.167

    batch_size: int = 32
    n_train: int = 1024
    n_rz_boundary: int = 128

    n_validate: int = 128
    val_eval_frequency: int = 20

    min_epochs: int = 50
    n_trials: int = 50
    top_k: int = 3
    n_startup_trials: int = 10


def create_validation_configs(
    config: SearchSpaceConfig,
) -> dict[str, Any]:
    """Create fixed validation plasma configs on CPU (numpy)."""
    lower, upper = DomainBounds.get_bounds()

    sobol = qmc.Sobol(d=len(lower), scramble=True, seed=123)
    samples = sobol.random(config.n_validate)
    plasma_configs = np.array(qmc.scale(samples, lower, upper), dtype=np.float32)

    return {"plasma_configs": plasma_configs}


def _calculate_validation_loss(
    manager: NetworkManager,
    val_data: dict[str, np.ndarray],
    n_rz_inner: int,
    n_rz_boundary: int,
) -> float:
    """Calculate validation loss in chunks using fast, gradient-free eval_step."""
    plasma_configs = val_data["plasma_configs"]
    n_configs = len(plasma_configs)

    # Large chunk size to maximize GPU throughput during validation.
    # We use a fixed large chunk as validation doesn't track gradients, saving VRAM.
    chunk_size = 128
    total_loss = 0.0
    n_chunks = 0

    weight_bc = manager.config.weight_boundary_condition
    sampler = manager.sampler
    sampler.precompute_coordinate_samples(n_rz_inner, n_rz_boundary)

    for i in range(0, n_configs, chunk_size):
        end = min(i + chunk_size, n_configs)
        batch = jnp.array(plasma_configs[i:end], dtype=jnp.float32)
        inputs = sampler.sample_flux_input(plasma_configs=batch)

        loss, _, _, _ = manager.eval_step(manager.state, inputs, weight_bc)
        total_loss += float(loss)
        n_chunks += 1

    return total_loss / n_chunks


class OptunaProgressDisplay:
    """Rich live dashboard showing optimization progress and top configs."""

    def __init__(self, config: SearchSpaceConfig, prior_trials: int = 0) -> None:
        self.config = config
        self._trials_data, self._best_configs = [], []
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
            self._build_layout(), refresh_per_second=4, console=console, vertical_overflow="visible"
        )

    def _get_trials_table(self) -> Table:
        t = Table(title="Previous Trials", show_header=True, header_style="bold cyan")
        for c in [
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
            t.add_column(c, justify="left" if c == "Status" else "right")
        for d in self._trials_data:
            t.add_row(
                *[
                    str(d.get(k, "?"))
                    for k in [
                        "trial",
                        "depth",
                        "width",
                        "lr_max",
                        "lr_min",
                        "wd",
                        "sig",
                        "loss",
                        "status",
                    ]
                ]
            )
        return t

    def _get_best_table(self) -> Table:
        t = Table(
            title=f"Top {self.config.top_k} Configs", show_header=True, header_style="bold green"
        )
        for c in [
            "Rank",
            "Depth",
            "Width",
            "Max LR",
            "Min LR",
            "Weight Decay",
            "Sig Adapt Sampling",
            "Val Loss",
        ]:
            t.add_column(c, justify="right")
        for rank, (p, loss) in enumerate(self._best_configs, 1):
            t.add_row(
                str(rank),
                str(p.get("depth", "?")),
                str(p.get("width", "?")),
                f"{p.get('lr_max', 0):.2e}",
                f"{p.get('lr_min', 0):.2e}",
                f"{p.get('wd', 0):.2e}",
                f"{p.get('sig', 0):.3f}",
                f"{loss:.4f}",
            )
        return t

    def _build_layout(self) -> Panel:
        elapsed_str = str(datetime.now() - self._start_time).split(".")[0]
        summary = (
            f"[bold]Best Val Loss:[/] {self._best_loss:.4f}  |  [bold]Elapsed:[/] {elapsed_str}\n"
            f"Session: {self._counts['done']} done  |  {self._counts['pruned']} pruned  |  {self._counts['failed']} failed"
            + (f"  |  Prior: {self._prior_trials}" if self._prior_trials else "")
        )

        curr_table = Table(show_header=False, box=box.SIMPLE)
        if self._current_trial_info:
            ct, p = self._current_trial_info, self._current_trial_info["params"]
            val = ct.get("val_loss")
            title, rows = (
                f"Current Trial: {ct['trial']}",
                [
                    ("Architecture:", f"{p.get('depth')}x{p.get('width')}"),
                    ("Max LR:", f"{p.get('lr_max', 0):.2e}"),
                    ("Min LR:", f"{p.get('lr_min', 0):.2e}"),
                    ("Weight Decay:", f"{p.get('wd', 0):.2e}"),
                    ("Sigma Res:", f"{p.get('sig', 0):.3f}"),
                    (
                        "Recent Val Loss:",
                        f"[bold cyan]{val:.4f}[/bold cyan]"
                        if val is not None
                        else "[bold cyan]--[/bold cyan]",
                    ),
                ],
            )
        else:
            title, rows = (
                "Current Trial: ---",
                [
                    (k, "---" if "Loss" not in k else "[bold cyan]---[/bold cyan]")
                    for k in [
                        "Architecture:",
                        "Max LR:",
                        "Min LR:",
                        "Weight Decay:",
                        "Sigma Res:",
                        "Recent Val Loss:",
                    ]
                ],
            )

        curr_table.title = title
        for k, v in rows:
            curr_table.add_row(k, v)

        elements = [
            self._progress,
            summary,
            self._get_best_table(),
            Panel(curr_table, border_style="magenta"),
            self._get_trials_table(),
        ]
        return Panel(
            Group(*elements),
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
        if len(self._best_configs) < self.config.top_k:
            return True
        return loss < self._best_configs[-1][1]

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
            self._best_configs.sort(key=lambda x: x[1])
            self._best_configs = self._best_configs[: self.config.top_k]
            self._best_loss = self._best_configs[0][1]

        c = {"pruned": "yellow", "failed": "red", "done": "green"}.get(status, "white")
        status_str = f"[{c}]{status}" + (f" @ {epoch}" if epoch is not None else "") + "[/]"

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
                "status": status_str,
            }
        )
        self._live.update(self._build_layout())

    def __enter__(self) -> "OptunaProgressDisplay":
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: ANN401
        self._live.__exit__(*args)


def check_capacity(config: SearchSpaceConfig) -> None:
    """Verify largest architecture fits in GPU memory before starting HPO."""
    max_depth = config.depth_range[1]
    max_width = max(config.width_choices)

    test_config = HyperParams(
        hidden_dims=tuple([max_width] * max_depth),
        learning_rate_max=1e-3,
        n_rz_inner_samples=config.n_rz_inner,
        n_rz_boundary_samples=config.n_rz_boundary,
        batch_size=config.batch_size,
        warmup_epochs=10,
        decay_epochs=10,
    )

    try:
        manager = NetworkManager(test_config)
        manager.train_epoch(0)
        jax.clear_caches()
        logger.info(f"Capacity check passed: {max_depth}x{max_width}")
    except Exception as e:
        logger.error(f"Capacity check failed: {e}")
        raise RuntimeError("Reduce search space - OOM on max architecture") from e


def build_hyperparams(trial: optuna.Trial, config: SearchSpaceConfig) -> HyperParams:
    """Sample hyperparameters from search space using define-by-run."""
    depth = trial.suggest_int("depth", *config.depth_range)
    width = trial.suggest_categorical("width", config.width_choices)

    lr_max = trial.suggest_float("lr_max", *config.lr_max_range, log=True)
    lr_min_ratio = trial.suggest_float("lr_min_ratio", *config.lr_min_ratio_range, log=True)
    weight_decay = trial.suggest_float("weight_decay", *config.weight_decay_range, log=True)
    sigma_residual = trial.suggest_float("sigma_residual", *config.sigma_residual_range)
    return HyperParams(
        hidden_dims=tuple([width] * depth),
        learning_rate_max=lr_max,
        learning_rate_min=lr_max * lr_min_ratio,
        weight_decay=weight_decay,
        weight_boundary_condition=config.weight_boundary_condition,
        sigma_residual_adaptive_sampling=sigma_residual,
        n_rz_inner_samples=config.n_rz_inner,
        n_rz_boundary_samples=config.n_rz_boundary,
        warmup_epochs=int(config.total_epochs * config.warmup_ratio),
        decay_epochs=config.total_epochs - int(config.total_epochs * config.warmup_ratio),
        batch_size=config.batch_size,
        n_train=config.n_train,
    )


def objective(
    trial: optuna.Trial,
    config: SearchSpaceConfig,
    display: OptunaProgressDisplay,
    val_data: dict[str, np.ndarray],
) -> float:
    """Train trial and return validation loss for unbiased HPO.

    Validation loss evaluated every val_eval_frequency epochs to balance
    generalization signal with computational overhead.
    """
    hp = build_hyperparams(trial, config)
    params_for_display = {
        "depth": len(hp.hidden_dims),
        "width": hp.hidden_dims[0],
        "lr_max": hp.learning_rate_max,
        "lr_min": hp.learning_rate_min,
        "wd": hp.weight_decay,
        "sig": hp.sigma_residual_adaptive_sampling,
    }

    try:
        manager = NetworkManager(hp)
        total_epochs = hp.warmup_epochs + hp.decay_epochs
        val_loss = float("inf")
        val_loss_history: list[float] = []
        display.start_trial(trial.number + 1, params_for_display, total_epochs)

        for epoch in range(total_epochs):
            manager.train_epoch(epoch)

            if epoch >= config.min_epochs and epoch % config.val_eval_frequency == 0:
                val_loss = _calculate_validation_loss(
                    manager, val_data, config.n_rz_inner, config.n_rz_boundary
                )
                val_loss_history.append(val_loss)
                if len(val_loss_history) > 3:
                    val_loss_history.pop(0)
                avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                trial.report(avg_val_loss, epoch)

                if trial.should_prune():
                    display.update(trial.number + 1, params_for_display, None, "pruned", epoch)
                    raise optuna.TrialPruned

            # Update live epoch progress display
            display.update_epoch(epoch + 1, val_loss if val_loss != float("inf") else None)

        val_loss = _calculate_validation_loss(
            manager, val_data, config.n_rz_inner, config.n_rz_boundary
        )
        if display.would_qualify_for_top_k(val_loss):
            manager.to_disk()
        jax.clear_caches()
        display.update(trial.number + 1, params_for_display, val_loss, "done", total_epochs)
        return val_loss

    except optuna.TrialPruned:
        raise
    except Exception:
        logger.exception(f"Trial {trial.number} failed")
        display.update(
            trial.number + 1,
            params_for_display,
            None,
            "failed",
            epoch if "epoch" in locals() else None,
        )
        return float("inf")


def get_warmstart_config() -> dict[str, Any]:
    """Return best known configuration for warm-starting optimization."""
    return {
        "depth": 4,
        "width": 128,
        "lr_max": 0.002,
        "lr_min_ratio": 0.025,
        "weight_decay": 1e-7,
        "sigma_residual": 0.05,

    }


def _save_top_configs(results: list[tuple[HyperParams, float]], study: optuna.Study) -> None:
    """Persist top-k configs with full hyperparameter details to JSON."""
    output_dir = Path("logs/hpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = output_dir / f"top_k_configs_{timestamp}.json"

    data = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "n_completed": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        "n_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        "best_loss": results[0][1] if results else None,
        "top_k": [
            {"rank": i + 1, "loss": loss, "config": hp.to_dict()}
            for i, (hp, loss) in enumerate(results)
        ],
    }

    output_file.write_text(json.dumps(data, indent=2))
    logger.info(f"Top-k configs saved to: {output_file}")


def run_optimization(
    config: SearchSpaceConfig | None = None, restart: bool = False
) -> list[tuple[HyperParams, float]]:
    """Execute HPO study with validation-based pruning and return top-k configs."""
    config = config or SearchSpaceConfig()
    check_capacity(config)

    # Disable Optuna's default stdout logging to prevent rich Live display corruption
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    val_data = create_validation_configs(config)

    storage_path = Path("logs/hpo/optuna_study.db")
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    if restart and storage_path.exists():
        storage_path.unlink()
        logger.info("Deleted existing study database")

    study = optuna.create_study(
        study_name="pinn_hpo",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=not restart,
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=config.n_startup_trials),
        pruner=HyperbandPruner(
            min_resource=config.min_epochs,
            max_resource=config.total_epochs,
            reduction_factor=2,
        ),
    )

    if len(study.trials) == 0:
        study.enqueue_trial(get_warmstart_config())

    prior_trials = sum(1 for t in study.trials if t.state != optuna.trial.TrialState.WAITING)
    remaining_trials = max(0, config.n_trials - prior_trials)
    if remaining_trials == 0:
        logger.info(
            f"Study already has {prior_trials} trials (target: {config.n_trials}). Nothing to run."
        )
        display = OptunaProgressDisplay(config, prior_trials=prior_trials)
    else:
        display = OptunaProgressDisplay(config, prior_trials=prior_trials)
        with display:
            study.optimize(
                lambda trial: objective(trial, config, display, val_data),
                n_trials=remaining_trials,
                show_progress_bar=False,
            )

    complete_trials = sorted(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value < float("inf")
        ],
        key=lambda t: t.value,
    )[: config.top_k]

    results = [(build_hyperparams(t, config), t.value) for t in complete_trials]

    console.print("\n" + "=" * 80)
    console.print("[bold green]HPO COMPLETE - TOP CONFIGURATIONS[/bold green]")
    console.print("=" * 80)
    for i, (hp, loss) in enumerate(results, 1):
        console.print(f"\n[bold]Rank {i}:[/] Val Loss: {loss:.4f}")
        console.print(f"  Architecture:  {len(hp.hidden_dims)}x{hp.hidden_dims[0]}")
        console.print(f"  LR max:        {hp.learning_rate_max:.2e}")
        console.print(f"  LR min:        {hp.learning_rate_min:.2e}")
        console.print(f"  Weight decay:  {hp.weight_decay:.2e}")
        console.print(f"  Sigma:         {hp.sigma_residual_adaptive_sampling:.3f}")
        console.print(f"  Warmup epochs: {hp.warmup_epochs}")
    console.print("\n" + "=" * 80 + "\n")

    _save_top_configs(results, study)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for PINN")
    parser.add_argument(
        "--restart-experiment", action="store_true", help="Delete existing study and start fresh"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run with minimal parameters for rapid iteration"
    )
    args = parser.parse_args()

    config = SearchSpaceConfig()
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

    try:
        run_optimization(config, restart=args.restart_experiment)
    except KeyboardInterrupt:
        logger.info("\nOptimization interrupted. Study saved to SQLite.")
        jax.clear_caches()
