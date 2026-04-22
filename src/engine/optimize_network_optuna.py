import json
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.engine.network import NetworkManager
from src.lib.logger import get_logger
from src.lib.network_config import HyperParams

logger = get_logger(name="OptunaHPO", log_dir="logs/hpo")
console = Console()


@dataclass
class SearchSpaceConfig:
    depth_range: tuple[int, int] = (3, 6)
    width_choices: tuple[int, ...] = (128, 160, 192, 224, 256)
    lr_max_range: tuple[float, float] = (1e-4, 5e-3)
    lr_min_ratio_range: tuple[float, float] = (0.01, 0.1)
    weight_decay_range: tuple[float, float] = (1e-9, 1e-5)
    weight_boundary_range: tuple[float, float] = (1.0, 50.0)
    sigma_residual_range: tuple[float, float] = (0.01, 0.2)
    n_rz_inner_choices: tuple[int, ...] = (1024, 1536, 2048, 3072, 4096)
    warmup_range: tuple[int, int] = (50, 150)
    decay_range: tuple[int, int] = (300, 800)

    batch_size: int = 64
    n_train: int = 1024
    n_rz_boundary: int = 256

    min_epochs: int = 50
    max_epochs: int = 600

    n_trials: int = 50
    top_k: int = 3
    n_startup_trials: int = 10


class OptunaProgressDisplay:
    """Rich live dashboard for HPO optimization progress."""

    def __init__(self, config: SearchSpaceConfig) -> None:
        self.config = config
        self._trials_data: list[dict[str, Any]] = []
        self._best_configs: list[tuple[dict[str, Any], float]] = []
        self._best_loss = float("inf")
        self._n_pruned = 0
        self._n_failed = 0
        self._n_completed = 0
        self._start_time = datetime.now()
        self._setup_widgets()

    def _setup_widgets(self) -> None:
        self._progress = Progress(
            TextColumn("[bold cyan]Progress:"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TextColumn("{task.completed}/{task.total} trials"),
            TextColumn("({task.percentage:.0f}%)"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task = self._progress.add_task("", total=self.config.n_trials)

        trials_title = "Recent Trials"
        self._trials_table = Table(
            title=trials_title, show_header=True, header_style="bold cyan"
        )
        self._trials_table.add_column("Trial", width=5, justify="right")
        self._trials_table.add_column("Depth", width=5, justify="right")
        self._trials_table.add_column("Width", width=5, justify="right")
        self._trials_table.add_column("LR", width=8, justify="right")
        self._trials_table.add_column("Loss", width=10, justify="right")
        self._trials_table.add_column("Status", width=8)

        best_title = f"Top {self.config.top_k} Configs"
        self._best_table = Table(
            title=best_title, show_header=True, header_style="bold green"
        )
        self._best_table.add_column("Rank", width=4, justify="right")
        self._best_table.add_column("Depth", width=5, justify="right")
        self._best_table.add_column("Width", width=5, justify="right")
        self._best_table.add_column("Loss", width=10, justify="right")

        self._live = Live(
            self._build_layout(), refresh_per_second=4, console=console
        )

    def _build_layout(self) -> Panel:
        elapsed = datetime.now() - self._start_time
        secs = elapsed.seconds
        elapsed_str = f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"

        summary_lines = [
            f"[bold]Best Loss:[/] {self._best_loss:.4f}  |  [bold]Elapsed:[/] {elapsed_str}",
            f"Completed: {self._n_completed}  |  Pruned: {self._n_pruned}  "
            f"| Failed: {self._n_failed}",
        ]
        summary = "\n".join(summary_lines)

        return Panel(
            Group(self._progress, summary, self._trials_table, self._best_table),
            title="[bold cyan]PINN HPO Optimization[/bold cyan]",
            border_style="cyan",
        )

    def update(
        self, trial_num: int, params: dict[str, Any], loss: float | None, status: str
    ) -> None:
        """Update display with new trial result."""
        self._progress.update(self._task, completed=trial_num)

        if status == "pruned":
            self._n_pruned += 1
        elif status == "failed":
            self._n_failed += 1
        elif status == "done":
            self._n_completed += 1

        if loss is not None:
            loss_str = f"{loss:.4f}"
            self._update_best_configs(params, loss)
        else:
            loss_str = "--"

        status_display = {
            "done": "[green]done[/green]",
            "pruned": "[yellow]pruned[/yellow]",
            "failed": "[red]failed[/red]",
        }.get(status, status)

        self._trials_data.append(
            {
                "trial": trial_num,
                "depth": params.get("depth", "?"),
                "width": params.get("width", "?"),
                "lr": f"{params.get('lr', 0):.1e}",
                "loss": loss_str,
                "status": status_display,
            }
        )

        if len(self._trials_data) > 8:
            self._trials_data.pop(0)

        self._rebuild_trials_table()
        self._rebuild_best_table()
        self._live.update(self._build_layout())

    def _update_best_configs(self, params: dict[str, Any], loss: float) -> None:
        self._best_configs.append((params.copy(), loss))
        self._best_configs.sort(key=lambda x: x[1])
        self._best_configs = self._best_configs[: self.config.top_k]
        if self._best_configs:
            self._best_loss = self._best_configs[0][1]

    def _rebuild_trials_table(self) -> None:
        trials_title = "Recent Trials"
        self._trials_table = Table(
            title=trials_title, show_header=True, header_style="bold cyan"
        )
        self._trials_table.add_column("Trial", width=5, justify="right")
        self._trials_table.add_column("Depth", width=5, justify="right")
        self._trials_table.add_column("Width", width=5, justify="right")
        self._trials_table.add_column("LR", width=8, justify="right")
        self._trials_table.add_column("Loss", width=10, justify="right")
        self._trials_table.add_column("Status", width=8)

        for data in self._trials_data:
            self._trials_table.add_row(
                str(data["trial"]),
                str(data["depth"]),
                str(data["width"]),
                data["lr"],
                data["loss"],
                data["status"],
            )

    def _rebuild_best_table(self) -> None:
        best_title = f"Top {self.config.top_k} Configs"
        self._best_table = Table(
            title=best_title, show_header=True, header_style="bold green"
        )
        self._best_table.add_column("Rank", width=4, justify="right")
        self._best_table.add_column("Depth", width=5, justify="right")
        self._best_table.add_column("Width", width=5, justify="right")
        self._best_table.add_column("Loss", width=10, justify="right")

        for rank, (params, loss) in enumerate(self._best_configs, 1):
            depth = str(params.get("depth", "?"))
            width = str(params.get("width", "?"))
            self._best_table.add_row(str(rank), depth, width, f"{loss:.4f}")

    def __enter__(self) -> "OptunaProgressDisplay":
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: ANN401
        self._live.__exit__(*args)


def check_capacity(config: SearchSpaceConfig) -> None:
    """Verify largest architecture fits in memory."""
    max_depth = config.depth_range[1]
    max_width = max(config.width_choices)

    test_config = HyperParams(
        hidden_dims=tuple([max_width] * max_depth),
        learning_rate_max=1e-3,
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
        msg = "Reduce search space bounds - OOM on max architecture"
        raise RuntimeError(msg) from e


def build_hyperparams(trial: optuna.Trial, config: SearchSpaceConfig) -> HyperParams:
    """Define-by-run search space construction."""
    depth = trial.suggest_int("depth", *config.depth_range)
    width = trial.suggest_categorical("width", config.width_choices)

    lr_max = trial.suggest_float("lr_max", *config.lr_max_range, log=True)
    lr_min_ratio = trial.suggest_float("lr_min_ratio", *config.lr_min_ratio_range, log=True)

    weight_decay = trial.suggest_float(
        "weight_decay", *config.weight_decay_range, log=True
    )
    weight_boundary = trial.suggest_float(
        "weight_boundary", *config.weight_boundary_range, log=True
    )
    sigma_residual = trial.suggest_float("sigma_residual", *config.sigma_residual_range)

    n_rz_inner = trial.suggest_categorical("n_rz_inner", config.n_rz_inner_choices)
    warmup = trial.suggest_int("warmup_epochs", *config.warmup_range, step=25)
    decay = trial.suggest_int("decay_epochs", *config.decay_range, step=100)

    return HyperParams(
        hidden_dims=tuple([width] * depth),
        learning_rate_max=lr_max,
        learning_rate_min=lr_max * lr_min_ratio,
        weight_decay=weight_decay,
        weight_boundary_condition=weight_boundary,
        sigma_residual_adaptive_sampling=sigma_residual,
        n_rz_inner_samples=n_rz_inner,
        n_rz_boundary_samples=config.n_rz_boundary,
        warmup_epochs=warmup,
        decay_epochs=decay,
        batch_size=config.batch_size,
        n_train=config.n_train,
    )


def objective(
    trial: optuna.Trial, config: SearchSpaceConfig, display: OptunaProgressDisplay
) -> float:
    """Train single trial with pruning support."""
    hyperparams = build_hyperparams(trial, config)
    params_for_display = {
        "depth": len(hyperparams.hidden_dims),
        "width": hyperparams.hidden_dims[0],
        "lr": hyperparams.learning_rate_max,
    }

    try:
        manager = NetworkManager(hyperparams)
        total_epochs = hyperparams.warmup_epochs + hyperparams.decay_epochs

        loss = float("inf")
        for epoch in range(total_epochs):
            loss, _, _ = manager.train_epoch(epoch)

            if epoch >= config.min_epochs:
                trial.report(loss, epoch)
                if trial.should_prune():
                    display.update(trial.number + 1, params_for_display, None, "pruned")
                    raise optuna.TrialPruned

        jax.clear_caches()
        display.update(trial.number + 1, params_for_display, loss, "done")
        return loss

    except optuna.TrialPruned:
        raise
    except Exception:
        logger.exception(f"Trial {trial.number} failed")
        display.update(trial.number + 1, params_for_display, None, "failed")
        return float("inf")


def get_warmstart_config() -> dict[str, Any]:
    """Return best known configuration for warm-starting."""
    return {
        "depth": 4,
        "width": 128,
        "lr_max": 0.002,
        "lr_min_ratio": 0.025,
        "weight_decay": 1e-7,
        "weight_boundary": 10.0,
        "sigma_residual": 0.05,
        "n_rz_inner": 2048,
        "warmup_epochs": 100,
        "decay_epochs": 500,
    }


def _save_top_configs(
    results: list[tuple[HyperParams, float]], study: optuna.Study
) -> None:
    """Save top-k configs to JSON for later retrieval."""
    output_dir = Path("logs/hpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = output_dir / f"top_k_configs_{timestamp}.json"

    top_k_data = [
        {"rank": i + 1, "loss": loss, "config": hp.to_dict()}
        for i, (hp, loss) in enumerate(results)
    ]

    data = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "n_completed": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ),
        "n_pruned": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ),
        "best_loss": results[0][1] if results else None,
        "top_k": top_k_data,
    }

    output_file.write_text(json.dumps(data, indent=2))
    logger.info(f"Top-k configs saved to: {output_file}")


def run_optimization(
    config: SearchSpaceConfig | None = None,
) -> list[tuple[HyperParams, float]]:
    """Run full HPO and return top-k configs with losses."""
    config = config or SearchSpaceConfig()

    check_capacity(config)

    storage_path = Path("logs/hpo/optuna_study.db")
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="pinn_hpo",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=42, n_startup_trials=config.n_startup_trials),
        pruner=HyperbandPruner(
            min_resource=config.min_epochs,
            max_resource=config.max_epochs,
            reduction_factor=3,
        ),
    )

    study.enqueue_trial(get_warmstart_config())

    display = OptunaProgressDisplay(config)
    with display:
        study.optimize(
            lambda trial: objective(trial, config, display),
            n_trials=config.n_trials,
            show_progress_bar=False,
        )

    complete_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value < float("inf")
    ]
    top_trials = sorted(complete_trials, key=lambda t: t.value)[: config.top_k]

    results = []
    for trial in top_trials:
        hp = build_hyperparams(trial, config)
        results.append((hp, trial.value))

    console.print("\n" + "=" * 60)
    console.print("[bold green]HPO COMPLETE - TOP CONFIGURATIONS[/bold green]")
    console.print("=" * 60)
    for i, (hp, loss) in enumerate(results, 1):
        dims = f"{len(hp.hidden_dims)}x{hp.hidden_dims[0]}"
        console.print(f"[bold]Rank {i}:[/] {dims} | Loss: {loss:.4f}")
    console.print("=" * 60 + "\n")

    _save_top_configs(results, study)

    return results


def train_top_k_models(results: list[tuple[HyperParams, float]]) -> None:
    """Full training of top-k configurations found."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TRAINING TOP-K MODELS WITH FULL EPOCHS[/bold cyan]")
    console.print("=" * 60)

    for rank, (hp_config, hpo_loss) in enumerate(results, 1):
        dims = f"{len(hp_config.hidden_dims)}x{hp_config.hidden_dims[0]}"
        console.print(f"\n[bold]--- Training Rank {rank} ({dims}) ---[/bold]")
        console.print(f"  HPO Loss: {hpo_loss:.4f}")

        manager = NetworkManager(hp_config)
        manager.train(save_to_disk=True)

        if manager.training_log:
            final_loss = manager.training_log[-1]["moving_avg_loss"]
            console.print(f"  Final Training Loss: {final_loss:.4f}")

        jax.clear_caches()


if __name__ == "__main__":
    try:
        config = SearchSpaceConfig()
        top_results = run_optimization(config)
        if top_results:
            train_top_k_models(top_results)
    except KeyboardInterrupt:
        logger.info("\nOptimization interrupted. Study saved to SQLite.")
        jax.clear_caches()