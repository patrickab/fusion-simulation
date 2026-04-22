"""Optuna-based hyperparameter optimization with physics-based validation.

Uses a fixed validation set of Sobol-sampled plasma configs to evaluate generalization.
Validation loss is computed every val_eval_frequency epochs for pruning decisions,
avoiding evaluation overhead on every epoch while still providing unbiased HPO signal.
"""

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

from src.engine.network import NetworkManager, Sampler
from src.lib.logger import get_logger
from src.lib.network_config import DomainBounds, FluxInput, HyperParams

logger = get_logger(name="OptunaHPO", log_dir="logs/hpo")
console = Console()


@dataclass
class SearchSpaceConfig:
    """Hyperparameter search bounds for Optuna HPO."""

    depth_range: tuple[int, int] = (3, 5)
    width_choices: tuple[int, ...] = (128, 160, 192)
    lr_max_range: tuple[float, float] = (1e-4, 5e-3)
    lr_min_ratio_range: tuple[float, float] = (0.01, 0.1)
    weight_decay_range: tuple[float, float] = (1e-9, 1e-5)
    sigma_residual_range: tuple[float, float] = (0.01, 0.2)
    n_rz_inner_choices: tuple[int, ...] = (512, 1024, 2048)

    weight_boundary_condition: float = 20.0
    total_epochs: int = 600
    warmup_ratio_range: tuple[float, float] = (0.1, 0.25)

    batch_size: int = 64
    n_train: int = 1024
    n_rz_boundary: int = 256

    n_validate: int = 256
    val_eval_frequency: int = 20

    min_epochs: int = 50
    n_trials: int = 50
    top_k: int = 3
    n_startup_trials: int = 10


def create_validation_inputs(config: SearchSpaceConfig, sampler_config: HyperParams) -> FluxInput:
    """Create fixed validation inputs using a Sampler with different seed.

    Reuses Sampler's Sobol infrastructure for consistency with training.
    """
    sampler = Sampler(sampler_config, seed=123)
    lower, upper = DomainBounds.get_bounds()
    plasma_configs = sampler._get_sobol_sample(
        n_samples=config.n_validate,
        lower_bounds=lower,
        upper_bounds=upper,
        sobol_sampler="domain",
    )
    return sampler.sample_flux_input(plasma_configs=plasma_configs)


class OptunaProgressDisplay:
    """Rich live dashboard showing optimization progress and top configs."""

    def __init__(self, config: SearchSpaceConfig) -> None:
        self.config = config
        self._trials_data: list[dict[str, Any]] = []
        self._best_configs: list[tuple[dict[str, Any], float]] = []
        self._best_loss = float("inf")
        self._n_pruned = 0
        self._n_failed = 0
        self._n_completed = 0
        self._start_time = datetime.now()

        self._progress = Progress(
            TextColumn("[bold cyan]Progress:"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TextColumn("{task.completed}/{task.total} trials"),
            TextColumn("({task.percentage:.0f}%)"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task = self._progress.add_task("", total=self.config.n_trials)
        self._trials_table = Table(show_header=True, header_style="bold cyan")
        self._best_table = Table(show_header=True, header_style="bold green")

        self._live = Live(self._build_layout(), refresh_per_second=4, console=console)

    def _build_layout(self) -> Panel:
        elapsed = datetime.now() - self._start_time
        secs = elapsed.seconds
        elapsed_str = f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"
        summary = (
            f"[bold]Best Val Loss:[/] {self._best_loss:.4f}  |  "
            f"[bold]Elapsed:[/] {elapsed_str}\n"
            f"Completed: {self._n_completed}  |  Pruned: {self._n_pruned}  "
            f"| Failed: {self._n_failed}"
        )
        return Panel(
            Group(self._progress, summary, self._trials_table, self._best_table),
            title="[bold cyan]PINN HPO Optimization[/bold cyan]",
            border_style="cyan",
        )

    def update(
        self, trial_num: int, params: dict[str, Any], loss: float | None, status: str
    ) -> None:
        self._progress.update(self._task, completed=trial_num)

        if status == "pruned":
            self._n_pruned += 1
        elif status == "failed":
            self._n_failed += 1
        elif status == "done":
            self._n_completed += 1

        if loss is not None:
            self._best_configs.append((params.copy(), loss))
            self._best_configs.sort(key=lambda x: x[1])
            self._best_configs = self._best_configs[: self.config.top_k]
            self._best_loss = self._best_configs[0][1]
            loss_str = f"{loss:.4f}"
        else:
            loss_str = "--"

        status_str = {
            "done": "[green]done[/]", "pruned": "[yellow]pruned[/]", "failed": "[red]failed[/]"
        }.get(status, status)

        self._trials_data.append({
            "trial": trial_num,
            "depth": params.get("depth", "?"),
            "width": params.get("width", "?"),
            "lr": f"{params.get('lr', 0):.0e}",
            "loss": loss_str,
            "status": status_str,
        })
        if len(self._trials_data) > 8:
            self._trials_data.pop(0)

        self._trials_table = Table(
            title="Recent Trials", show_header=True, header_style="bold cyan",
        )
        for col in ["Trial", "Depth", "Width", "LR", "Val Loss", "Status"]:
            self._trials_table.add_column(col, width=5 if col != "Status" else 7, justify="right")
        for d in self._trials_data:
            self._trials_table.add_row(
                str(d["trial"]), str(d["depth"]), str(d["width"]), d["lr"], d["loss"], d["status"]
            )

        self._best_table = Table(
            title=f"Top {self.config.top_k} Configs", show_header=True, header_style="bold green",
        )
        for col in ["Rank", "Depth", "Width", "LR", "Val Loss"]:
            w = 9 if col in ("Rank", "Val Loss") else 5
            self._best_table.add_column(col, width=w, justify="right")
        for rank, (p, loss) in enumerate(self._best_configs, 1):
            self._best_table.add_row(
                str(rank), str(p.get("depth", "?")), str(p.get("width", "?")),
                f"{p.get('lr', 0):.0e}", f"{loss:.4f}"
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
    n_rz_inner = trial.suggest_categorical("n_rz_inner", config.n_rz_inner_choices)

    warmup_ratio = trial.suggest_float("warmup_ratio", *config.warmup_ratio_range)

    return HyperParams(
        hidden_dims=tuple([width] * depth),
        learning_rate_max=lr_max,
        learning_rate_min=lr_max * lr_min_ratio,
        weight_decay=weight_decay,
        weight_boundary_condition=config.weight_boundary_condition,
        sigma_residual_adaptive_sampling=sigma_residual,
        n_rz_inner_samples=n_rz_inner,
        n_rz_boundary_samples=config.n_rz_boundary,
        warmup_epochs=int(config.total_epochs * warmup_ratio),
        decay_epochs=config.total_epochs - int(config.total_epochs * warmup_ratio),
        batch_size=config.batch_size,
        n_train=config.n_train,
    )


def objective(
    trial: optuna.Trial,
    config: SearchSpaceConfig,
    display: OptunaProgressDisplay,
    val_inputs: FluxInput,
) -> float:
    """Train trial and return validation loss for unbiased HPO.

    Validation loss evaluated every val_eval_frequency epochs to balance
    generalization signal with computational overhead.
    """
    hp = build_hyperparams(trial, config)
    params_for_display = {
        "depth": len(hp.hidden_dims),
        "width": hp.hidden_dims[0],
        "lr": hp.learning_rate_max,
    }

    try:
        manager = NetworkManager(hp)
        total_epochs = hp.warmup_epochs + hp.decay_epochs
        val_loss = float("inf")

        for epoch in range(total_epochs):
            manager.train_epoch(epoch)

            if epoch >= config.min_epochs and epoch % config.val_eval_frequency == 0:
                val_loss = manager.calculate_loss(val_inputs)
                trial.report(val_loss, epoch)

                if trial.should_prune():
                    display.update(trial.number + 1, params_for_display, None, "pruned")
                    raise optuna.TrialPruned

        val_loss = manager.calculate_loss(val_inputs)
        jax.clear_caches()
        display.update(trial.number + 1, params_for_display, val_loss, "done")
        return val_loss

    except optuna.TrialPruned:
        raise
    except Exception:
        logger.exception(f"Trial {trial.number} failed")
        display.update(trial.number + 1, params_for_display, None, "failed")
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
        "n_rz_inner": 2048,
        "warmup_ratio": 0.167,
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


def run_optimization(config: SearchSpaceConfig | None = None) -> list[tuple[HyperParams, float]]:
    """Execute HPO study with validation-based pruning and return top-k configs."""
    config = config or SearchSpaceConfig()
    check_capacity(config)

    sampler_hp = HyperParams(
        n_rz_inner_samples=config.n_validate,
        n_rz_boundary_samples=config.n_rz_boundary,
        batch_size=config.n_validate,
    )
    val_inputs = create_validation_inputs(config, sampler_hp)

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
            max_resource=config.total_epochs,
            reduction_factor=3,
        ),
    )

    study.enqueue_trial(get_warmstart_config())

    display = OptunaProgressDisplay(config)
    with display:
        study.optimize(
            lambda trial: objective(trial, config, display, val_inputs),
            n_trials=config.n_trials,
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
        console.print(f"  n_rz_inner:    {hp.n_rz_inner_samples}")
        console.print(f"  Warmup epochs: {hp.warmup_epochs}")
    console.print("\n" + "=" * 80 + "\n")

    _save_top_configs(results, study)
    return results


def train_top_k_models(results: list[tuple[HyperParams, float]]) -> None:
    """Train top-k configs from scratch with full epochs and save to disk."""
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]TRAINING TOP-K MODELS WITH FULL EPOCHS[/bold cyan]")
    console.print("=" * 80)

    for rank, (hp, hpo_loss) in enumerate(results, 1):
        dims = f"{len(hp.hidden_dims)}x{hp.hidden_dims[0]}"
        console.print(f"\n[bold]--- Rank {rank}: {dims} ---[/bold]")
        console.print(f"  HPO Val Loss: {hpo_loss:.4f}")

        manager = NetworkManager(hp)
        manager.train(save_to_disk=True)

        if manager.training_log:
            console.print(f"  Final Train Loss: {manager.training_log[-1]['moving_avg_loss']:.4f}")

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

