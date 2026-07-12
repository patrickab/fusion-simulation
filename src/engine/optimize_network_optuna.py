"""Optuna hyperparameter search using the standard network training path."""

import argparse
from collections import deque
import json
import logging
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import jax
import optuna
from optuna.pruners import HyperbandPruner, NopPruner
from optuna.samplers import TPESampler
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, RichLog, Static

from src.engine.network import NetworkManager
from src.engine.network import logger as network_logger
from src.lib.config import current_commit
from src.lib.logger import get_logger
from src.lib.network_config import HyperParams

logger = get_logger(name="OptunaHPO")
console = Console(width=160)

HPO_ROOT = Path("logs/hpo")


def study_dir(study_name: str, commit: str | None = None) -> Path:
    """Per-study bundle under the running commit:

    ``logs/hpo/<commit>/<study>/{study.db, optuna.log, top_trials.json, trials.json}``.

    Mirrors the per-commit benchmark tree (``logs/benchmarks/<commit>/<run>/``):
    a commit is one code revision, and all HPO studies run against that revision
    land together. ``commit`` defaults to HEAD so callers don't have to thread it.
    """
    commit = commit or current_commit()
    path = HPO_ROOT / commit / study_name
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    n_trials: int = 5
    top_k: int = 3
    n_startup_trials: int = 10

    huber_delta: float = 1.0
    n_fourier_features: int = 64
    lbfgs_steps: int = 0
    study_name: str = "pinn_hpo"
    prune_trials: bool = True
    save_all: bool = False
    save_checkpoints: bool = True  # False (--test): keep no checkpoints/benchmark dirs


class OptunaProgressDisplay:
    """Rich live dashboard showing optimization progress and top configs."""

    def __init__(self, config: SearchSpaceConfig, prior_trials: int = 0, live: bool = True) -> None:
        self.config = config
        # Per-trial events (markup strings or Rich renderables) for the TUI detail view;
        # drained by HpoApp on its UI timer. RichLog snapshots renderables at write time.
        self.events: deque[Any] = deque(maxlen=2000)
        self._trial_rows: list[tuple[str, ...]] = []
        # Manager of the currently training trial; the TUI renders its metrics table live.
        self.current_manager: NetworkManager | None = None
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
        # live=False: HpoApp renders _build_layout() itself; no terminal-owning Live loop.
        self._live = (
            Live(
                self._build_layout(),
                refresh_per_second=4,
                console=console,
                vertical_overflow="visible",
            )
            if live
            else None
        )

    def _sync(self) -> None:
        if self._live is not None:
            self._live.update(self._build_layout())

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
        if trial_num > 1:
            self.events.append("")
            self.events.append("")
        self.events.append(f"[bold cyan]── trial {trial_num} ──[/]")
        self._sync()

    def update_epoch(self, epoch: int, val_loss: float | None = None) -> None:
        if not self._current_trial_info:
            return
        self._current_trial_info["epoch"] = epoch
        if val_loss is not None:
            self._current_trial_info["val_loss"] = val_loss
        self._progress.update(self._epoch_task, completed=epoch)
        self._sync()

    def add_metrics_row(self, row: tuple[str, ...]) -> None:
        """Collect finalized training-table rows; flushed as one styled table per trial."""
        self._trial_rows.append(row)

    def _flush_trial_table(self) -> None:
        """Write the finished trial's full metrics history as a styled Rich table."""
        if self._trial_rows and self.current_manager is not None:
            table = self.current_manager._new_table()
            for row in self._trial_rows:
                table.add_row(*row)
            self.events.append(table)
        self._trial_rows = []

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
        self._flush_trial_table()
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
        self.events.append(
            f"[bold cyan]trial {trial_num}[/] {status_text}"
            + (f"  val_loss {loss:.4f}" if loss is not None else "")
        )
        self._sync()

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
        weight_decay=trial.suggest_float("weight_decay", *config.weight_decay_range, log=True),
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
    commit: str,
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
    display.current_manager = manager
    manager.metrics_row_sink = display.add_metrics_row
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
            save_to_disk=config.save_all and config.save_checkpoints,
            validation_callback=report,
            show_progress=False,
        )
        if config.save_checkpoints and not config.save_all:
            completed = sorted(
                t.value
                for t in trial.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
                if t.value is not None
            )
            if len(completed) < config.top_k or val_loss < completed[config.top_k - 1]:
                trial.set_user_attr("checkpoint", manager.to_disk())
        elif config.save_all:  # train() already saved; stem doubles as the checkpoint name
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
        # trial's own run dir is touched — sibling runs on the same commit
        # (other experiments) are never deleted, and the commit dir is only
        # removed once empty.
        manager.discard_unsaved_run()
        _write_trials_json(trial.study, commit)
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
    if (
        config.depth_range[0] <= warmstart["depth"] <= config.depth_range[1]
        and warmstart["width"] in config.width_choices
    ):
        return warmstart
    return None


def _write_trials_json(study: optuna.Study, commit: str) -> None:
    """Ledger of which runs/checkpoints belong to this study.

    Rewritten after every trial so an aborted study still leaves an on-disk
    record; without it, benchmark run dirs on a shared commit could not be
    attributed to their experiment.
    """
    (study_dir(study.study_name, commit) / "trials.json").write_text(
        json.dumps(
            [
                {"trial": t.number, "state": t.state.name, "value": t.value, **t.user_attrs}
                for t in study.get_trials(deepcopy=False)
            ],
            indent=2,
        )
    )


def _save_top_configs(
    results: list[tuple[HyperParams, float]], study: optuna.Study, commit: str
) -> None:
    output_file = study_dir(study.study_name, commit) / "top_trials.json"
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
    display: OptunaProgressDisplay | None = None,
) -> list[tuple[HyperParams, float]]:
    """Run or resume a study, guarded by a per-study single-process lock.

    Concurrent create_study calls on one sqlite file race optuna's
    check-then-insert and corrupt the db with duplicate study rows
    (observed 2026-07-12: MultipleResultsFound on load). Crashed TUIs stay
    open by design, so a relaunch while one is alive is the common case.
    """
    config = config or SearchSpaceConfig()
    # Freeze the commit once for the whole run: a commit made mid-study would
    # otherwise move study.db / optuna.log / trials.json to a different path
    # and split the study across two commit dirs.
    commit = current_commit()
    lock = study_dir(config.study_name, commit) / ".lock"
    if lock.exists():
        pid = lock.read_text().strip()
        if pid.isdigit() and Path(f"/proc/{pid}").exists():
            raise RuntimeError(
                f"Study '{config.study_name}' is already running (pid {pid}). "
                "Concurrent runs corrupt the study db — quit the other process "
                "first (crashed TUIs stay open; press q there)."
            )
    # ponytail: check-then-write lock; atomic O_EXCL takeover not worth it for
    # the human-relaunch-after-crash case this guards.
    lock.write_text(str(os.getpid()))
    try:
        return _run_optimization(config, restart, display, commit)
    finally:
        lock.unlink(missing_ok=True)


def _run_optimization(
    config: SearchSpaceConfig,
    restart: bool,
    display: OptunaProgressDisplay | None,
    commit: str,
) -> list[tuple[HyperParams, float]]:
    """Run or resume a study and return its best configurations."""
    bundle_dir = study_dir(config.study_name, commit)
    file_handler = logging.FileHandler(bundle_dir / "optuna.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    check_capacity(config)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_path = bundle_dir / "study.db"
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
    if display is None:  # plain-console mode owns its own Live loop
        with OptunaProgressDisplay(config, prior_trials=prior_trials) as display:
            study.optimize(
                lambda trial: objective(trial, config, display, commit),
                n_trials=max(0, config.n_trials - prior_trials),
                catch=(Exception,),
            )
    else:  # TUI mode: HpoApp owns rendering, we just feed it state
        display._prior_trials = prior_trials
        study.optimize(
            lambda trial: objective(trial, config, display, commit),
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
    _save_top_configs(results, study, commit)
    _write_trials_json(study, commit)
    return results


class _EventLogHandler(logging.Handler):
    """Routes log records (incl. network.py's per-epoch DEBUG lines) into the detail pane."""

    def __init__(self, events: deque[str]) -> None:
        # INFO: the per-epoch DEBUG lines are already on screen as the metrics table
        super().__init__(logging.INFO)
        self._events = events

    def emit(self, record: logging.LogRecord) -> None:
        self._events.append(f"[dim]{record.levelname}[/]  " + escape(record.getMessage()))


class HpoApp(App):
    """Fullscreen HPO dashboard; Tab flips between study overview and per-trial log."""

    BINDINGS: ClassVar = [
        Binding("tab", "toggle_view", "overview/detail", priority=True),
        Binding("q", "quit", "quit"),
    ]
    CSS = """
    #overview-pane { height: 1fr; }
    #detail { height: 1fr; display: none; }
    """

    def __init__(self, config: SearchSpaceConfig, restart: bool) -> None:
        super().__init__()
        # ansi-dark passes the terminal's own palette/background through instead of
        # Textual's truecolor theme.
        self.theme = "ansi-dark"
        self._config, self._restart = config, restart
        self._state = OptunaProgressDisplay(config, live=False)
        handler = _EventLogHandler(self._state.events)
        for source in (logger, network_logger):
            source.addHandler(handler)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="overview-pane"):
            yield Static(id="overview")
        yield RichLog(id="detail", markup=True, wrap=True, max_lines=5000)
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.25, self._refresh)
        self.run_worker(self._run_study, thread=True)

    def _refresh(self) -> None:
        # ponytail: the worker thread mutates display state while we render it; the GIL makes
        # this tear-free enough for a dashboard — add a lock only if frames visibly glitch.
        self.query_one("#overview", Static).update(self._state._build_layout())
        log = self.query_one("#detail", RichLog)
        while self._state.events:
            log.write(self._state.events.popleft())

    def action_toggle_view(self) -> None:
        overview, detail = self.query_one("#overview-pane"), self.query_one("#detail")
        overview.display, detail.display = not overview.display, not detail.display

    def _run_study(self) -> None:
        try:
            run_optimization(self._config, restart=self._restart, display=self._state)
            self._state.events.append("[bold green]study complete[/] — press q to quit")
        except Exception as error:  # keep the app up so the log stays inspectable
            self._state.events.append(f"[bold red]study crashed:[/] {error!r}")


def _resolve_reset_choice(storage_path: Path, study_name: str, commit: str) -> bool:
    """Ask whether to reset an existing study db; only called when neither
    --reset-sqlite nor --resume-sqlite was passed.

    Shows the commit + study so the user knows exactly what they'd be
    resuming. A resumed study can silently break once the search space has
    changed since it was created (observed 2026-07-12: a stale trial crashed
    build_hyperparams against a since-changed width_choices). Non-interactive
    launches (e.g. over SSH) can't be prompted and must pass a flag.
    """
    if not storage_path.exists():
        return False
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"{storage_path} exists. Pass --reset-sqlite or --resume-sqlite "
            "explicitly — non-interactive launches can't be prompted."
        )
    print(f"\nExisting study found: {study_name} @ commit {commit}")
    print(f"  {storage_path}")
    return input("Reset database? (y/n): ").strip().lower().startswith("y")


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
        config.save_checkpoints = False  # tests must not pollute data/networks
        # Separate study name: --test's shrunk search space (depth/width/etc.)
        # is incompatible with a real run's distributions. Sharing a study
        # name let a --test run's leftover trials corrupt a later real run's
        # build_hyperparams() (ValueError: stale width choice not in new
        # categorical distribution) — observed 2026-07-12.
        config.study_name = f"{config.study_name}_test"

    if args.test:
        reset = True  # tests always start clean; never prompt
    elif args.reset_sqlite:
        reset = True
    elif args.resume_sqlite:
        reset = False
    else:
        commit = current_commit()
        reset = _resolve_reset_choice(
            study_dir(config.study_name, commit) / "study.db",
            config.study_name,
            commit,
        )

    # TUI whenever a terminal exists; piped/nohup output gets the plain Live dashboard,
    # which degrades to sequential text on non-ttys. All logging/benchmark artifacts
    # (study.db, optuna.log, trials.json, train.log, checkpoints) are identical either way.
    if sys.stdout.isatty():
        HpoApp(config, restart=reset).run()
    else:
        run_optimization(config, restart=reset)


if __name__ == "__main__":
    main()
