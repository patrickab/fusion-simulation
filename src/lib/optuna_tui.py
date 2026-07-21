"""Rich/Textual display layer for Optuna HPO: live dashboard, event-log handler, TUI app."""

from collections import deque
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable, ClassVar

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
from textual.geometry import Size
from textual.widgets import Footer, RichLog, Static

from src.engine.network import logger as network_logger
from src.lib.logger import get_logger
from src.lib.utils import format_hpo_params

if TYPE_CHECKING:
    import optuna

    from src.engine.network_manager import NetworkManager
    from src.engine.optimize_network_optuna import StudyConfig
    from src.lib.network_config import HyperParams

logger = get_logger(name="OptunaHPO")
console = Console(width=160)


class OptunaProgressDisplay:
    """Rich live dashboard showing optimization progress and top configs."""

    def __init__(
        self,
        study: "StudyConfig",
        prior_trials: int = 0,
        live: bool = True,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.config = study
        # State-change-triggered repaint instead of a fixed-rate timer poll (which
        # flickers over higher-latency SSH links); throttled to 1/s like network.py's
        # _MetricsManager so a fast-epoch trial doesn't flood the terminal.
        self._on_change = on_change
        self._last_refresh_at = 0.0
        # Detail-pane events (markup strings or Rich renderables); drained by
        # HpoApp._refresh, invoked via on_change.
        self.events: deque[Any] = deque(maxlen=2000)
        self.current_manager: NetworkManager | None = None
        # Rolling window: an ever-growing table eventually exceeds the terminal height
        # and desynchronizes Rich's cursor-based redraw when the terminal scrolls.
        self._trials_data: deque[dict[str, Any]] = deque(maxlen=15)
        self._best_configs: list[tuple[dict[str, Any], float, float | None, float | None]] = []
        self._best_loss, self._start_time = float("inf"), datetime.now()
        self._counts = {"pruned": 0, "failed": 0, "done": 0}
        self._trials_processed, self._prior_trials = 0, prior_trials
        self._warmstart_trials = 0
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

    def _sync(self, force: bool = False) -> None:
        if self._live is not None:
            self._live.update(self._build_layout())
        if self._on_change is not None:
            now = time.monotonic()
            if force or now - self._last_refresh_at >= 1.0:
                self._last_refresh_at = now
                self._on_change()

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
            "Median",
            "P95",
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
                        "median",
                        "p95",
                        "status",
                    )
                ]
            )
        return table

    def get_live_training_panel(self) -> Panel | None:
        """NetworkManager's own standalone-`train()` dashboard, reused as-is so a trial
        running inside the HPO TUI looks identical to training it alone. Detail pane
        only (see HpoApp._refresh) — the overview pane stays a static summary."""
        manager = self.current_manager
        if manager is None or getattr(manager, "_metrics", None) is None:
            return None
        panel = manager.training_renderable()
        panel.border_style = "magenta"
        if trial := self._current_trial_info.get("trial"):
            panel.title = f"Current Trial: {trial}"
        return panel

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
            "Median",
            "P95",
        ]:
            table.add_column(column, justify="right")
        for rank, (params, _loss, median, p95) in enumerate(self._best_configs, 1):
            fmt = format_hpo_params(params)
            table.add_row(
                str(rank),
                str(params.get("depth", "?")),
                str(params.get("width", "?")),
                fmt["lr_max"],
                fmt["lr_min"],
                fmt["wd"],
                fmt["sig"],
                f"{median:.4f}" if median is not None else "--",
                f"{p95:.4f}" if p95 is not None else "--",
            )
        return table

    def _build_layout(self) -> Panel:
        elapsed_str = str(datetime.now() - self._start_time).split(".")[0]
        summary = (
            f"[bold]Best Val Loss:[/] {self._best_loss:.4f}  |  "
            f"[bold]Elapsed:[/] {elapsed_str}\n"
            f"Session: {self._counts['done']} done  |  {self._counts['pruned']} pruned  |  "
            f"{self._counts['failed']} failed"
            + (f"  |  Warmstart: {self._warmstart_trials}" if self._warmstart_trials else "")
            + (f"  |  Prior: {self._prior_trials}" if self._prior_trials else "")
        )

        current_table = Table(show_header=False, box=box.SIMPLE)
        if self._current_trial_info:
            current = self._current_trial_info
            params = current["params"]
            val_loss = current.get("val_loss")
            title = f"Current Trial: {current['trial']}"
            fmt = format_hpo_params(params)
            rows = [
                ("Architecture:", f"{params.get('depth')}x{params.get('width')}"),
                ("Max LR:", fmt["lr_max"]),
                ("Min LR:", fmt["lr_min"]),
                ("Weight Decay:", fmt["wd"]),
                ("Sigma Res:", fmt["sig"]),
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
        self._sync(force=True)

    def update_epoch(self, epoch: int, val_loss: float | None = None) -> None:
        if not self._current_trial_info:
            return
        self._current_trial_info["epoch"] = epoch
        if val_loss is not None:
            self._current_trial_info["val_loss"] = val_loss
        self._progress.update(self._epoch_task, completed=epoch)
        self._sync()

    def add_metrics_row(self, _row: tuple[str, ...]) -> None:
        """Sink for network.py's per-logged-epoch rows; row data lives on
        current_manager._metrics, this just triggers a throttled repaint."""
        self._sync()

    def _flush_trial_table(self, trial_num: int) -> None:
        """Write the finished trial's standalone training dashboard (same renderable
        shown live) into the scrollable detail log."""
        manager = self.current_manager
        if manager is not None and getattr(manager, "_metrics", None) is not None:
            panel = manager.training_renderable()
            panel.title = f"Trial {trial_num} — final"
            panel.border_style = "cyan"
            self.events.append(panel)

    def add_warmstart_trials(
        self, candidates: list[tuple["HyperParams", float, float | None, float | None]]
    ) -> None:
        """Seed the trials table + best-configs list with injected warmstart
        trials so they're visible (but not counted as live progress)."""
        for hp, loss, median, p95 in candidates:
            params = {
                "depth": len(hp.hidden_dims),
                "width": hp.hidden_dims[0],
                "lr_max": hp.learning_rate_max,
                "lr_min": hp.learning_rate_min,
                "wd": hp.weight_decay,
                "sig": hp.sigma_residual_adaptive_sampling,
            }
            self._best_configs.append((params.copy(), loss, median, p95))
            self._best_configs.sort(key=lambda item: item[1])
            self._best_configs = self._best_configs[: self.config.top_k]
            self._best_loss = self._best_configs[0][1] if self._best_configs else self._best_loss
            self._trials_data.append(
                {
                    "trial": f"ws{len(self._trials_data)}",
                    "depth": params["depth"],
                    "width": params["width"],
                    **format_hpo_params(params),
                    "median": f"{median:.4f}" if median is not None else "--",
                    "p95": f"{p95:.4f}" if p95 is not None else "--",
                    "loss": f"{loss:.4f}" if loss is not None else "--",
                    "status": "[blue]warmstart[/]",
                }
            )
        self._sync(force=True)

    def update(
        self,
        trial_num: int,
        params: dict[str, Any],
        loss: float | None,
        status: str,
        epoch: int | None = None,
        median: float | None = None,
        p95: float | None = None,
    ) -> None:
        self._trials_processed += 1
        self._progress.update(self._task, completed=self._trials_processed)
        self._progress.update(self._epoch_task, visible=False)
        self._current_trial_info = {}
        self._flush_trial_table(trial_num)
        if status in self._counts:
            self._counts[status] += 1

        if loss is not None:
            self._best_configs.append((params.copy(), loss, median, p95))
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
                **format_hpo_params(params),
                "median": f"{median:.4f}" if median is not None else "--",
                "p95": f"{p95:.4f}" if p95 is not None else "--",
                "loss": f"{loss:.4f}" if loss is not None else "--",
                "status": status_text,
            }
        )
        self.events.append(
            f"[bold cyan]trial {trial_num}[/] {status_text}"
            + (f"  median {median:.4f}  p95 {p95:.4f}" if median is not None else "")
        )
        self._sync(force=True)

    def __enter__(self) -> "OptunaProgressDisplay":
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:  # noqa: ANN401
        self._live.__exit__(*args)


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

    def __init__(
        self,
        study: "StudyConfig",
        restart: bool,
        trial_callback: "Callable[[optuna.Study, optuna.trial.FrozenTrial], None] | None" = None,
    ) -> None:
        super().__init__()
        # ansi-dark passes the terminal's own palette/background through instead of
        # Textual's truecolor theme.
        self.theme = "ansi-dark"
        self._study, self._restart = study, restart
        self._trial_callback = trial_callback
        self._state = OptunaProgressDisplay(study, live=False, on_change=self._request_refresh)
        handler = _EventLogHandler(self._state.events)
        # get_logger() wires each source to a Console(stderr=True) RichHandler, whose direct
        # terminal writes land outside Textual's alt-screen buffer and corrupt the display.
        # Swap them out for the run (restored in on_unmount).
        self._log_sources = (logger, network_logger)
        self._original_handlers = {source: source.handlers[:] for source in self._log_sources}
        for source in self._log_sources:
            source.handlers = [handler]
        # Line count of the live training panel's last write, so _refresh can trim
        # just that tail instead of re-rendering the whole log every tick.
        self._live_line_count = 0
        # Filled by _run_study on a crash; read after .run() returns so the caller
        # can print the traceback to the real terminal once Textual has restored it
        # (prints inside the alternate screen would be wiped on exit).
        self.crash_traceback: str | None = None

    def on_unmount(self) -> None:
        for source, handlers in self._original_handlers.items():
            source.handlers = handlers

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="overview-pane"):
            yield Static(id="overview")
        yield RichLog(id="detail", markup=True, wrap=True, max_lines=5000)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()  # paint the empty-state layout immediately; further frames are event-driven
        self.run_worker(self._run_study, thread=True)

    def _request_refresh(self) -> None:
        # Called from the worker thread (OptunaProgressDisplay._sync); call_from_thread
        # marshals the repaint onto Textual's own event loop, the only thread allowed
        # to touch widgets.
        self.call_from_thread(self._refresh)

    def _refresh(self) -> None:
        self.query_one("#overview", Static).update(self._state._build_layout())
        log = self.query_one("#detail", RichLog)
        was_at_bottom = log.is_vertical_scroll_end

        # Drop only the previous live-panel tail (rewritten below); history above it is
        # never touched, so a scrolled-up view holds its position instead of jumping on
        # every throttled tick.
        if self._live_line_count:
            del log.lines[-self._live_line_count :]
            log._line_cache.clear()  # stale entries would otherwise show old tail content
            self._live_line_count = 0

        while self._state.events:
            log.write(self._state.events.popleft(), scroll_end=False)

        if (live_panel := self._state.get_live_training_panel()) is not None:
            lines_before = len(log.lines)
            log.write(live_panel, scroll_end=False)
            self._live_line_count = len(log.lines) - lines_before

        log.virtual_size = Size(log._widest_line_width, len(log.lines))
        log.refresh()
        if was_at_bottom:
            log.scroll_end(animate=False, immediate=True)

    def action_toggle_view(self) -> None:
        overview, detail = self.query_one("#overview-pane"), self.query_one("#detail")
        overview.display, detail.display = not overview.display, not detail.display

    def _run_study(self) -> None:
        # Deferred import: optimize_network_optuna imports HpoApp/OptunaProgressDisplay from
        # this module at module scope, so a top-level import back here would deadlock.
        from src.engine.optimize_network_optuna import run_optimization

        try:
            run_optimization(
                self._study,
                restart=self._restart,
                display=self._state,
                trial_callback=self._trial_callback,
            )
        except KeyboardInterrupt:
            # run_optimization already removed the study dir; just exit cleanly.
            pass
        except Exception:
            # Stashed for the caller to print after .run() returns, once Textual's
            # alt-screen is gone; printing here would be wiped on exit.
            self.crash_traceback = traceback.format_exc()
            logger.error(f"study crashed:\n{self.crash_traceback}")
        finally:
            # Always exit: lets sequential drivers (tmux benchmark runs) proceed
            # without waiting for 'q'.
            self.call_from_thread(self.exit)


def resolve_reset_choice(storage_path: Path, study_name: str, commit: str) -> bool:
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
