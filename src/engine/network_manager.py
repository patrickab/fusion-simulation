import argparse
from contextlib import nullcontext, suppress
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Callable

import flax.serialization
from flax.training import train_state
import jax
import plotext
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.measure import Measurement
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from src.engine.network import (
    BASE_SEED,
    EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_ROLLING_WINDOW,
    N_VALIDATION_SIZE,
    VALIDATION_FREQUENCY,
    FluxPINN,
    FoundationModel,
    Sampler,
    Trainer,
)
from src.lib.config import Filepaths, current_commit
from src.lib.geometry_config import PlasmaConfig
from src.lib.logger import get_logger
from src.lib.network_config import EpochMetrics, HyperParams
from src.lib.run_artifacts import format_duration, gpu_name, kpi_values, load_run, write_json

console = Console()
logger = get_logger(
    name="Network",
)

LOG_FREQUENCY = 10  # Refresh the live table and flush metrics every N epochs
CHART_HEIGHT = 18  # fixed fallback for the static --show replay (no live terminal to size against)
CHART_MIN_HEIGHT = 10  # never shrink the live charts below this even in short terminals


# ---------------------------------------------------------------------------
# Terminal charts (plotext rendered as Rich text) + _MetricsManager
# ---------------------------------------------------------------------------


class _PlotextChart:
    """A plotext figure as a Rich renderable.

    Rebuilt only when the data length or terminal width changes — Rich Live
    re-renders at 10 Hz, so the built ANSI text is cached between epochs.
    """

    def __init__(self, draw: Callable[[list[dict]], None], rows: list[dict], height: int) -> None:
        self._draw = draw
        self._rows = rows
        self._height = height
        self._cache: tuple[tuple[int, int], Text] | None = None

    def __rich_console__(self, console: Console, options: any) -> any:
        key = (options.max_width, len(self._rows))
        if self._cache is None or self._cache[0] != key:
            try:
                plotext.clf()
                plotext.plotsize(options.max_width - 2, self._height)
                plotext.theme("pro")  # transparent bg, default fg: inherits terminal colors
                self._draw(self._rows)
                built = Text.from_ansi(plotext.build())
            except Exception as exc:  # a chart bug must never kill a training run
                built = Text(f"chart unavailable: {exc!r}", style="dim")
            self._cache = (key, built)
        yield self._cache[1]

    def __rich_measure__(self, console: Console, options: any) -> Measurement:
        # Charts stretch to whatever width they're given (lets a grid split evenly).
        return Measurement(20, options.max_width)


def _sci(x: float) -> str:
    """Compact scientific notation: 2e-3, 5e-5, 1.5e-3 (trailing zeros trimmed)."""
    mantissa, exp = f"{x:.1e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"


def _config_summary(config: "HyperParams") -> Table:
    """Two-row architecture/operating hyperparameter header for the metrics panel."""
    dims = config.hidden_dims
    layers = f"{dims[0]}×{len(dims)}" if len(set(dims)) == 1 else "-".join(map(str, dims))  # noqa: RUF001
    sep = " [dim]·[/] "
    architecture = sep.join(
        (
            f"[dim]arch[/] {config.arch}",
            f"[dim]layers[/] {layers}",
            f"[dim]soft_bc[/] {'on' if config.soft_bc else 'off'}",
            f"[dim]rwf[/] {'on' if config.rwf else 'off'}",
        )
    )
    total = config.warmup_epochs + config.decay_epochs
    operating = sep.join(
        (
            f"[dim]lr[/] {_sci(config.learning_rate_max)}→{_sci(config.learning_rate_min)}",
            f"[dim]epochs[/] {total} ({config.warmup_epochs}w/{config.decay_epochs}d)",
            f"[dim]wd[/] {_sci(config.weight_decay)}",
            f"[dim]batch[/] {config.batch_size}",
            f"[dim]σ_resample[/] {config.sigma_residual_adaptive_sampling:g}",  # noqa: RUF001
        )
    )
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", style="bold")
    grid.add_column()
    grid.add_row("architecture", architecture)
    grid.add_row("operating", operating)
    return grid


def _charts_renderable(rows: list[dict], chart_height: int) -> any:
    """Validation, loss and lr/||∇L|| charts side by side; validation joins after the first run."""
    if not rows:
        return None
    loss_chart = _chart_block(
        "Loss",
        Text.assemble(("● train", "cyan"), "  ", ("● val", "green")),
        _PlotextChart(_draw_loss_chart, rows, chart_height),
    )
    lr_chart = _chart_block(
        "Optimization",
        Text.assemble(("● lr", "yellow"), "  ", ("● ||∇L||", "magenta")),
        _PlotextChart(_draw_lr_grad_chart, rows, chart_height),
    )
    grid = Table.grid(expand=True, padding=(0, 2), collapse_padding=True, pad_edge=False)
    if not any(r["val_kpi_p50"] is not None for r in rows):
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(loss_chart, lr_chart)
        return Group(Text(""), grid)
    validation_chart = _chart_block(
        "Validation |R_GS|",
        Text.assemble(
            ("● p95", "grey62"),
            "  ",
            ("● p50", "green"),
            "  ",
            ("● p05", "grey62"),
        ),
        _PlotextChart(_draw_validation_chart, rows, chart_height),
    )
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(validation_chart, loss_chart, lr_chart)
    return Group(Text(""), grid)


def _chart_block(title: str, legend: Text, chart: _PlotextChart) -> Group:
    """Chart heading and legend outside plotext so they never cover data."""
    header = Table.grid(expand=True, padding=(0, 1))
    header.add_column()
    header.add_column(justify="right")
    header.add_row(Text(title, style="bold"), legend)
    return Group(header, chart)


def _log_series(x: list, y: list) -> tuple[list, list]:
    """Drop points with y <= 0 — plotext's log scale raises on them (e.g. warmup lr=0)."""
    pairs = [(a, b) for a, b in zip(x, y, strict=False) if b is not None and b > 0]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _log_ticks(values: list[float]) -> tuple[list[float], list[str]]:
    """1-2-5 log-decade ticks covering the data range, labeled in scientific notation."""
    lo, hi = min(values), max(values)
    ticks, labels = [], []
    for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            if lo <= m * 10.0**k <= hi:
                ticks.append(m * 10.0**k)
                labels.append(f"{m}e{k}")
    thin = max(1, math.ceil(len(ticks) / 6))
    return ticks[::thin], labels[::thin]


def _linear_ticks(values: list[float]) -> tuple[list[float], list[str]]:
    """Round-number ticks (1-2-5 steps) for a linear axis."""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = 10.0 ** math.floor(math.log10(span / 5))
    for mult in (1, 2, 5, 10):
        if span / (step * mult) <= 5:
            step *= mult
            break
    ticks = [t * step for t in range(math.ceil(lo / step), math.floor(hi / step) + 1)]
    return ticks, [f"{t:g}" for t in ticks]


def _draw_validation_chart(rows: list[dict]) -> None:
    """Validation |R_GS| p05/p50/p95 vs epoch, log y. Colors match the table's Val KPI column."""
    val_rows = [r for r in rows if r["val_kpi_p50"] is not None]
    all_epochs, all_vals = [], []
    for key, color in (
        ("val_kpi_p95", "gray"),
        ("val_kpi_p50", "green"),
        ("val_kpi_p05", "gray"),
    ):
        epochs, series = _log_series([r["epoch"] for r in val_rows], [r.get(key) for r in val_rows])
        if epochs:
            plotext.plot(epochs, series, marker="braille", color=color)
            all_epochs += epochs
            all_vals += series
    plotext.yscale("log")
    if all_vals:
        plotext.yticks(*_log_ticks(all_vals))
        plotext.xticks(*_linear_ticks(all_epochs))
    plotext.ylabel("|R_GS|")
    plotext.xlabel("epoch")


def _draw_loss_chart(rows: list[dict]) -> None:
    """Train loss and validation median residual vs epoch, shared log y-axis."""
    epochs = [r["epoch"] for r in rows]
    train_x, train_y = _log_series(epochs, [r["loss"] for r in rows])
    val_x, val_y = _log_series(epochs, [r.get("val_kpi_p50") for r in rows])
    all_vals = []
    if train_x:
        plotext.plot(train_x, train_y, marker="braille", color="cyan")
        all_vals += train_y
    if val_x:
        plotext.plot(val_x, val_y, marker="braille", color="green")
        all_vals += val_y
    plotext.yscale("log")
    if all_vals:
        plotext.yticks(*_log_ticks(all_vals))
        plotext.xticks(*_linear_ticks(epochs))
    plotext.ylabel("loss")
    plotext.xlabel("epoch")


def _draw_lr_grad_chart(rows: list[dict]) -> None:
    """lr (left axis) and ||∇L|| (right axis) vs epoch, both log y."""
    epochs = [r["epoch"] for r in rows]
    lr_x, lr_y = _log_series(epochs, [r["lr"] for r in rows])
    gn_x, gn_y = _log_series(epochs, [r["grad_norm"] for r in rows])
    if lr_x:
        plotext.plot(lr_x, lr_y, marker="braille", color="orange")
        plotext.yticks(*_log_ticks(lr_y))
    if gn_x:
        plotext.plot(gn_x, gn_y, marker="braille", color="magenta", yside="right")
        plotext.yticks(*_log_ticks(gn_y), yside="right")
    plotext.yscale("log")
    plotext.yscale("log", yside="right")
    if epochs:
        plotext.xticks(*_linear_ticks(epochs))
    plotext.ylabel("lr")
    plotext.ylabel("||∇L||", yside="right")
    plotext.xlabel("epoch")


class _MetricsManager:
    """Owns the Rich training table, progress bar, and completed metric windows."""

    def __init__(self, total_epochs: int, config: "HyperParams | None" = None) -> None:
        self._total_epochs = total_epochs
        self._config = config
        self.rows: list[dict] = []
        self._table_rows: list[tuple[str, ...]] = []
        self._progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(style="cyan", complete_style="bold cyan"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self._epoch_task = self._progress.add_task("Training", total=total_epochs)
        self._acc_loss = self._acc_res = self._acc_bnd = self._acc_gn = self._acc_t = 0.0
        self._acc_count = 0
        self._live: Live | None = None
        self._last_refresh_at = 0.0
        self.metrics_row_sink: Callable[[tuple[str, ...]], None] | None = None

    def renderable(self) -> Panel:
        """2x2 layout: metrics table + loss chart on top, validation + lr/||∇L|| below.

        The top and bottom rows split the panel's remaining height evenly, so the table/loss
        row and the validation/lr row each get half the screen. The table shows as many of
        the most recent epochs as fit in its half rather than a fixed row count.
        """
        parts: list[any] = [
            Align.center(Text("Training Metrics", style="italic")),
            Align.center(self._progress),
        ]
        if self._config is not None:
            parts.append(Align.center(_config_summary(self._config)))
        if not self.rows:
            parts.append(Align.center(_new_table(with_title=False)))
            return Panel(Group(*parts), border_style="cyan")

        # border, title, progress, config, blank spacer, safety margin
        fixed = 2 + 1 + 1 + (2 if self._config is not None else 0) + 2
        available = max(2 * CHART_MIN_HEIGHT, console.size.height - fixed)
        top_row_height = available // 2
        bottom_row_height = available - top_row_height

        table_capacity = max(1, top_row_height - 4)  # table box: borders, header, separator
        table = _new_table(with_title=False)
        for r in self._table_rows[-table_capacity:]:
            table.add_row(*r)

        loss_canvas = max(CHART_MIN_HEIGHT, top_row_height - 1)  # -1 for the chart's header line
        top_row = Table.grid(expand=True, padding=(0, 2), collapse_padding=True, pad_edge=False)
        top_row.add_column(ratio=1)
        top_row.add_column(ratio=1)
        top_row.add_row(
            table,
            _chart_block(
                "Loss",
                Text.assemble(("● train", "cyan"), "  ", ("● val", "green")),
                _PlotextChart(_draw_loss_chart, self.rows, loss_canvas),
            ),
        )
        parts.append(top_row)

        bottom_canvas = max(CHART_MIN_HEIGHT, bottom_row_height - 1)  # -1 for the row's header line
        lr_chart = _chart_block(
            "Optimization",
            Text.assemble(("● lr", "yellow"), "  ", ("● ||∇L||", "magenta")),
            _PlotextChart(_draw_lr_grad_chart, self.rows, bottom_canvas),
        )
        if not any(r["val_kpi_p50"] is not None for r in self.rows):
            parts.append(Group(Text(""), lr_chart))
        else:
            validation_chart = _chart_block(
                "Validation |R_GS|",
                Text.assemble(
                    ("● p95", "grey62"),
                    "  ",
                    ("● p50", "green"),
                    "  ",
                    ("● p05", "grey62"),
                ),
                _PlotextChart(_draw_validation_chart, self.rows, bottom_canvas),
            )
            bottom_row = Table.grid(
                expand=True, padding=(0, 2), collapse_padding=True, pad_edge=False
            )
            bottom_row.add_column(ratio=1)
            bottom_row.add_column(ratio=1)
            bottom_row.add_row(validation_chart, lr_chart)
            parts.append(Group(Text(""), bottom_row))
        return Panel(Group(*parts), border_style="cyan")

    def log(
        self,
        epoch: int,
        loss: float,
        residual: float,
        boundary: float,
        val_kpis: tuple[float, float, float] | None,
        lr: float,
        grad_norm: float,
        epoch_time: float,
    ) -> dict | None:
        self._acc_loss += loss
        self._acc_res += residual
        self._acc_bnd += boundary
        self._acc_gn += grad_norm
        self._acc_t += epoch_time
        self._acc_count += 1

        if (epoch + 1) % LOG_FREQUENCY == 0 or val_kpis is not None:
            p05, p50, p95 = val_kpis or (None, None, None)
            persisted = {
                "epoch": epoch + 1,
                "lr": lr,
                "loss": self._acc_loss / self._acc_count,
                "residual": self._acc_res / self._acc_count,
                "boundary": self._acc_bnd / self._acc_count,
                "grad_norm": self._acc_gn / self._acc_count,
                "epoch_time_seconds": self._acc_t / self._acc_count,
                "val_kpi_p05": p05,
                "val_kpi_p50": p50,
                "val_kpi_p95": p95,
            }
            self.rows.append(persisted)
            display_row = _metrics_row(
                epoch=epoch + 1,
                total_epochs=self._total_epochs,
                lr=lr,
                grad_norm=persisted["grad_norm"],
                loss=persisted["loss"],
                val_kpis=val_kpis,
                epoch_time=persisted["epoch_time_seconds"],
            )
            self._table_rows.append(display_row)
            if self.metrics_row_sink is not None:
                self.metrics_row_sink(display_row)
            self._acc_loss = self._acc_res = self._acc_bnd = self._acc_gn = self._acc_t = 0.0
            self._acc_count = 0
        else:
            persisted = None

        self._progress.update(self._epoch_task, advance=1)
        if self._live is not None:
            now = time.monotonic()
            if now - self._last_refresh_at >= 1.0 or epoch + 1 == self._total_epochs:
                self._live.update(self.renderable(), refresh=True)
                self._last_refresh_at = now
        return persisted


# ---------------------------------------------------------------------------
# _FileStorageManager — run-dir + artifact I/O
# ---------------------------------------------------------------------------


class _FileStorageManager:
    """Owns the run directory, artifact stem, and all file I/O for one training run."""

    def __init__(
        self,
        name: str,
        output_dir: Path | None,
        *,
        stage1_run_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.output_dir = output_dir
        self._stage1_run_dir = stage1_run_dir
        self.artifact_stem: str | None = None

    def new_artifact_stem(self) -> str:
        if self._stage1_run_dir is not None:
            return f"{self._stage1_run_dir.name}/stage2"
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.output_dir:
            return f"pinn_{timestamp}"
        return f"{timestamp}_{self.name}_{current_commit()}"

    def run_dir(self) -> Path:
        if self.artifact_stem is None:
            self.artifact_stem = self.new_artifact_stem()
        if self._stage1_run_dir is not None:
            return self._stage1_run_dir / "stage2"
        base = self.output_dir or Filepaths.BENCHMARKS
        return base / self.artifact_stem

    def discard_unsaved_run(self) -> None:
        """Delete the run dir unless a checkpoint was saved."""
        if self.artifact_stem is None:
            return
        run_dir = self.run_dir()
        if (run_dir / "network.flax").exists():
            return
        shutil.rmtree(run_dir, ignore_errors=True)
        if self.output_dir:
            with suppress(OSError):
                run_dir.parent.rmdir()

    def write_params(self, run_dir: Path, params: any) -> None:
        path = run_dir / "network.flax"
        temporary = path.with_suffix(".flax.tmp")
        temporary.write_bytes(flax.serialization.to_bytes(params))
        temporary.replace(path)

    def read_params(self, pinn_path: Path, target_params: any) -> any:
        with open(pinn_path, "rb") as f:
            return flax.serialization.from_bytes(target_params, f.read())

    def write_metrics(self, run_dir: Path, rows: list[dict]) -> None:
        fields = (
            "lr",
            "loss",
            "residual",
            "boundary",
            "grad_norm",
            "epoch_time_seconds",
            "val_kpi_p05",
            "val_kpi_p50",
            "val_kpi_p95",
        )
        metrics = {key: [row[key] for row in rows] for key in fields}
        write_json(
            run_dir / "metrics.json",
            {"format_version": 1, "logging_distance": LOG_FREQUENCY, **metrics},
        )

    def write_run(
        self,
        run_dir: Path,
        manager: "NetworkManager",
        duration: str | None,
        result: dict,
    ) -> None:
        write_json(
            run_dir / "run.json",
            {
                "format_version": 1,
                "commit": current_commit(),
                "duration": duration,
                "device": manager.device,
                "seed": manager.seed,
                "config": manager.config.to_dict(),
                "result": result,
            },
        )

    def benchmark(self, manager: "NetworkManager", test_mode: bool) -> dict:
        """Save the residual montage and return post-training KPIs."""
        if test_mode:
            return {}
        from src.engine.model_evaluation import (
            EVAL_RESOLUTION,
            N_PLOTS,
            build_kpi_record,
            evaluate_plasma_grids,
            evaluate_plasma_kpis,
            kpi_benchmark_configs,
            plot_plasma_grid_montage,
        )
        from src.lib.config import KPI_EVAL_CONFIGS, KPI_POINTS_PER_CONFIG

        run_dir = self.run_dir()
        configs = kpi_benchmark_configs(manager, KPI_EVAL_CONFIGS)
        kpis = evaluate_plasma_kpis(manager, configs, sample_size=KPI_POINTS_PER_CONFIG)
        grids = evaluate_plasma_grids(
            manager, configs[:N_PLOTS], resolution=EVAL_RESOLUTION, quantities=("residual",)
        )
        plot_plasma_grid_montage(
            grids,
            run_dir / "residual.png",
            quantity="residual",
            title=self.artifact_stem,
            metadata=manager.config.to_dict(),
            display_parameters=(
                "huber_delta",
                "learning_rate_max",
                "n_fourier_features",
                "lbfgs_steps",
            ),
            kpis=kpis,
        )
        record = build_kpi_record(manager, kpis, KPI_EVAL_CONFIGS, KPI_POINTS_PER_CONFIG, 0.85)
        logger.info(f"residual plot saved to {run_dir}")
        return kpi_values(record)

    def save_training_curves(self, run_dir: Path, rows: list[dict], artifact_stem: str) -> None:
        if not rows:
            return
        from src.engine.model_evaluation import plot_training_curves

        plot_training_curves(
            run_dir / "metrics.json", run_dir / "training_curves.png", title=artifact_stem
        )


# --- Manager for Training / Inference ---
class NetworkManager:
    """Facade composing a Trainer with Rich/Plotext display and filesystem I/O.

    Owns everything network.py's Trainer deliberately doesn't: the live
    training table/charts, run-dir/checkpoint/metrics.json persistence, and
    the CLI. Delegates model/optimizer/training-loop concerns to a Trainer
    instance so callers keep the same public surface (train, to_disk,
    from_disk, get_psi, sampler, config, state, ...) they had before the split.
    """

    # Optional consumer of finalized metrics-table rows (the HPO TUI appends them
    # to its sequential per-trial log). None = CLI mode, table rendering only.
    metrics_row_sink: Callable[[tuple[str, ...]], None] | None = None

    def __init__(
        self,
        config: HyperParams,
        seed: int = BASE_SEED,
        n_validation_size: int = N_VALIDATION_SIZE,
        test_mode: bool = False,
        output_dir: Path | None = None,
        name: str = "default",
        *,
        prior: FoundationModel | None = None,
        scale: float = 1.0,
        stage1_run_dir: Path | None = None,
    ) -> None:
        self.seed = seed
        self.test_mode = test_mode
        self.output_dir = output_dir
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(
                "Network name must contain only letters, numbers, underscores, or hyphens"
            )
        self.name = name
        self._trainer = Trainer(
            config, seed=seed, n_validation_size=n_validation_size, prior=prior, scale=scale
        )

        if stage1_run_dir is not None and prior is None:
            raise ValueError("stage1_run_dir is only valid for a corrector")
        self._prior = prior
        self._scale = scale
        self._files = _FileStorageManager(
            name=name,
            output_dir=output_dir,
            stage1_run_dir=stage1_run_dir,
        )

        self.training_log: list[dict] = []
        self.training_summary: dict | None = None
        self.artifact_stem: str | None = None
        self.device = "unknown"
        self.training_duration_seconds: float | None = None

    @classmethod
    def for_inference(
        cls,
        config: HyperParams,
        params: any,
        *,
        prior: FoundationModel | None = None,
        scale: float = 1.0,
        seed: int = BASE_SEED,
    ) -> "NetworkManager":
        """Construct a minimal manager for inference only (no training artifacts needed).

        Builds the full NetworkManager (sampler is needed by kpi_benchmark_configs /
        build_sample_response) but skips any disk I/O. Params are injected directly.
        """
        mgr = cls(config, seed=seed, prior=prior, scale=scale)
        mgr.state = mgr.state.replace(params=params)
        return mgr

    # ------------------------------------------------------------------
    # Delegation to the composed Trainer
    # ------------------------------------------------------------------

    @property
    def config(self) -> HyperParams:
        return self._trainer.config

    @property
    def sampler(self) -> Sampler:
        return self._trainer.sampler

    @property
    def model(self) -> FluxPINN:
        return self._trainer.model

    @property
    def state(self) -> train_state.TrainState:
        return self._trainer.state

    @state.setter
    def state(self, value: train_state.TrainState) -> None:
        self._trainer.state = value

    @property
    def epochs(self) -> int:
        return self._trainer.epochs

    def make_psi_fn(self) -> Callable[[any, any, any, PlasmaConfig], any]:
        """Factory returning the psi function for this field (single or composed)."""
        return self._trainer.make_psi_fn()

    def get_psi(self, R: any, Z: any, config: PlasmaConfig) -> any:
        """Evaluate magnetic flux psi at physical coordinates."""
        return self._trainer.get_psi(R, Z, config)

    def validation_configs(self) -> list:
        return self._trainer.validation_configs()

    def train_epoch(self, epoch: int) -> tuple[float, float, float, float]:
        return self._trainer.train_epoch(epoch)

    eval_step = staticmethod(Trainer.eval_step)

    # ------------------------------------------------------------------
    # Artifact stem / run-dir (delegates to _FileStorageManager)
    # ------------------------------------------------------------------

    def _new_artifact_stem(self) -> str:
        return self._files.new_artifact_stem()

    def run_dir(self) -> Path:
        stem = self._files.artifact_stem or self._files.new_artifact_stem()
        self._files.artifact_stem = stem
        self.artifact_stem = stem
        return self._files.run_dir()

    def discard_unsaved_run(self) -> None:
        """Delete the benchmark run dir unless this run's checkpoint was saved."""
        self._files.discard_unsaved_run()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_disk(self) -> str:
        """Save params, consolidated run data, metrics, and benchmark plots.

        Correctors configured with ``stage1_run_dir`` are nested under
        ``<stage1_run_dir>/stage2/`` and record their scale in ``run.json``.
        """
        if self._files.artifact_stem is None:
            self._files.artifact_stem = self._files.new_artifact_stem()
        self.artifact_stem = self._files.artifact_stem

        run_dir = self._files.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.device == "unknown":
            self.device = gpu_name()

        self._files.write_metrics(run_dir, self.training_log)
        kpis = self._files.benchmark(self, self.test_mode)
        result = {
            "status": "completed",
            **(self.training_summary or {}),
            "optimizer_updates": int(self.state.step),
            "examples_processed": int(self.state.step) * self.config.batch_size,
            "peak_memory_bytes": int(
                (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)
            ),
            "kpis": kpis,
        }
        if self._prior is not None:
            result["stage2_scale"] = self._scale
        self._files.write_run(
            run_dir,
            self,
            format_duration(self.training_duration_seconds)
            if self.training_duration_seconds is not None
            else None,
            result,
        )
        self._files.save_training_curves(run_dir, self.training_log, self.artifact_stem)
        self._files.write_params(run_dir, self.state.params)
        if self._trainer.lbfgs_params is not None:
            (run_dir / "network_lbfgs.flax").write_bytes(
                flax.serialization.to_bytes(self._trainer.lbfgs_params)
            )
        return self.artifact_stem

    def from_disk(self, pinn_path) -> any:  # noqa
        """Load Flax model parameters from disk."""
        return self._files.read_params(pinn_path, self.state.params)

    # ------------------------------------------------------------------
    # Training orchestration: Rich display + persistence around Trainer.train
    # ------------------------------------------------------------------

    def training_renderable(self) -> Panel:
        """Current metrics table + progress bar; rendered by Live or the HPO TUI."""
        return self._metrics.renderable()

    def train(
        self,
        save_to_disk: bool = True,
        validation_callback: Callable[[int, float | None], None] | None = None,
        show_progress: bool = True,
    ) -> float:
        "Orchestration layer for training - actual training happens in network.py"
        self._files.artifact_stem = self._files.new_artifact_stem()
        self.artifact_stem = self._files.artifact_stem

        # Only materialise a benchmark run dir when we intend to keep artifacts.
        # Correctors (save_to_disk=False) train in memory and persist later.
        run_dir = None
        if save_to_disk:
            run_dir = self._files.run_dir()
            run_dir.mkdir(parents=True, exist_ok=True)
            self.device = gpu_name()
            self._files.write_run(run_dir, self, None, {"status": "running"})

        training_started = time.perf_counter()
        try:
            self._metrics = _MetricsManager(total_epochs=self._trainer.epochs, config=self.config)
            self._metrics.metrics_row_sink = self.metrics_row_sink
            self._live = None
            if show_progress:
                # screen=True takes over the full terminal (alt-screen buffer) instead of
                # scrolling inline; _panel_chart_height sizes the charts to match so nothing
                # is cropped or left wasted at the bottom.
                live = Live(
                    self.training_renderable(),
                    auto_refresh=False,
                    console=console,
                    screen=True,
                )
            else:
                live = nullcontext()

            def on_epoch(event: EpochMetrics) -> None:
                val_kpis = (
                    (event.validation.p05, event.validation.p50, event.validation.p95)
                    if event.validation is not None
                    else None
                )
                persisted = self._metrics.log(
                    event.epoch,
                    event.loss,
                    event.residual_loss,
                    event.boundary_loss,
                    val_kpis,
                    event.learning_rate,
                    event.gradient_norm,
                    event.duration_seconds,
                )
                if run_dir is not None and (persisted is not None or val_kpis is not None):
                    self._files.write_metrics(run_dir, self._metrics.rows)
                if validation_callback is not None and not event.should_stop:
                    validation_callback(event.epoch + 1, val_kpis[1] if val_kpis else None)

            with live as active_live:
                if show_progress:
                    self._live = active_live
                    self._metrics._live = active_live
                result = self._trainer.train(observer=on_epoch)

            self.training_log = self._metrics.rows
            self.training_summary = {
                "stop_reason": result.stop_reason,
                "trained_epochs": result.trained_epochs,
                "planned_epochs": result.planned_epochs,
                "final_val_kpi_p05": result.final_validation.p05,
                "final_val_kpi_p50": result.final_validation.p50,
                "final_val_kpi_p95": result.final_validation.p95,
                **(
                    {
                        "lbfgs_steps_run": self.config.lbfgs_steps,
                        "lbfgs_val_kpi_p05": result.lbfgs_validation.p05,
                        "lbfgs_val_kpi_p50": result.lbfgs_validation.p50,
                        "lbfgs_val_kpi_p95": result.lbfgs_validation.p95,
                    }
                    if result.lbfgs_validation is not None
                    else {}
                ),
                "best_smoothed_val_kpi_p50": result.best_smoothed_validation_p50,
                "best_validation_epoch": result.best_epoch,
                "validation_frequency": VALIDATION_FREQUENCY,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "early_stopping_min_relative_improvement": (
                    EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT
                ),
                "early_stopping_rolling_window": EARLY_STOPPING_ROLLING_WINDOW,
            }
            self.training_duration_seconds = result.duration_seconds

            if save_to_disk:
                self.to_disk()
            return result.final_validation.p50
        except BaseException as exc:
            if run_dir is not None:
                self.training_duration_seconds = time.perf_counter() - training_started
                metrics = getattr(self, "_metrics", None)
                if metrics is not None and metrics.rows:
                    self._files.write_metrics(run_dir, metrics.rows)
                self._files.write_run(
                    run_dir,
                    self,
                    format_duration(self.training_duration_seconds),
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
            raise

    def lbfgs(self, steps: int, learning_rate: float | None = 0.1) -> None:
        """Polish AdamW-trained params with L-BFGS on one fixed batch."""
        self._trainer.lbfgs(steps, learning_rate=learning_rate)

    @staticmethod
    def _new_table() -> Table:
        """Training Metrics table skeleton; kept as a static method for HPO TUI compat."""
        return _new_table()


def _new_table(with_title: bool = True) -> Table:
    # The live panel and --show replay render their own centered heading; the HPO
    # TUI's sequential per-trial log still wants the title on the table itself.
    table = Table(
        title="Training Metrics" if with_title else None,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Epoch", justify="right", style="cyan")
    table.add_column("LR", justify="right", style="yellow")
    table.add_column("||∇L||", justify="right", style="magenta")
    table.add_column("Loss", justify="right", style="magenta")
    table.add_column("Val p05", justify="right", style="green")
    table.add_column("Val p50", justify="right", style="green")
    table.add_column("Val p95", justify="right", style="green")
    table.add_column("Time/Ep", justify="right")
    return table


def _metrics_row(
    epoch: int,
    total_epochs: int,
    lr: float,
    grad_norm: float,
    loss: float,
    val_kpis: tuple[float, float, float] | None,
    epoch_time: float,
) -> tuple[str, ...]:
    """One Training Metrics table row shared by live display and replay."""
    p05, p50, p95 = val_kpis or (None, None, None)
    return (
        f"{epoch}/{total_epochs}",
        f"{lr:.2e}",
        f"{grad_norm:.2e}",
        f"{loss:.2e}",
        f"{p05:.2e}" if p05 is not None else "-",
        f"{p50:.2e}" if p50 is not None else "-",
        f"{p95:.2e}" if p95 is not None else "-",
        f"{epoch_time:.2f}s",
    )


def show_run(run: str) -> None:
    """Re-render the Training Metrics table for a stored run from metrics.json."""
    run_dir = Path(run)
    if not run_dir.is_dir():
        candidates = [
            Filepaths.BENCHMARKS / run,
            *Filepaths.BENCHMARKS.glob(f"*/{run}"),
            *(Filepaths.DATA / "hpo").glob(f"*/{run}"),
        ]
        run_dir = next((p for p in candidates if p.is_dir()), run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"no metrics.json found for run '{run}'")

    metrics = json.loads(metrics_path.read_text())
    distance = int(metrics["logging_distance"])
    run_record = load_run(run_dir)
    total_epochs = int(run_record.get("result", {}).get("trained_epochs", 0))
    if not total_epochs:
        total_epochs = len(metrics["lr"]) * distance
    rows = [
        {
            "epoch": (index + 1) * distance,
            "lr": lr,
            "loss": metrics["loss"][index],
            "grad_norm": metrics["grad_norm"][index],
            "val_kpi_p05": metrics["val_kpi_p05"][index],
            "val_kpi_p50": metrics["val_kpi_p50"][index],
            "val_kpi_p95": metrics["val_kpi_p95"][index],
        }
        for index, lr in enumerate(metrics["lr"])
    ]
    table = _new_table(with_title=False)
    for index, row in enumerate(rows):
        table.add_row(
            *_metrics_row(
                epoch=row["epoch"],
                total_epochs=total_epochs,
                lr=row["lr"],
                grad_norm=row["grad_norm"],
                loss=metrics["loss"][index],
                val_kpis=(row["val_kpi_p05"], row["val_kpi_p50"], row["val_kpi_p95"])
                if row["val_kpi_p50"] is not None
                else None,
                epoch_time=metrics["epoch_time_seconds"][index],
            )
        )
    parts: list[any] = [
        Align.center(Text("Training Metrics", style="italic")),
        Align.center(table),
    ]
    if (charts := _charts_renderable(rows, CHART_HEIGHT)) is not None:
        parts.append(charts)
    console.print(Panel(Group(*parts), border_style="cyan"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PINN network")
    parser.add_argument(
        "--test", action="store_true", help="Run with minimal parameters for rapid iteration"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate_max")
    parser.add_argument(
        "--name", default="default", help="Artifact name in <timestamp>_<name>_<commit>"
    )
    parser.add_argument(
        "--fourier-features",
        type=int,
        default=64,
        help="Random Fourier features on (r,z); 0 = off (default 64, per grid-2 ablation)",
    )
    parser.add_argument(
        "--lbfgs", type=int, default=0, help="L-BFGS polish steps after AdamW; 0 = off"
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="PDE loss: >0 = Huber with this delta (default 1.0), 0.0 = MSE",
    )
    parser.add_argument(
        "--weight-flux-scale",
        type=float,
        default=None,
        help="Weight of the interior-mean-ψ collapse-guard hinge (default 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total epochs; split 1:5 warmup:decay (default: HyperParams 100+500)",
    )
    parser.add_argument(
        "--soft-bc",
        action="store_true",
        help="Legacy soft-BC training: raw ψ + Dirichlet/Neumann penalties (no envelope)",
    )
    parser.add_argument(
        "--rwf",
        action="store_true",
        help="Random Weight Factorization (Wang et al. arXiv 2210.01274): reparametrize "
        "each dense kernel as W = V * exp(s) to improve PINN accuracy",
    )
    parser.add_argument(
        "--arch",
        choices=["mlp", "piratenet"],
        default="mlp",
        help="Network architecture: 'mlp' = plain MLP (default), 'piratenet' = PirateNet "
        "residual blocks (arXiv 2402.00326)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default=None,
        help="Comma-separated hidden layer widths, e.g. 128,128,128,128; for piratenet each "
        "entry is one residual block of that width (default: HyperParams default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Plasma configs per train step (default: HyperParams 64); smaller batches "
        "halve activation memory, unlocking wider nets on 12GB",
    )
    parser.add_argument(
        "--n-train", type=int, default=None, help="Training configurations sampled per epoch"
    )
    parser.add_argument(
        "--inner-samples", type=int, default=None, help="Interior collocation points per config"
    )
    parser.add_argument(
        "--boundary-samples", type=int, default=None, help="Boundary points per config"
    )
    parser.add_argument(
        "--show",
        metavar="RUN",
        default=None,
        help="Render the stored Training Metrics table for a run "
        "(dir path, artifact slug, or pinn_<timestamp>) and exit",
    )
    args = parser.parse_args()

    if args.show:
        show_run(args.show)
        raise SystemExit

    manager = None
    try:
        if not args.test:
            config = HyperParams(
                huber_delta=args.huber_delta,
                n_fourier_features=args.fourier_features,
                lbfgs_steps=args.lbfgs,
                soft_bc=args.soft_bc,
                rwf=args.rwf,
                arch=args.arch,
            )
            if args.lr is not None:
                config = config.replace(learning_rate_max=args.lr)
            if args.weight_flux_scale is not None:
                config = config.replace(weight_flux_scale=args.weight_flux_scale)
            if args.epochs is not None:
                config = config.replace(
                    warmup_epochs=max(1, args.epochs // 6),
                    decay_epochs=args.epochs - max(1, args.epochs // 6),
                )
            if args.hidden_dims is not None:
                config = config.replace(
                    hidden_dims=tuple(int(d) for d in args.hidden_dims.split(","))
                )
            if args.batch_size is not None:
                config = config.replace(batch_size=args.batch_size)
            if args.n_train is not None:
                config = config.replace(n_train=args.n_train)
            if args.inner_samples is not None:
                config = config.replace(n_rz_inner_samples=args.inner_samples)
            if args.boundary_samples is not None:
                config = config.replace(n_rz_boundary_samples=args.boundary_samples)
            if (
                min(
                    config.batch_size,
                    config.n_train,
                    config.n_rz_inner_samples,
                    config.n_rz_boundary_samples,
                )
                <= 0
            ):
                parser.error("training and sample budgets must be positive")
            if config.n_train % config.batch_size:
                parser.error("--n-train must be divisible by --batch-size")
            manager = NetworkManager(config, test_mode=args.test, name=args.name)
            manager.train(save_to_disk=True)
        else:
            globals()["N_VALIDATION_SIZE"] = 16
            import src.engine.network as _network_module

            _network_module.VALIDATION_FREQUENCY = 20
            config = HyperParams(
                huber_delta=args.huber_delta,
                soft_bc=args.soft_bc,
                lbfgs_steps=64,
                hidden_dims=(32, 32),
                batch_size=8,
                n_rz_inner_samples=64,
                n_rz_boundary_samples=16,
                n_train=64,
                warmup_epochs=20,
                decay_epochs=20,
            )
            manager = NetworkManager(config, test_mode=args.test)
            manager.train(save_to_disk=False)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Execution failed with an error: {e}", exc_info=True)
        raise
    finally:
        if manager is not None:
            manager.discard_unsaved_run()
