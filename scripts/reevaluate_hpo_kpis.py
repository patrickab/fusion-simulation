"""Re-score trained networks in an HPO study bundle under the current global protocol.

Values are RE-SCORED using KPI_POINTS_PER_CONFIG x KPI_EVAL_CONFIGS
(docs/evaluation/kpi-accuracy-benchmark.md) by default.  To reproduce a study's
original setting (e.g. old studies used n_validate=20) pass --n-validate and
--score-beta explicitly.  Flattened single-config benchmark dirs are re-evaluated
with ``uv run python -m src.engine.model_evaluation <slugs>`` instead — this
script handles the study.db / trials.json side of HPO bundles only.

Usage:
    uv run python scripts/reevaluate_hpo_kpis.py <run_dir>
        [--n-validate N] [--score-beta 0.3]
        [--n-points N] [--kpi-n-configs N] [--kpi-n-points N]
        [--core-rho 0.85] [--no-plots] [--plot-resolution 600]
        [--dry-run] [--study-name NAME]

<run_dir> is the HPO study bundle, e.g.
    data/hpo/2026_07_14_10_05_06_pinn_hpo_n6_lr_sigma_2400ep_d46257d

--n-validate and --score-beta MUST match the StudyConfig the run was produced
with; they are not recoverable from the run dir itself.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import optuna
from optuna.storages import RDBStorage
from rich.console import Console
from rich.table import Table
import sqlalchemy

from src.engine.model_evaluation import (
    EVAL_RESOLUTION,
    N_PLOTS,
    build_kpi_record,
    evaluate_plasma_grids,
    evaluate_plasma_kpis,
    evaluate_validation_loss_stats,
    kpi_benchmark_configs,
    plot_plasma_grid_montage,
)
from src.engine.network import NetworkManager
from src.engine.optimize_network_optuna import _write_trials_json
from src.lib.config import KPI_EVAL_CONFIGS, KPI_POINTS_PER_CONFIG
from src.lib.network_config import HyperParams

console = Console()


def _discover_study_name(storage: str) -> str | None:
    """The non-legacy study with the most trials in this study.db."""
    summaries = optuna.get_all_study_summaries(storage)
    eligible = [s for s in summaries if s.study_name != "warmstart_experiments" and s.n_trials > 0]
    if not eligible:
        return None
    return max(eligible, key=lambda s: s.n_trials).study_name


def _lock_is_live(run_dir: Path) -> bool:
    """True if a .lock in run_dir holds a live PID (HPO may be writing)."""
    lock = run_dir / ".lock"
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text().strip())
    except (ValueError, OSError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _load_manager(checkpoint_dir: Path, n_validate: int) -> NetworkManager:
    hp = HyperParams.from_json(str(checkpoint_dir / "config.json"))
    manager = NetworkManager(hp, n_validation_size=n_validate)
    loaded = manager.from_disk(pinn_path=str(checkpoint_dir / "network.flax"))
    manager.state = manager.state.replace(params=loaded)
    return manager


def _network_name(checkpoint_dir: Path) -> str:
    """Preserve the existing kpis.json `network` field if present."""
    kpi_path = checkpoint_dir / "kpis.json"
    if kpi_path.exists():
        try:
            return json.loads(kpi_path.read_text()).get("network", checkpoint_dir.name)
        except (ValueError, OSError):
            pass
    return checkpoint_dir.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate trained HPO networks and update study.db trial values."
    )
    parser.add_argument("run_dir", type=Path, help="HPO study bundle dir (contains study.db)")
    parser.add_argument(
        "--n-validate",
        type=int,
        default=KPI_EVAL_CONFIGS,
        help="Study's n_validate (must match original run).",
    )
    parser.add_argument(
        "--score-beta", type=float, default=0.3, help="Study's score_beta (must match)."
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=KPI_POINTS_PER_CONFIG,
        help="Sobol sample size for the fused validation score.",
    )
    parser.add_argument(
        "--kpi-n-configs",
        type=int,
        default=KPI_EVAL_CONFIGS,
        help="Config count for the kpis.json benchmark suite.",
    )
    parser.add_argument(
        "--kpi-n-points",
        type=int,
        default=KPI_POINTS_PER_CONFIG,
        help="Sample size for the kpis.json benchmark suite.",
    )
    parser.add_argument("--core-rho", type=float, default=0.85)
    parser.add_argument("--no-plots", action="store_true", help="Skip residual.png regeneration.")
    parser.add_argument("--plot-resolution", type=int, default=EVAL_RESOLUTION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and print a diff table; write nothing.",
    )
    parser.add_argument("--study-name", default=None, help="Override auto-discovered study name.")
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    db_path = run_dir / "study.db"
    if not db_path.exists():
        console.print(f"[red]error:[/red] {db_path} not found")
        return 2

    if not args.dry_run and _lock_is_live(run_dir):
        console.print(f"[red]error:[/red] {run_dir} is locked by a live HPO process (.lock)")
        return 3

    storage = f"sqlite:///{db_path}"
    study_name = args.study_name or _discover_study_name(storage)
    if study_name is None:
        console.print(f"[red]error:[/red] no eligible study found in {db_path}")
        return 4
    console.print(f"study: [cyan]{study_name}[/cyan]  ({db_path})")

    study = optuna.load_study(study_name=study_name, storage=storage)
    rdb = RDBStorage(storage)
    study_id = rdb.get_study_id_from_name(study_name)

    table = Table(title="Re-evaluated trial values", show_lines=True)
    table.add_column("trial", justify="right", style="cyan")
    table.add_column("old value", style="yellow")
    table.add_column("new fused", style="green")
    table.add_column("median")
    table.add_column("p95")
    table.add_column("kpis.json", style="magenta")
    table.add_column("status")

    n_updated = n_skipped = 0
    update_sql = sqlalchemy.text(
        "UPDATE trial_values SET value = :v "
        "WHERE objective = 0 "
        "AND trial_id = (SELECT trial_id FROM trials WHERE study_id = :sid AND number = :num)"
    )

    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        stem = t.user_attrs.get("run")
        if stem is None:
            table.add_row(str(t.number), f"{t.value:.6e}", "-", "-", "-", "-", "no run attr")
            n_skipped += 1
            continue
        checkpoint_dir = run_dir / stem
        if not (checkpoint_dir / "network.flax").exists():
            table.add_row(
                str(t.number),
                f"{t.value:.6e}",
                "-",
                "-",
                "-",
                "-",
                f"checkpoint missing ({stem})",
            )
            n_skipped += 1
            continue

        manager = _load_manager(checkpoint_dir, args.n_validate)
        median, p95 = evaluate_validation_loss_stats(manager, sample_size=args.n_points)
        fused = median + args.score_beta * p95

        # Benchmark KPI suite for kpis.json (shared stream via kpi_benchmark_configs).
        configs = kpi_benchmark_configs(manager, args.kpi_n_configs)
        kpis = evaluate_plasma_kpis(
            manager, configs, sample_size=args.kpi_n_points, core_rho=args.core_rho
        )

        kpi_status = "skipped"
        if not args.dry_run:
            record = build_kpi_record(
                manager,
                kpis,
                args.kpi_n_configs,
                args.kpi_n_points,
                args.core_rho,
                network_name=_network_name(checkpoint_dir),
            )
            (checkpoint_dir / "kpis.json").write_text(json.dumps(record, indent=2) + "\n")
            kpi_status = "rewritten"

            if not args.no_plots:
                grids = evaluate_plasma_grids(
                    manager,
                    configs[:N_PLOTS],
                    resolution=args.plot_resolution,
                    quantities=("residual",),
                )
                plot_plasma_grid_montage(
                    grids,
                    checkpoint_dir / "residual.png",
                    quantity="residual",
                    title=stem,
                    metadata=manager.config.to_dict(),
                    display_parameters=(
                        "huber_delta",
                        "learning_rate_max",
                        "n_fourier_features",
                        "lbfgs_steps",
                    ),
                    kpis=kpis,
                )

            with rdb.engine.begin() as conn:
                conn.execute(update_sql, {"v": fused, "sid": study_id, "num": t.number})
            n_updated += 1
        else:
            n_updated += 1

        table.add_row(
            str(t.number),
            f"{t.value:.6e}",
            f"{fused:.6e}",
            f"{median:.3e}",
            f"{p95:.3e}",
            kpi_status,
            "updated" if not args.dry_run else "dry-run",
        )

    console.print(table)
    console.print(
        f"\n[green]updated[/green] {n_updated}   [yellow]skipped[/yellow] {n_skipped}"
        + ("   [dim](dry-run: nothing written)[/dim]" if args.dry_run else "")
    )

    if not args.dry_run and n_updated:
        # Reload so the trial cache reflects the SQL UPDATEs just performed;
        # the in-memory `study` was loaded before any write and holds stale values.
        fresh = optuna.load_study(study_name=study_name, storage=storage)
        _write_trials_json(fresh, run_dir)
        console.print("trials.json refreshed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
