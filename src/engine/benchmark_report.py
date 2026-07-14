"""LaTeX benchmark report generator for PINN HPO runs.

Builds an in-memory LaTeX document from a set of trained-network run
directories (each holding config.json, kpis.json, residual.png), renders it
to PDF via a direct ``pdflatex`` call, and writes only the resulting PDF.

Per config (sorted ascending by Optuna objective): a compact settings grid,
KPI table, and 4x2 residual montage, each on its own page. A ranked summary
table with hyperlinks to each section opens the report.

System prereqs: a LaTeX engine (pdflatex by default). pandoc is NOT used --
it strips raw ``\\clearpage`` commands, so we call pdflatex directly on the
complete LaTeX document we already build in memory.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

# Edit this list to render a manual report via `uv run python -m src.engine.benchmark_report`.
NETWORK_PATHS: list[Path] = []

# KPI keys shown in the per-config table, in display order.
_KPI_FIELDS: tuple[str, ...] = (
    "loss_median",
    "loss_mean",
    "loss_p95",
    "loss_p05",
    "core_loss_median",
    "core_loss_mean",
    "core_loss_p95",
    "core_loss_p05",
    "edge_loss_p95",
    "boundary_leak_max",
)

# Summary-table columns (subset of _KPI_FIELDS).
_SUMMARY_KPI_FIELDS: tuple[str, ...] = (
    "loss_median",
    "core_loss_median",
    "edge_loss_p95",
    "boundary_leak_max",
)


def generate_report(
    paths: list[Path] = NETWORK_PATHS,
    *,
    output_path: Path | None = None,
    pdf_engine: str = "pdflatex",
) -> None:
    """Render a ranked PDF report from ``paths`` (one run dir each).

    The LaTeX source is written to a temp .tex in the output dir, rendered via
    ``pdf_engine`` (run twice for hyperref cross-refs), then only the PDF is
    kept. Returns None.
    """
    if not paths:
        raise ValueError("No run paths provided -- set NETWORK_PATHS or pass paths explicitly.")
    paths = [Path(p) for p in paths]
    output_path = (
        Path(output_path) if output_path is not None else _default_output_path(paths)
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = _collect(paths)
    if not summaries:
        raise RuntimeError("No valid run directories found (none had config.json).")
    summaries.sort(key=lambda s: s.objective)

    latex = _build_document(summaries)
    _render_pdf(latex, output_path, pdf_engine)
    logger.info(f"Benchmark report written to {output_path}")


@dataclass
class _RunSummary:
    """One run dir's loaded artifacts."""

    run_dir: Path
    stem: str
    config: dict
    kpis: dict
    objective: float


def _collect(paths: list[Path]) -> list[_RunSummary]:
    objectives = _objectives_from_ledger(paths)
    summaries: list[_RunSummary] = []
    for run_dir in paths:
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            logger.warning(f"Skipping {run_dir.name}: missing config.json")
            continue
        config = json.loads(cfg_path.read_text())
        kpis = _load_json(run_dir / "kpis.json")
        stem = run_dir.name
        objective = objectives.get(stem, float(kpis.get("loss_median", float("inf"))))
        summaries.append(
            _RunSummary(
                run_dir=run_dir,
                stem=stem,
                config=config,
                kpis=kpis,
                objective=objective,
            )
        )
    return summaries


def _objectives_from_ledger(paths: list[Path]) -> dict[str, float]:
    """Read Optuna objectives from a trials.json ledger in the shared parent.

    A COMPLETE trial's `value` is the validation-loss median used for ranking.
    Falls back to {} if no ledger is found (manual benchmark paths).
    """
    parents = {p.parent for p in paths}
    if len(parents) != 1:
        return {}
    ledger_path = next(iter(parents)) / "trials.json"
    if not ledger_path.exists():
        return {}
    try:
        ledger = json.loads(ledger_path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Could not parse {ledger_path}")
        return {}
    return {
        t["run"]: float(t["value"])
        for t in ledger
        if t.get("state") == "COMPLETE" and t.get("value") is not None and "run" in t
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Could not parse {path}")
        return {}


def _default_output_path(paths: list[Path]) -> Path:
    parents = {p.parent for p in paths}
    if len(parents) == 1:
        return next(iter(parents)) / "benchmark_report.pdf"
    return Path("benchmark_report.pdf")


def _build_document(summaries: list[_RunSummary]) -> str:
    parts: list[str] = [_preamble()]
    parts.append(_title_block(summaries))
    parts.append(_summary_table(summaries))
    parts.append(r"\clearpage")
    for s in summaries:
        parts.append(_config_section(s))
        parts.append(r"\clearpage")
    parts.append(r"\end{document}")
    return "\n".join(parts)


def _preamble() -> str:
    # We call pdflatex directly (not via pandoc), so all packages must be
    # loaded here. Margins are set via the geometry package.
    return r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
"""


def _title_block(summaries: list[_RunSummary]) -> str:
    study_name = ""
    parent = summaries[0].run_dir.parent
    top_path = parent / "top_trials.json"
    if top_path.exists():
        meta = _load_json(top_path)
        study_name = meta.get("study_name", "")
    if not study_name:
        study_name = parent.name
    n = len(summaries)
    date = datetime.now().strftime("%Y-%m-%d")
    best = summaries[0].objective if summaries else float("nan")
    return (
        r"\title{PINN Benchmark Report \\ \large "
        + _escape(study_name)
        + "}\n"
        + r"\author{Hyperparameter Optimization}"
        + "\n"
        + r"\date{"
        + date
        + "}\n"
        + r"\begin{document}"
        + "\n"
        + r"\maketitle"
        + "\n\n"
        + f"{n} completed configuration(s). Best objective: ${best:.6e}$.\n\n"
    )


def _summary_table(summaries: list[_RunSummary]) -> str:
    header = (
        "Rank & Config & Objective & "
        + " & ".join(_escape(k) for k in _SUMMARY_KPI_FIELDS)
        + r" \\"
    )
    lines = [
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{llr" + "r" * len(_SUMMARY_KPI_FIELDS) + r"}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for rank, s in enumerate(summaries, 1):
        kpi_vals = [_fmt_kpi(s.kpis.get(k)) for k in _SUMMARY_KPI_FIELDS]
        link = rf"\hyperref[sec:{s.stem}]{{{_escape(s.stem)}}}"
        row = f"{rank} & {link} & ${s.objective:.6e}$ & " + " & ".join(kpi_vals) + r" \\"
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\par\smallskip\textit{Configurations ranked by Optuna objective "
        r"(validation median $|R_{GS}|$).}\par"
    )
    lines.append(r"\end{center}")
    return "\n".join(lines) + "\n"


def _config_section(s: _RunSummary) -> str:
    lines: list[str] = []
    lines.append(rf"\section{{{_escape(s.stem)}}}")
    lines.append(rf"\label{{sec:{s.stem}}}")
    lines.append("")
    lines.append(_one_line_summary(s))
    lines.append("")
    lines.append(_settings_table(s.config))
    lines.append("")
    lines.append(_kpi_table(s.kpis))
    lines.append("")
    lines.append(_figure(s, "residual.png", "Residual montage (4 cols x 2 rows)."))
    return "\n".join(lines) + "\n"


def _one_line_summary(s: _RunSummary) -> str:
    dims = s.config.get("hidden_dims", [])
    arch = f"{len(dims)}x{dims[0]}" if dims else "?"
    bc = "soft" if s.config.get("soft_bc", False) else "hard"
    lr_max = s.config.get("learning_rate_max", float("nan"))
    lr_min = s.config.get("learning_rate_min", float("nan"))
    warmup = s.config.get("warmup_epochs", 0)
    decay = s.config.get("decay_epochs", 0)
    epochs = warmup + decay
    return (
        rf"Architecture ${arch}$, {bc}-BC, "
        rf"LR ${lr_max:.2e} \rightarrow {lr_min:.2e}$, {epochs} epochs, "
        rf"objective ${s.objective:.6e}$."
    )


def _settings_table(config: dict) -> str:
    dims = config.get("hidden_dims", [])
    settings = (
        ("Architecture", f"{len(dims)}x{dims[0]}" if dims else "?"),
        ("Boundary condition", "Soft" if config.get("soft_bc", False) else "Hard"),
        ("Epochs", _fmt_int(config.get("warmup_epochs", 0) + config.get("decay_epochs", 0))),
        (
            "Learning rate",
            f"{_fmt_value(config.get('learning_rate_max'))} to "
            f"{_fmt_value(config.get('learning_rate_min'))}",
        ),
        ("PDE loss", _loss_label(config.get("huber_delta"))),
        ("Fourier features", _fourier_label(config)),
        ("L-BFGS steps", _fmt_int(config.get("lbfgs_steps"))),
        ("Weight decay", _fmt_value(config.get("weight_decay"))),
        ("Batch size", _fmt_int(config.get("batch_size"))),
        (
            "Sampling",
            f"{_fmt_int(config.get('n_train'))} train, "
            f"{_fmt_int(config.get('n_rz_inner_samples'))}/"
            f"{_fmt_int(config.get('n_rz_boundary_samples'))} points",
        ),
        ("Adaptive sampling", _fmt_value(config.get("sigma_residual_adaptive_sampling"))),
        (
            "Loss weights",
            f"BC {_fmt_value(config.get('weight_boundary_condition'))}, "
            f"flux {_fmt_value(config.get('weight_flux_scale'))}",
        ),
    )
    lines = [
        r"\subsection*{Training settings}",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{@{}ll@{\qquad}ll@{}}",
    ]
    for left, right in zip(settings[::2], settings[1::2], strict=True):
        lines.append(
            rf"\textbf{{{_escape(left[0])}}} & {_escape(left[1])} & "
            rf"\textbf{{{_escape(right[0])}}} & {_escape(right[1])} \\\\"
        )
    lines.extend((r"\end{tabular}", r"\end{center}"))
    return "\n".join(lines)


def _kpi_table(kpis: dict) -> str:
    if not kpis:
        return r"\subsection*{KPIs}\textit{(kpis.json missing)}"
    lines = [
        r"\subsection*{KPIs}",
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{@{}lr@{\qquad}lr@{}}",
        r"\toprule",
        r"KPI & value & KPI & value \\",
        r"\midrule",
    ]
    for left, right in zip(_KPI_FIELDS[::2], _KPI_FIELDS[1::2], strict=True):
        lines.append(
            rf"{_escape(left)} & {_fmt_kpi(kpis.get(left))} & "
            rf"{_escape(right)} & {_fmt_kpi(kpis.get(right))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    n_cfg = kpis.get("n_configs", "?")
    n_pts = kpis.get("n_points", "?")
    core = kpis.get("core_rho", "?")
    loss = _escape(str(kpis.get("loss", "?")))
    lines.append(r"\par\smallskip")
    lines.append(
        rf"\textit{{Evaluated on {n_cfg} configs, {n_pts} points, "
        rf"core $\rho<{core}$, loss={loss}.}}\par"
    )
    lines.append(r"\end{center}")
    return "\n".join(lines)


def _figure(s: _RunSummary, filename: str, caption: str) -> str:
    path = (s.run_dir / filename).resolve()
    if not path.exists():
        return rf"\subsection*{{{filename}}}\textit{{(missing)}}"
    return (
        r"\begin{center}"
        "\n"
        rf"\includegraphics[width=0.9\textwidth]{{{path}}}"
        "\n"
        rf"\par\smallskip\textit{{{caption}}}\par"
        "\n"
        r"\end{center}"
    )


def _fmt_kpi(v: float | None) -> str:
    if v is None:
        return r"--"
    try:
        return f"${float(v):.6e}$"
    except (TypeError, ValueError):
        return _escape(str(v))


def _fmt_value(value: object) -> str:
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return "--"


def _fmt_int(value: object) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "--"


def _loss_label(huber_delta: object) -> str:
    try:
        delta = float(huber_delta)
    except (TypeError, ValueError):
        return "--"
    return "MSE" if delta == 0 else f"Huber ({delta:.2g})"


def _fourier_label(config: dict) -> str:
    count = _fmt_int(config.get("n_fourier_features"))
    if count == "0":
        return "Off"
    return f"{count} (sigma {_fmt_value(config.get('fourier_sigma'))})"


def _escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("^", r"\textasciicircum{}")
        .replace("~", r"\textasciitilde{}")
    )


def _render_pdf(latex: str, output_path: Path, pdf_engine: str) -> None:
    # Write the .tex to the output dir, run the LaTeX engine twice (hyperref
    # cross-refs settle on the second pass), then keep only the PDF.
    tex_path = output_path.with_suffix(".tex")
    tex_path.write_text(latex)
    stem = tex_path.stem
    for _ in range(2):
        proc = subprocess.run(
            [
                pdf_engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={output_path.parent}",
                str(tex_path),
            ],
            cwd=str(output_path.parent),
            capture_output=True,
        )
        if proc.returncode != 0:
            log_path = output_path.with_suffix(".log")
            log_tail = log_path.read_text(errors="replace")[-2000:] if log_path.exists() else ""
            raise RuntimeError(
                f"{pdf_engine} failed (exit {proc.returncode}):\nlog tail:\n{log_tail}"
            )
    produced = output_path.parent / f"{stem}.pdf"
    if produced != output_path:
        produced.replace(output_path)
    for ext in (".aux", ".log", ".out", ".tex"):
        (output_path.parent / f"{stem}{ext}").unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_report()
