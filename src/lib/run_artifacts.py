import json
import math
from numbers import Integral, Real
from pathlib import Path
import subprocess
from typing import Any

KPI_FIELDS = (
    "n_configs",
    "n_points",
    "core_rho",
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


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}h-{minutes:02d}min-{seconds:02d}sec"


def gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return next((line.strip() for line in result.stdout.splitlines() if line.strip()), "unknown")


def write_json(path: Path, value: object) -> None:
    """Atomically write valid JSON with five-significant-digit scientific floats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_encode_json(value) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def _encode_json(value: object, level: int = 0) -> str:
    indent = "  " * level
    child_indent = "  " * (level + 1)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON does not support non-finite floats")
        return format(number, ".4e")
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        entries = [
            f"{child_indent}{json.dumps(str(key))}: {_encode_json(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + f"\n{indent}}}"
    if isinstance(value, list | tuple):
        if not value:
            return "[]"
        entries = [f"{child_indent}{_encode_json(item, level + 1)}" for item in value]
        return "[\n" + ",\n".join(entries) + f"\n{indent}]"
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def load_run(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def load_config(run_dir: Path) -> dict[str, Any]:
    return load_run(run_dir)["config"]


def load_kpis(run_dir: Path) -> dict[str, float]:
    return load_run(run_dir).get("result", {}).get("kpis", {})


def kpi_values(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in KPI_FIELDS if key in record}


def update_run_result(run_dir: Path, **updates: object) -> None:
    run = load_run(run_dir)
    run["result"] = {**run.get("result", {}), **updates}
    write_json(run_dir / "run.json", run)
