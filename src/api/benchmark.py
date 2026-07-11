"""Multi-checkpoint benchmark comparison, streamed row-by-row over SSE."""

from collections.abc import Iterator
import json

from src.api.network import build_kpis, build_plasma_grids
from src.api.state import get_manager
from src.lib.config import Filepaths
from src.streamlit.network_utils import filter_networks_by_commit


def run_benchmark(
    networks: list[str],
    commit: str | None,
    mode: str,
    seed: int,
    sample_size: int,
    resolution: int,
    kpi_sample_size: int,
) -> Iterator[str]:
    filtered = filter_networks_by_commit(networks, commit)
    if not filtered:
        yield _sse_event({"type": "error", "message": "No networks found for the selected commit."})
        return

    for name in reversed(filtered):
        config_path = (Filepaths.NETWORKS / name).with_suffix(".json")
        if not config_path.exists():
            yield _sse_event({"type": "row_error", "network": name, "message": "Missing config"})
            continue

        manager = get_manager(name)
        row: dict = {"type": "row", "network": name, "config": json.loads(config_path.read_text())}

        quantities = {
            "Flux Prediction": ("flux",),
            "GS Residual": ("residual",),
            "Both": ("flux", "residual"),
        }[mode]
        grids = build_plasma_grids(manager, seed, sample_size, resolution, quantities)
        row["kpis"] = build_kpis(manager, seed, sample_size, kpi_sample_size)
        if "flux" in grids:
            row["flux_grids"] = grids["flux"]
        if "residual" in grids:
            row["residual_grids"] = grids["residual"]

        yield _sse_event(row)

    yield _sse_event({"type": "done"})


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
