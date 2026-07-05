"""Multi-checkpoint benchmark comparison, streamed row-by-row over SSE."""

from collections.abc import Iterator
import json

from src.api.network import build_flux_grids, build_residual_grids
from src.api.state import get_manager
from src.lib.config import Filepaths
from src.streamlit.network_utils import filter_networks_by_commit


def run_benchmark(
    networks: list[str], commit: str | None, mode: str, seed: int, sample_size: int, resolution: int
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

        if mode in ("Flux Prediction", "Both"):
            row["flux_grids"] = build_flux_grids(manager, seed, sample_size, resolution)
        if mode in ("GS Residual", "Both"):
            row["residual_grids"] = build_residual_grids(manager, seed, sample_size, resolution)

        yield _sse_event(row)

    yield _sse_event({"type": "done"})


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
