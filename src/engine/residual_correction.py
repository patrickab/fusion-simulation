"""Neural residual-correction training and checkpoint loading."""

import argparse
from pathlib import Path

from src.engine.network import FoundationModel
from src.engine.network_manager import NetworkManager
from src.lib.config import KPI_EVAL_CONFIGS, NEURAL_CORRECTOR_DIR
from src.lib.logger import get_logger
from src.lib.network_config import Architecture, HyperParams
from src.lib.run_artifacts import load_config, load_run
from src.streamlit.network_utils import resolve_run_directory

logger = get_logger(name="ResidualCorrection")


def corrector_foundation_dir(corrector_dir: Path) -> Path | None:
    """Return the foundation that owns a nested neural corrector, if any."""
    for path in (corrector_dir, *corrector_dir.parents):
        if path.name == NEURAL_CORRECTOR_DIR:
            return path.parent
    return None


def load_foundation(
    foundation: str | Path, *, soft_bc: bool | None = None
) -> tuple[FoundationModel, HyperParams, Path]:
    """Load a plain checkpoint as a frozen foundation."""
    if isinstance(foundation, Path):
        foundation_dir = foundation
    else:
        path = Path(foundation)
        foundation_dir = path if path.is_dir() else resolve_run_directory(foundation)
    if corrector_foundation_dir(foundation_dir) is not None:
        raise ValueError("A neural corrector checkpoint cannot be used as a foundation")
    hp = HyperParams.from_dict(load_config(foundation_dir))
    if soft_bc is not None and hp.soft_bc != soft_bc:
        raise ValueError(
            f"Foundation soft_bc={hp.soft_bc} does not match corrector soft_bc={soft_bc}"
        )
    manager = NetworkManager(hp)
    params = manager.from_disk(foundation_dir / "network.flax")
    return FoundationModel(model=manager.model, params=params), hp, foundation_dir


def _load_composed(
    foundation_dir: Path, corrector_dir: Path, n_validation_size: int = KPI_EVAL_CONFIGS
) -> NetworkManager:
    """Load a frozen foundation plus its neural corrector."""
    if not (corrector_dir / "network.flax").exists():
        raise FileNotFoundError(f"no neural corrector found under {corrector_dir}")

    scale = float(load_run(corrector_dir).get("result", {}).get("corrector_scale", 1.0))
    prior, foundation_hp, foundation_dir = load_foundation(foundation_dir)
    corrector_hp = HyperParams.from_dict(load_config(corrector_dir))
    if foundation_hp.soft_bc != corrector_hp.soft_bc:
        raise ValueError(
            f"Foundation soft_bc={foundation_hp.soft_bc} does not match "
            f"corrector soft_bc={corrector_hp.soft_bc}"
        )
    manager = NetworkManager(
        corrector_hp,
        prior=prior,
        scale=scale,
        n_validation_size=n_validation_size,
    )
    manager.state = manager.state.replace(params=manager.from_disk(corrector_dir / "network.flax"))
    manager.artifact_stem = str(corrector_dir)
    return manager


def load_checkpoint(
    name: str | Path, *, n_validation_size: int = KPI_EVAL_CONFIGS
) -> NetworkManager:
    """Load a plain foundation or a path-owned neural corrector checkpoint."""
    if isinstance(name, Path):
        run_dir = name
    else:
        path = Path(name)
        run_dir = path if path.is_dir() else resolve_run_directory(name)
    direct_corrector = run_dir / NEURAL_CORRECTOR_DIR
    if (direct_corrector / "network.flax").exists():
        return _load_composed(run_dir, direct_corrector, n_validation_size)
    if foundation_dir := corrector_foundation_dir(run_dir):
        return _load_composed(foundation_dir, run_dir, n_validation_size)

    hp = HyperParams.from_dict(load_config(run_dir))
    manager = NetworkManager(hp, n_validation_size=n_validation_size)
    manager.state = manager.state.replace(params=manager.from_disk(run_dir / "network.flax"))
    manager.artifact_stem = str(name)
    return manager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural corrector on top of a frozen foundation checkpoint."
    )
    parser.add_argument("foundation", help="Foundation run slug accepted by resolve_run_directory.")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--hidden-dims", type=str, default="128,128,128,128")
    parser.add_argument(
        "--arch", type=Architecture, choices=list(Architecture), default=Architecture.mlp
    )
    parser.add_argument("--rwf", action="store_true")
    parser.add_argument("--fourier-features", type=int, default=64)
    parser.add_argument("--fourier-sigma", type=float, default=2.0)
    parser.add_argument("--corrector-scale", type=float, default=0.01)
    args = parser.parse_args()

    prior, foundation_hp, foundation_dir = load_foundation(args.foundation)
    warmup = max(1, args.epochs // 6)
    corrector_hp = HyperParams(
        hidden_dims=tuple(int(d) for d in args.hidden_dims.split(",")),
        arch=args.arch,
        rwf=args.rwf,
        n_fourier_features=args.fourier_features,
        fourier_sigma=args.fourier_sigma,
        warmup_epochs=warmup,
        decay_epochs=args.epochs - warmup,
        soft_bc=foundation_hp.soft_bc,
    )
    corrector = NetworkManager(
        corrector_hp,
        prior=prior,
        scale=args.corrector_scale,
        foundation_dir=foundation_dir,
    )
    corrector.train()
    logger.info(f"neural corrector saved to {foundation_dir / NEURAL_CORRECTOR_DIR}")
