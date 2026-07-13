"""Overnight sweep driver: benchmark-plan.md R1-R5 plus hard-BC counterparts.

R1-R5 anchor to bb503b0's exact recorded hyperparameters (its config.json),
adding only the mandatory collapse-guard, and vary one axis at a time.
N1-N3 give the hard-BC (new-architecture) family the same 2400-epoch budget.

Usage: uv run python scripts/run_sweep.py <RUN_NAME>
Prints "SWEEP_RUN_NAME <commit>/<run>" on success for the bash driver.
"""

import sys

from src.engine.network import NetworkManager
from src.lib.config import current_commit
from src.lib.network_config import HyperParams

# bb503b0's config.json verbatim; weight_flux_scale is the one mandatory new field.
BB503B0 = {
    "hidden_dims": (128, 128, 128, 128),
    "soft_bc": True,
    "learning_rate_max": 1.8410729205738696e-4,
    "learning_rate_min": 2.636729089671234e-6,
    "weight_decay": 1.7073967431528118e-9,
    "weight_boundary_condition": 20.0,
    "sigma_residual_adaptive_sampling": 0.1745734676972377,
    "batch_size": 32,
    "n_rz_inner_samples": 512,
    "n_rz_boundary_samples": 256,
    "n_train": 1024,
    "weight_flux_scale": 10.0,
    "huber_delta": 0.0,
    "n_fourier_features": 0,
    "lbfgs_steps": 0,
}

# Current hard-BC defaults, pinned explicitly so the sweep is self-describing.
HARD = {
    "hidden_dims": (128, 128, 128, 128),
    "soft_bc": False,
    "learning_rate_max": 2e-4,
    "learning_rate_min": 5e-5,
    "weight_decay": 1e-7,
    "weight_boundary_condition": 10.0,
    "sigma_residual_adaptive_sampling": 0.05,
    "batch_size": 64,
    "n_rz_inner_samples": 512,
    "n_rz_boundary_samples": 128,
    "n_train": 1024,
    "weight_flux_scale": 10.0,
    "huber_delta": 0.0,
    "n_fourier_features": 0,
    "lbfgs_steps": 0,
}


def epochs(n: int) -> dict[str, int]:
    """Split total epochs at bb503b0's warmup ratio (114/600 = 0.19)."""
    warmup = round(n * 0.19)
    return {"warmup_epochs": warmup, "decay_epochs": n - warmup}


RUNS = {
    # closest possible bb503b0 reproduction + mandatory fixes only
    "R1": {**BB503B0, **epochs(600)},
    # budget alone
    "R2": {**BB503B0, **epochs(2400)},
    # was bb503b0's aggressive resampling a bug-compensation artifact?
    "R3": {**BB503B0, **epochs(2400), "sigma_residual_adaptive_sampling": 0.05},
    # Huber vs MSE, unavailable at bb503b0's time
    "R4": {**BB503B0, **epochs(2400), "huber_delta": 1.0},
    # Fourier features, unavailable at bb503b0's time
    "R5": {**BB503B0, **epochs(2400), "n_fourier_features": 64},
    # hard-BC counterpart of R2: envelope instead of soft penalties
    "N1": {**HARD, **epochs(2400)},
    # hard-BC + Fourier features at lit-suggested sigma=1 + Huber
    "N2": {
        **HARD,
        **epochs(2400),
        "n_fourier_features": 64,
        "fourier_sigma": 1.0,
        "huber_delta": 1.0,
    },
    # bb503b0's tuned hyperparameters, hard-BC envelope
    "N3": {**BB503B0, **epochs(2400), "soft_bc": False},
    # --- architecture & batch sweep (user guidance) ---
    # bigger net: 5x200 may beat the 4x128 bb503b0 shape that was never tuned for size
    "R6": {**BB503B0, **epochs(2400), "hidden_dims": (200, 200, 200, 200, 200)},
    "N4": {**HARD, **epochs(2400), "hidden_dims": (200, 200, 200, 200, 200)},
    # larger batch of plasma configs per gradient step
    "R7": {**BB503B0, **epochs(2400), "batch_size": 128},
    "N5": {**HARD, **epochs(2400), "batch_size": 128},
    # --- round 2: combine the winners (N3 = hard-BC + bb503b0 schedule) ---
    "N6": {**BB503B0, **epochs(2400), "soft_bc": False, "hidden_dims": (200,) * 5},
    "N7": {**BB503B0, **epochs(2400), "soft_bc": False, "lbfgs_steps": 100},
}

if __name__ == "__main__":
    name = sys.argv[1]
    manager = NetworkManager(HyperParams(**RUNS[name]))
    try:
        manager.train(save_to_disk=True)
        print(f"SWEEP_RUN_NAME {current_commit()}/{manager.artifact_stem}", flush=True)
    finally:
        manager.discard_unsaved_run()
