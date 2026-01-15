import os

# Set JAX memory preallocation to false to allow co-existence with PyTorch
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys

from botorch.acquisition import qMaxValueEntropy
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import jax
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from src.engine.network import NetworkManager
from src.lib.logger import get_logger
from src.lib.network_config import BATCH_SIZE, N_TRAIN, HyperParams

logger = get_logger(name="HPO", log_dir="logs/hpo")

# Registry for early stopping: {epoch_milestone: best_loss_at_milestone}
EARLY_STOP_THRESHOLD = 2.0  # N times worse
N_INIT_SAMPLES = 1


class SearchSpace:
    """
    Defines hyperparameter bounds matching the HyperParams class.
    Internal optimization happens in a normalized [0, 1] unit cube.
    """

    # Physical Bounds for interpolation
    hidden_dims_depth = (3, 6)
    hidden_dims_width = (128, 256)
    learning_rate_max = (1e-4, 5e-3)

    # BoTorch bounds: Always [0, 1] for all dimensions
    bounds = torch.tensor(
        [[0.0] * 4, [1.0] * 4],
        dtype=torch.double,
    )

    @classmethod
    def to_hyperparams(cls, x: torch.Tensor) -> HyperParams:
        """Maps unit cube tensor [0, 1]^4 to physical HyperParams."""
        # 0. Depth (Linear interpolation between physical bounds)
        depth = int(
            torch.round(
                cls.hidden_dims_depth[0]
                + x[0] * (cls.hidden_dims_depth[1] - cls.hidden_dims_depth[0])
            ).item()
        )

        # 1. Width (Log-space interpolation for better coverage of orders of magnitude)
        log_w_min, log_w_max = (
            np.log10(cls.hidden_dims_width[0]),
            np.log10(cls.hidden_dims_width[1]),
        )
        width = int(torch.round(10 ** (log_w_min + x[1] * (log_w_max - log_w_min))).item())

        log_lr_min, log_lr_max = (
            np.log10(cls.learning_rate_max[0]),
            np.log10(cls.learning_rate_max[1]),
        )
        lr = float(10 ** (log_lr_min + x[2].item() * (log_lr_max - log_lr_min)))

        return HyperParams(
            hidden_dims=tuple([width] * depth),
            learning_rate_max=lr,
            learning_rate_min=lr * 0.01,
            warmup_epochs=10,
            decay_epochs=50,
            n_train=N_TRAIN,
            batch_size=BATCH_SIZE,
        )


def check_max_capacity(ss: SearchSpace) -> None:
    """
    Smoke test: Initialize the largest possible network and run one step.
    Terminates with error if it fails (e.g., OOM).
    """
    logger.info("Executing capacity check (largest architecture)...")
    try:
        # Use upper bounds to generate max configuration
        # Convert PyTorch tensor to numpy array to avoid JAX tracing issues
        max_bounds = ss.bounds[1].cpu().numpy()
        max_config = ss.to_hyperparams(torch.tensor(max_bounds, dtype=torch.double))

        manager = NetworkManager(max_config)
        train_batch = manager.train_set[0 : max_config.batch_size]
        inputs = manager.sampler.sample_flux_input(
            seed=0,
            n_samples=max_config.n_rz_inner_samples,
            n_boundary_samples=max_config.n_rz_boundary_samples,
            plasma_configs=train_batch,
        )
        # Compilation and first step execution
        manager.state, _ = manager.train_step(manager.state, inputs)

        jax.clear_caches()
        logger.info(
            f"Capacity check passed for: {len(max_config.hidden_dims)}x{max_config.hidden_dims[0]}"
        )
    except Exception:
        logger.exception("CRITICAL: Capacity check failed for max architecture.")
        sys.exit(1)


def optimize_hyperparameters() -> HyperParams:
    ss = SearchSpace()

    # 1. Capacity Check
    check_max_capacity(ss)

    # Local state for pruning
    best_loss = []

    def eval_trial(x: torch.Tensor) -> torch.Tensor:
        """Trains the network and returns negative final loss."""
        config = ss.to_hyperparams(x)
        logger.info(
            f"Evaluating: depth={len(config.hidden_dims)}, width={config.hidden_dims[0]}, "
            f"lr={config.learning_rate_max:.2e}"
        )

        try:
            manager = NetworkManager(config)
            epochs = manager.epochs
            halfway = max(1, epochs // 2)

            logger.info(f"Starting trial training for {epochs} epochs...")

            for epoch in range(epochs):
                loss = manager.train_epoch(epoch)

                if best_loss == []:
                    best_loss.append(loss)

                # Early Stop / Pruning Logic
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch} | Train Loss: {loss:.2f}")

                if epoch == halfway and loss > EARLY_STOP_THRESHOLD * best_loss[0]:
                    logger.warning(
                        f"Early stopping trial at halfway ({halfway}). "
                        f"Loss {loss:.4f} is > {EARLY_STOP_THRESHOLD}x "
                        f"best seen {best_loss[0]:.4f}"
                    )
                    # Return early with current (potentially bad) loss
                    return -torch.tensor([float(loss)], dtype=torch.double)

            jax.clear_caches()
            if loss < best_loss[0]:
                best_loss[0] = loss
            return -torch.tensor([float(loss)], dtype=torch.double)

        except Exception:
            logger.exception("Trial failed due to an unexpected error")
            raise

    # 3. Initial Sampling
    sobol = SobolEngine(dimension=ss.bounds.shape[1], scramble=True, seed=0)
    train_x = ss.bounds[0] + sobol.draw(N_INIT_SAMPLES).to(torch.double) * (
        ss.bounds[1] - ss.bounds[0]
    )

    logger.info(f"Running {N_INIT_SAMPLES} initial trials...")
    train_y = torch.cat([eval_trial(x) for x in train_x]).unsqueeze(-1)

    # 4. BO Loop
    for i in range(N_OPTIMIZATION_STEPS):
        logger.info(f"\n--- BO Iteration {i + 1}/{N_OPTIMIZATION_STEPS} ---")

        # Standardize outcomes internally for GP numerical stability and better prior matching
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        candidate_set = torch.rand(1000, ss.bounds.shape[1], dtype=torch.double)
        qMES = qMaxValueEntropy(model, candidate_set)

        new_x, _ = optimize_acqf(qMES, bounds=ss.bounds, q=1, num_restarts=5, raw_samples=20)

        new_y = eval_trial(new_x.squeeze(0)).unsqueeze(-1)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])

        # Report best loss in physical scale (positive value)
        logger.info(f"Iteration {i + 1} Best Val Loss: {-train_y.max().item():.4e}")

    # 5. Summary
    best_idx = train_y.argmax()
    best_hp: HyperParams = ss.to_hyperparams(train_x[best_idx])
    logger.info("\n" + "=" * 40)
    logger.info("HYPERPARAMETER OPTIMIZATION COMPLETE")
    logger.info(f"Best Depth:      {len(best_hp.hidden_dims)}")
    logger.info(f"Best Width:      {best_hp.hidden_dims[0]}")
    logger.info(f"Best LR:         {best_hp.learning_rate_max:.2e}")
    logger.info(f"Best Val Loss:   {-train_y.max().item():.4f}")
    logger.info("=" * 40)
    return best_hp


if __name__ == "__main__":
    try:
        optimal_hparams = optimize_hyperparameters()
        hyperparams = HyperParams(
            learning_rate_max=optimal_hparams.learning_rate_max,
            learning_rate_min=optimal_hparams.learning_rate_min,
            hidden_dims=optimal_hparams.hidden_dims,
            warmup_epochs=200,
            decay_epochs=1000,
        )
        nn_manager = NetworkManager(hyperparams)
        nn_manager.train()
    except KeyboardInterrupt:
        logger.info("\n[bold red]Optimization interrupted by user. Cleaning up...[/bold red]")
        jax.clear_caches()
        sys.exit(0)
