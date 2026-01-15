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
from src.lib.network_config import HyperParams

logger = get_logger(name="HPO", log_dir="logs/hpo")

# Registry for early stopping: {epoch_milestone: best_loss_at_milestone}
EARLY_STOP_THRESHOLD = 2.0  # N times worse


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

        # 2. Learning Rate (Linear interpolation on original scale as requested)
        lr = (
            cls.learning_rate_max[0] + x[2] * (cls.learning_rate_max[1] - cls.learning_rate_max[0])
        ).item()

        return HyperParams(
            hidden_dims=tuple([width] * depth),
            learning_rate_max=lr,
            learning_rate_min=lr * 0.01,
            warmup_steps=100,
            decay_steps=500,
            n_train=1024,
            batch_size=128,
        )


def check_max_capacity(ss: SearchSpace) -> None:
    """
    Smoke test: Initialize the largest possible network and run one step.
    Terminates with error if it fails (e.g., OOM).
    """
    logger.info("Executing capacity check (largest architecture)...")
    try:
        # Use upper bounds to generate max configuration
        max_config = ss.to_hyperparams(ss.bounds[1])

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
    except Exception as e:
        logger.error(f"CRITICAL: Capacity check failed for max architecture. Error: {e}")
        sys.exit(1)


def optimize_hyperparameters() -> HyperParams:
    ss = SearchSpace()

    # 1. Capacity Check
    check_max_capacity(ss)

    # 2. Setup fixed validation set
    dummy_manager = NetworkManager(HyperParams())
    l_bounds, u_bounds = dummy_manager.sampler._build_domain_bounds()
    val_configs = dummy_manager.sampler._get_sobol_sample(
        n_samples=128, seed=42, lower_bounds=l_bounds, upper_bounds=u_bounds
    )
    val_inputs = dummy_manager.sampler.sample_flux_input(
        seed=123,
        n_samples=HyperParams().n_rz_inner_samples,
        n_boundary_samples=HyperParams().n_rz_boundary_samples,
        plasma_configs=val_configs,
    )
    del dummy_manager
    jax.clear_caches()

    # Local state for pruning
    milestone_best_loss = {}

    def eval_trial(x: torch.Tensor) -> torch.Tensor:
        """Trains the network and returns negative validation loss."""
        config = ss.to_hyperparams(x)
        logger.info(
            f"Evaluating: depth={len(config.hidden_dims)}, width={config.hidden_dims[0]}, "
            f"lr={config.learning_rate_max:.2e}"
        )

        try:
            manager = NetworkManager(config)
            epochs = config.decay_steps + config.warmup_steps

            def prune_callback(epoch: int, loss: float) -> bool:
                # Intermediate Reporting & Early Stop logic
                if epoch % 50 == 0:
                    milestone = epoch
                    logger.info(f"Epoch {milestone} | Train Loss: {loss:.2f}")

                    if milestone not in milestone_best_loss:
                        milestone_best_loss[milestone] = loss
                    else:
                        best_seen = milestone_best_loss[milestone]
                        halfway = max(1, epochs // 2)
                        if (
                            loss > EARLY_STOP_THRESHOLD * best_seen
                            and epoch % halfway != 0
                            and epoch != 0
                        ):
                            logger.warning(
                                f"Early stopping trial at epoch {milestone}. "
                                f"Loss {loss:.4f} is > {EARLY_STOP_THRESHOLD}x "
                                f"best seen {best_seen:.4f}"
                            )
                            return True  # Stop training

                        if loss < best_seen:
                            milestone_best_loss[milestone] = loss
                return False

            logger.info(f"Starting trial training for {epochs} epochs...")

            # Train and get final loss (validation loss because we pass val_inputs)
            val_loss = manager.train(
                validation_inputs=val_inputs,
                callback=prune_callback,
                save_to_disk=False,
            )

            jax.clear_caches()
            return -torch.tensor([float(val_loss)], dtype=torch.double)

        except Exception:
            logger.exception("Trial failed due to an unexpected error")
            return torch.tensor([-10.0], dtype=torch.double)

    # 3. Initial Sampling
    n_init = 5
    sobol = SobolEngine(dimension=ss.bounds.shape[1], scramble=True, seed=0)
    train_x = ss.bounds[0] + sobol.draw(n_init).to(torch.double) * (ss.bounds[1] - ss.bounds[0])

    logger.info(f"Running {n_init} initial trials...")
    train_y = torch.cat([eval_trial(x) for x in train_x]).unsqueeze(-1)

    # 4. BO Loop
    n_iterations = 15
    for i in range(n_iterations):
        logger.info(f"\n--- BO Iteration {i + 1}/{n_iterations} ---")

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
            warmup_steps=200,
            decay_steps=1000,
        )
        nn_manager = NetworkManager(optimal_hparams)
        nn_manager.train()
    except KeyboardInterrupt:
        logger.info("\n[bold red]Optimization interrupted by user. Cleaning up...[/bold red]")
        jax.clear_caches()
        sys.exit(0)
