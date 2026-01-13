from flax import linen as nn
import flax.serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from scipy.stats import qmc

from src.engine.physics import pinn_loss_function
from src.engine.plasma import calculate_poloidal_boundary, get_poloidal_points
from src.lib.config import Filepaths
from src.lib.geometry_config import (
    PlasmaConfig,
    PlasmaGeometry,
    PlasmaState,
)
from src.lib.logger import get_logger
from src.lib.network_config import DomainBounds, FluxInput, HyperParams, normalize_plasma_config

logger = get_logger(
    name="Network",
)


# --- Network (simple MLP) ---
class FluxPINN(nn.Module):
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(
        self,
        r: jnp.ndarray,
        z: jnp.ndarray,
        r0: jnp.ndarray,
        a: jnp.ndarray,
        kappa: jnp.ndarray,
        delta: jnp.ndarray,
        p0: jnp.ndarray,
        f_axis: jnp.ndarray,
        alpha: jnp.ndarray,
        exponent: jnp.ndarray,
    ) -> jnp.ndarray:
        # Broadcast all inputs to match coordinate shape (B, N)
        target_shape = r.shape
        params = [r, z, r0, a, kappa, delta, p0, f_axis, alpha, exponent]
        x = jnp.stack([jnp.broadcast_to(p, target_shape) for p in params], axis=-1)

        for dim in self.hidden_dims:
            x = nn.Dense(features=dim, dtype=jnp.bfloat16)(x)
            x = nn.tanh(x)

        psi_hat = nn.Dense(features=1)(x)
        return psi_hat

    def to_disk(params, filepath: str) -> None:
        """Save Flax model parameters to disk."""
        with open(filepath, "wb") as f:
            f.write(flax.serialization.to_bytes(params))


# --- Sampler ---
BASE_SEED = 42


class Sampler:
    def __init__(self, config: HyperParams) -> None:
        self.config = config

    def _build_domain_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Create lower and upper bounds arrays for parameter sampling."""
        bound_names = ("R0", "a", "kappa", "delta", "p0", "F_axis", "alpha", "exponent")
        l_bounds = jnp.array([getattr(DomainBounds, name)[0] for name in bound_names])
        u_bounds = jnp.array([getattr(DomainBounds, name)[1] for name in bound_names])
        return l_bounds, u_bounds

    def _get_sobol_sample(
        self,
        n_samples: int,
        seed: int,
        lower_bounds: jnp.ndarray,
        upper_bounds: jnp.ndarray,
    ) -> jnp.ndarray:
        """Generate Sobol sequence samples within specified bounds."""
        sampler = qmc.Sobol(d=len(lower_bounds), scramble=True, seed=seed)
        sample_unit = jnp.array(sampler.random(n_samples), dtype=jnp.bfloat16)
        return sample_unit * (upper_bounds - lower_bounds) + lower_bounds

    def sample_flux_input(
        self,
        seed: int,
        n_samples: int,
        n_boundary_samples: int,
        plasma_configs: jnp.ndarray,
    ) -> FluxInput:
        """Sample interior and boundary points for a batch of plasma configurations."""
        sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
        samples = jnp.array(sampler.random(n_samples), dtype=jnp.bfloat16)
        theta_int = samples[:, 0] * 2 * jnp.pi
        rho_int = jnp.sqrt(samples[:, 1])

        # Use Sobol sampling for boundary points
        sampler_b = qmc.Sobol(d=1, scramble=True, seed=seed + 9999)
        theta_b = (
            jnp.array(sampler_b.random(n_boundary_samples), dtype=jnp.bfloat16).flatten()
            * 2
            * jnp.pi
        )

        def compute_single_config(
            plasma_config: jnp.ndarray,
        ) -> tuple[PlasmaConfig, jnp.ndarray, jnp.ndarray]:
            geom = PlasmaGeometry(
                R0=plasma_config[0],
                a=plasma_config[1],
                kappa=plasma_config[2],
                delta=plasma_config[3],
            )
            state = PlasmaState(
                p0=plasma_config[4],
                F_axis=plasma_config[5],
                pressure_alpha=plasma_config[6],
                field_exponent=plasma_config[7],
            )
            boundary = calculate_poloidal_boundary(theta_b, geom)

            # Interior points
            r_interior, z_interior = jax.vmap(lambda t, r: get_poloidal_points(t, geom, r))(
                theta_int, rho_int
            )

            return (
                PlasmaConfig(Geometry=geom, Boundary=boundary, State=state),
                r_interior,
                z_interior,
            )

        configs, R_int, Z_int = jax.vmap(compute_single_config)(plasma_configs)

        return FluxInput(R_sample=R_int, Z_sample=Z_int, config=configs)


# --- Trainer ---
class PINNTrainer:
    def __init__(self, config: HyperParams) -> None:
        self.config = config
        self.model = FluxPINN(hidden_dims=config.hidden_dims)
        self.sampler: Sampler = Sampler(config)
        self.state = self._init_state()

        lower_bounds, upper_bounds = self.sampler._build_domain_bounds()
        self.train_set = self.sampler._get_sobol_sample(
            n_samples=config.n_train,
            seed=BASE_SEED,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def _init_state(self) -> train_state.TrainState:
        """Initialize the training state with dummy data."""
        key = jax.random.PRNGKey(BASE_SEED)
        d_rz = jnp.ones((1, self.config.n_rz_inner_samples))
        d_p = jnp.ones(1)

        geom = PlasmaGeometry(R0=d_p, a=d_p, kappa=d_p, delta=d_p)
        state = PlasmaState(p0=d_p, F_axis=d_p, pressure_alpha=d_p, field_exponent=d_p)
        boundary = calculate_poloidal_boundary(jnp.zeros(1), geom)

        dummy_config = PlasmaConfig(Geometry=geom, Boundary=boundary, State=state)
        dummy_input = FluxInput(R_sample=d_rz, Z_sample=d_rz, config=dummy_config)

        norm_params = dummy_input.get_norm_params()
        r_n, z_n = dummy_input.normalize_coords(dummy_input.R_sample, dummy_input.Z_sample)
        params = self.model.init(key, r=r_n, z=z_n, **norm_params)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate_max,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.decay_steps,
            end_value=self.config.learning_rate_min,
        )
        tx = optax.adam(learning_rate=schedule)
        return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    @staticmethod
    @jax.jit
    def train_step(
        state: train_state.TrainState,
        inputs: FluxInput,
    ) -> tuple[train_state.TrainState, jnp.ndarray]:
        """Perform a single training step using physics-informed gradients."""

        def loss_fn(params: any) -> jnp.ndarray:
            def psi_fn(p: any, R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> jnp.ndarray:
                # Map physical (R, Z) to normalized network input
                r_n, z_n = (R - cfg.Geometry.R0) / cfg.Geometry.a, Z / cfg.Geometry.a
                p_n = normalize_plasma_config(cfg)
                psi_n = state.apply_fn(p, r=r_n, z=z_n, **p_n)

                # Denormalize output: psi = psi_net * (F_axis * a)
                return (psi_n * cfg.State.F_axis * cfg.Geometry.a).squeeze()

            return pinn_loss_function(
                psi_fn, params, inputs.R_sample, inputs.Z_sample, inputs.config
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    def train(self) -> None:
        """Train the model for specified number of epochs."""
        epochs = self.config.decay_steps + self.config.warmup_steps
        logger.info(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            for i in range(0, len(self.train_set), self.config.batch_size):
                train_batch = self.train_set[i : i + self.config.batch_size]
                # Sample RZ points for each plasma configuration in the batch
                inputs = self.sampler.sample_flux_input(
                    seed=epoch + i,
                    n_samples=self.config.n_rz_inner_samples,
                    n_boundary_samples=self.config.n_rz_boundary_samples,
                    plasma_configs=train_batch,
                )
                self.state, loss = self.train_step(state=self.state, inputs=inputs)

            if epoch % 1 == 0 and epoch > 0:
                logger.info(f"Epoch {epoch:5d} | Loss: [bold magenta]{loss:.2f}[/bold magenta] | ")

        self.model.to_disk(params=self.state.params, filepath=Filepaths.PINN_PATH)

    def predict(self, inputs: FluxInput) -> jnp.ndarray:
        """Generate predictions for given inputs."""
        norm_params = inputs.get_norm_params()
        r_n, z_n = inputs.normalize_coords(inputs.R_sample, inputs.Z_sample)
        psi_norm = self.model.apply(self.state.params, r=r_n, z=z_n, **norm_params)
        return psi_norm * inputs.get_physical_scale()


if __name__ == "__main__":
    config = HyperParams()
    trainer = PINNTrainer(config)
    trainer.train()
