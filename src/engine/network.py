from typing import Literal

from flax import linen as nn
import flax.serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from scipy.stats import qmc

from src.engine.physics import pinn_loss_function, toroidal_field_flux_function
from src.engine.plasma import calculate_poloidal_boundary, get_poloidal_points
from src.lib.config import Filepaths
from src.lib.geometry_config import (
    PlasmaConfig,
    PlasmaGeometry,
    PlasmaState,
)
from src.lib.logger import get_logger
from src.lib.network_config import DomainBounds, FluxInput, HyperParams

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
            x = nn.Dense(features=dim, dtype=jnp.float32)(x)
            x = nn.swish(x)

        psi_hat = nn.Dense(features=1)(x)
        return psi_hat


# --- Sampler ---
BASE_SEED = 42
RESAMPLING_FREQUENCY = 10  # Resample training configs every N epochs


class Sampler:
    def __init__(self, config: HyperParams, seed: int = BASE_SEED) -> None:
        self.config = config
        self.seed = seed
        self._domain_lower_bounds, self._domain_upper_bounds = self._build_domain_bounds()

        # Instatiate separate Sobol samplers for domain, interior points, and boundary points.
        self._sobol_domain = qmc.Sobol(
            d=len(self._domain_lower_bounds),
            scramble=True,
            seed=self.seed,
        )
        self._sobol_inner = qmc.Sobol(d=2, scramble=True, seed=self.seed + 1)
        self._sobol_boundary = qmc.Sobol(d=1, scramble=True, seed=self.seed + 2)

        # Pre-computed per-epoch coordinate samples.
        self._theta_int: jnp.ndarray | None = None
        self._rho_int: jnp.ndarray | None = None
        self._theta_b: jnp.ndarray | None = None

        self.precompute_coordinate_samples()

    def _build_domain_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Create lower and upper bounds arrays for parameter sampling."""
        # Get all field names from DomainBounds dataclass
        bound_names = list(DomainBounds.__dataclass_fields__.keys())
        l_bounds = jnp.array([getattr(DomainBounds, name)[0] for name in bound_names])
        u_bounds = jnp.array([getattr(DomainBounds, name)[1] for name in bound_names])
        return l_bounds, u_bounds

    def _get_sobol_sample(
        self,
        n_samples: int,
        lower_bounds: jnp.ndarray | None = None,
        upper_bounds: jnp.ndarray | None = None,
        sobol_sampler: Literal["interior", "boundary", "domain"] | None = None,
    ) -> jnp.ndarray:
        """Generate Sobol sequence samples within specified bounds."""

        sobol_sampler = {
            "interior": self._sobol_inner,
            "boundary": self._sobol_boundary,
            "domain": self._sobol_domain,
        }.get(sobol_sampler, self._sobol_domain)

        sample_unit = jnp.array(sobol_sampler.random(n_samples), dtype=jnp.float32)
        return sample_unit * (upper_bounds - lower_bounds) + lower_bounds

    def precompute_coordinate_samples(
        self,
        n_inner_samples: int | None = None,
        n_boundary_samples: int | None = None,
    ) -> None:
        """Precompute Sobol interior and boundary coordinates for one epoch."""
        n_inner = self.config.n_rz_inner_samples if n_inner_samples is None else n_inner_samples
        n_boundary = (
            self.config.n_rz_boundary_samples if n_boundary_samples is None else n_boundary_samples
        )

        inner_samples = self._get_sobol_sample(
            n_samples=n_inner,
            lower_bounds=jnp.array([0.0, 0.0], dtype=jnp.float32),
            upper_bounds=jnp.array([2 * jnp.pi, 1.0], dtype=jnp.float32),
            sobol_sampler="interior",
        )
        self._theta_int = inner_samples[:, 0]
        self._rho_int = jnp.sqrt(inner_samples[:, 1])

        boundary_samples = self._get_sobol_sample(
            n_samples=n_boundary,
            lower_bounds=jnp.array([0.0], dtype=jnp.float32),
            upper_bounds=jnp.array([2 * jnp.pi], dtype=jnp.float32),
            sobol_sampler="boundary",
        )
        self._theta_b = boundary_samples[:, 0]

    def sample_flux_input(
        self,
        plasma_configs: jnp.ndarray,
    ) -> FluxInput:
        """Sample interior and boundary points for a batch of plasma configurations."""
        if self._theta_int is None or self._rho_int is None or self._theta_b is None:
            self.precompute_coordinate_samples()

        theta_int = self._theta_int
        rho_int = self._rho_int
        theta_b = self._theta_b

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
class NetworkManager:
    def __init__(self, config: HyperParams, seed: int = BASE_SEED) -> None:
        self.config = config
        self.seed = seed
        self.model = FluxPINN(
            hidden_dims=config.hidden_dims,
        )
        self.sampler: Sampler = Sampler(config, seed=self.seed)
        self.state = self._init_state()

        self.train_set = self.sampler._get_sobol_sample(
            n_samples=self.config.n_train,
            lower_bounds=self.sampler._domain_lower_bounds,
            upper_bounds=self.sampler._domain_upper_bounds,
        )

    @staticmethod
    def to_disk(params: jnp.ndarray) -> None:
        """Save Flax model parameters to disk."""
        with open(Filepaths.PINN_PATH, "wb") as f:
            f.write(flax.serialization.to_bytes(params))

    def from_disk(self, pinn_path=None) -> any:  # noqa
        """Load Flax model parameters from disk."""
        pinn_path = Filepaths.PINN_PATH if pinn_path is None else pinn_path
        with open(pinn_path, "rb") as f:
            return flax.serialization.from_bytes(self.state.params, f.read())

    def _init_state(self) -> train_state.TrainState:
        """Initialize the training state with dummy data."""
        key = jax.random.PRNGKey(self.seed)
        d_rz = jnp.ones((1, self.config.n_rz_inner_samples))
        d_p = jnp.ones(1)

        geom = PlasmaGeometry(R0=d_p, a=d_p, kappa=d_p, delta=d_p)
        state = PlasmaState(p0=d_p, F_axis=d_p, pressure_alpha=d_p, field_exponent=d_p)
        boundary = calculate_poloidal_boundary(jnp.zeros(1), geom)

        dummy_config = PlasmaConfig(Geometry=geom, Boundary=boundary, State=state)
        dummy_input = FluxInput(R_sample=d_rz, Z_sample=d_rz, config=dummy_config)

        norm_params, r_n, z_n = dummy_input.normalize()
        params = self.model.init(key, r=r_n, z=z_n, **norm_params)

        steps_per_epoch = self.config.n_train // self.config.batch_size
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate_max,
            warmup_steps=self.config.warmup_epochs * steps_per_epoch,
            decay_steps=self.config.decay_epochs * steps_per_epoch,
            end_value=self.config.learning_rate_min,
        )
        tx = optax.adam(learning_rate=schedule)
        return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    @property
    def epochs(self) -> int:
        """Calculate total epochs."""
        return self.config.warmup_epochs + self.config.decay_epochs

    @staticmethod
    @jax.jit
    def train_step(
        state: train_state.TrainState,
        inputs: FluxInput,
    ) -> tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Perform a single training step using physics-informed gradients.

        Returns:
            Tuple of (updated_state, total_loss, residual_loss, boundary_loss).
        """

        def loss_fn(params: any) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
            @jax.checkpoint
            def psi_fn(p: any, R: jnp.ndarray, Z: jnp.ndarray, cfg: PlasmaConfig) -> jnp.ndarray:
                inp = FluxInput(R_sample=R, Z_sample=Z, config=cfg)
                p_n, r_n, z_n = inp.normalize()
                psi_n = state.apply_fn(p, r=r_n, z=z_n, **p_n)
                return (psi_n * cfg.State.F_axis * cfg.Geometry.a).squeeze()

            total, l_res, l_dir = pinn_loss_function(
                psi_fn, params, inputs.R_sample, inputs.Z_sample, inputs.config
            )
            # Return total for grad, carry components as aux
            return total, (l_res, l_dir)

        (loss, (l_res, l_dir)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss, l_res, l_dir

    def calculate_loss(self, inputs: FluxInput) -> float:
        """Calculate loss for given inputs without updating state."""
        _, loss, _, _ = self.train_step(self.state, inputs)
        return float(loss)

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (total_loss, residual_loss, boundary_loss).
        """
        loss, l_res, l_dir = 0.0, 0.0, 0.0
        self.sampler.precompute_coordinate_samples()

        for i in range(0, len(self.train_set), self.config.batch_size):
            train_batch = self.train_set[i : i + self.config.batch_size]
            inputs = self.sampler.sample_flux_input(plasma_configs=train_batch)
            self.state, loss, l_res, l_dir = self.train_step(state=self.state, inputs=inputs)

        # Periodic dataset resampling
        if epoch % RESAMPLING_FREQUENCY == 0 and epoch > 0:
            self.train_set = self.sampler._get_sobol_sample(
                n_samples=self.config.n_train,
                lower_bounds=self.sampler._domain_lower_bounds,
                upper_bounds=self.sampler._domain_upper_bounds,
            )
        return float(loss), float(l_res), float(l_dir)

    def train(
        self,
        save_to_disk: bool = True,
    ) -> float:
        """
        Train the model.

        Args:
            save_to_disk: Whether to save model params to disk after training

        Returns:
            Final training loss
        """
        logger.info(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            loss_total, l_residual, l_boundary = self.train_epoch(epoch)
            if epoch % 10 == 0 and epoch > 0:
                logger.info(
                    f"Epoch {epoch:5d} | Loss: [bold magenta]{loss_total:.2f}[/bold magenta] "
                    f"(residual={l_residual:.3f}, boundary={l_boundary:.3f})"
                )

        if save_to_disk:
            self.to_disk(params=self.state.params)

    def predict(self, inputs: FluxInput) -> jnp.ndarray:
        """Generate predictions for given inputs."""
        norm_params, r_n, z_n = inputs.normalize()
        psi_norm = self.model.apply(self.state.params, r=r_n, z=z_n, **norm_params)
        return psi_norm * inputs.get_physical_scale()

    def get_psi(self, R: jnp.ndarray, Z: jnp.ndarray, config: PlasmaConfig) -> jnp.ndarray:
        """Calculate physical magnetic flux ψ at given coordinates.

        Args:
            R: Major radial coordinates [m]
            Z: Vertical coordinates [m]
            config: Plasma geometry and state parameters

        Returns:
            (N,) array of ψ in Weber
        """
        R_arr = jnp.atleast_1d(R).flatten()
        Z_arr = jnp.atleast_1d(Z).flatten()

        inputs = FluxInput(
            R_sample=jnp.atleast_2d(R_arr), Z_sample=jnp.atleast_2d(Z_arr), config=config
        )
        norm_params, r_n, z_n = inputs.normalize()

        # Extract normalized coordinates and parameters
        r_n_vec = r_n.squeeze()
        z_n_vec = z_n.squeeze()
        norm_params_single = {k: v.squeeze() for k, v in norm_params.items()}

        scale = inputs.get_physical_scale().squeeze()
        params = self.state.params
        apply_fn = self.model.apply

        @jax.jit
        def _calculate_psi_jit(rn, zn):
            psi_n = apply_fn(params, rn, zn, **norm_params_single)
            return (psi_n * scale).squeeze()

        return _calculate_psi_jit(r_n_vec, z_n_vec)

    def get_b_field(self, R: jnp.ndarray, Z: jnp.ndarray, config: PlasmaConfig) -> jnp.ndarray:
        """Calculate magnetic field from flux function via Grad-Shafranov equilibrium.

        Theory: B = ∇ψ * ∇φ + F(ψ)∇φ in axisymmetric geometry
        Yields: B_R = -(1/R)∂ψ/∂Z, B_Z = (1/R)∂ψ/∂R, B_φ = F(ψ)/R

        Args:
            R: Major radial coordinates [m]
            Z: Vertical coordinates [m]
            config: Plasma geometry and state parameters

        Returns:
            (N, 3) array of [B_R, B_Z, B_φ] in Tesla
        """
        # Ensure inputs are 1D arrays for vectorization
        R_arr = jnp.atleast_1d(R).flatten()
        Z_arr = jnp.atleast_1d(Z).flatten()

        # Prepare inputs for normalization
        inputs = FluxInput(
            R_sample=jnp.atleast_2d(R_arr), Z_sample=jnp.atleast_2d(Z_arr), config=config
        )
        norm_params, r_n, z_n = inputs.normalize()

        # Extract normalized coordinates and parameters
        r_n_vec = r_n.squeeze()
        z_n_vec = z_n.squeeze()
        norm_params_single = {k: v.squeeze() for k, v in norm_params.items()}

        scale = inputs.get_physical_scale().squeeze()
        a = config.Geometry.a
        params = self.state.params
        apply_fn = self.model.apply

        @jax.jit
        def _calculate_b_jit(r_vec, z_vec, R_phys):
            def psi_phys(rn, zn):
                """Scalar physical psi for differentiation."""
                # Ensure inputs are JAX arrays for .shape access in model
                rn_arr = jnp.asarray(rn)
                zn_arr = jnp.asarray(zn)
                psi_n = apply_fn(params, rn_arr, zn_arr, **norm_params_single)
                return (psi_n * scale).squeeze()

            # Compute derivatives w.r.t normalized coordinates
            grad_psi_fn = jax.vmap(jax.grad(psi_phys, argnums=(0, 1)))
            dpsi_drn, dpsi_dzn = grad_psi_fn(r_vec, z_vec)

            # Chain rule for physical coordinates: ∂ψ/∂R = (∂ψ/∂r_n) * (1/a)
            dpsi_dR = dpsi_drn / a
            dpsi_dZ = dpsi_dzn / a

            # Poloidal field components
            BR = -dpsi_dZ / R_phys
            BZ = dpsi_dR / R_phys

            # Toroidal field component F(ψ)/R
            psi_vals = jax.vmap(psi_phys)(r_vec, z_vec)
            psi_axis = psi_phys(0.0, 0.0)

            F_val = toroidal_field_flux_function(
                psi_vals, psi_axis, config.State.F_axis, config.State.field_exponent
            )
            Bphi = F_val / R_phys

            return jnp.column_stack([BR, BZ, Bphi])

        return _calculate_b_jit(r_n_vec, z_n_vec, R_arr)

    def get_b_field_cartesian(
        self, X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray, config: PlasmaConfig
    ) -> jnp.ndarray:
        """Transform cylindrical B-field to Cartesian coordinates.

        Coordinate transform: (R, φ, Z) → (X, Y, Z) where R = √(X²+Y²), φ = atan2(Y,X)
        Basis transform: e_R = cos(φ)e_X + sin(φ)e_Y, e_φ = -sin(φ)e_X + cos(φ)e_Y

        Args:
            X, Y, Z: Cartesian coordinates [m]
            config: Plasma parameters

        Returns:
            (N, 3) array of [B_X, B_Y, B_Z] in Tesla
        """
        # Ensure inputs are arrays
        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
        Z = jnp.asarray(Z)

        R = jnp.sqrt(X**2 + Y**2)
        phi = jnp.arctan2(Y, X)

        B_cyl = self.get_b_field(R, Z, config)
        BR, BZ_cyl, Bphi = B_cyl[:, 0], B_cyl[:, 1], B_cyl[:, 2]

        # Rotate cylindrical components to Cartesian basis
        BX = BR * jnp.cos(phi) - Bphi * jnp.sin(phi)
        BY = BR * jnp.sin(phi) + Bphi * jnp.cos(phi)

        return jnp.column_stack([BX, BY, BZ_cyl])


if __name__ == "__main__":
    config = HyperParams()
    manager = NetworkManager(config)
    manager.train(save_to_disk=True)
    # params = manager.from_disk(manager.state.params)
    # manager.state = manager.state.replace(params=params)
