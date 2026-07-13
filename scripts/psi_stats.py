"""ψ-field gate check per checkpoint: ψ(R₀,0), min/median/max ψ per eval config.

Acceptance gates (problem-analysis-solution-proposal.md §3.2):
ψ(R₀,0) > 0 (pinned sign convention), max|ψ| > 1 Wb (no trivial collapse).
Uses the same Sobol eval configs as model_evaluation (seed BASE_SEED+123).
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from src.engine.network import BASE_SEED, NetworkManager
from src.engine.plasma import get_poloidal_points
from src.lib.network_config import DomainBounds, HyperParams
from src.streamlit.network_utils import resolve_run_directory

parser = argparse.ArgumentParser()
parser.add_argument("networks", nargs="+", help="Checkpoint names as commit/run")
parser.add_argument("--n-configs", type=int, default=8)
args = parser.parse_args()

lower, upper = DomainBounds.get_bounds()
sobol = qmc.Sobol(d=len(lower), scramble=True, seed=BASE_SEED + 123)
val_configs = jnp.array(
    qmc.scale(sobol.random(args.n_configs), np.asarray(lower), np.asarray(upper)),
    dtype=jnp.float32,
)

unit = qmc.Sobol(d=2, scramble=True, seed=BASE_SEED + 124).random_base2(12)
theta = jnp.asarray(2.0 * np.pi * unit[:, 0], dtype=jnp.float32)
rho = jnp.asarray(np.sqrt(unit[:, 1]), dtype=jnp.float32)

for name in args.networks:
    run_dir = resolve_run_directory(name)
    hp = HyperParams.from_json(str(run_dir / "config.json"))
    manager = NetworkManager(hp)
    loaded = manager.from_disk(pinn_path=run_dir / "network.flax")
    manager.state = manager.state.replace(params=loaded)
    psi_fn = manager.make_psi_fn()
    params = manager.state.params
    inputs = manager.sampler.sample_flux_input(plasma_configs=val_configs)
    configs = [inputs.config[i] for i in range(args.n_configs)]

    print(f"\n{name}  (soft_bc={hp.soft_bc})")
    print(
        f"{'cfg':>3} {'psi(R0,0)':>10} {'min':>9} {'mean':>9} {'med':>9} {'max':>9} "
        f"{'hinge floor':>12}"
    )
    for i, cfg in enumerate(configs):
        R, Z = jax.vmap(lambda t, r, c=cfg: get_poloidal_points(t, c.Geometry, r))(theta, rho)
        psi = jax.vmap(lambda r, z, c=cfg, fn=psi_fn, p=params: fn(p, r, z, c))(R, Z)
        p0 = float(jnp.squeeze(psi_fn(params, jnp.asarray(cfg.Geometry.R0), jnp.array(0.0), cfg)))
        floor = 0.05 * float(cfg.State.F_axis) * float(cfg.Geometry.a)
        flags = []
        if float(jnp.mean(psi)) <= 0:
            flags.append("SIGN")
        if float(jnp.max(jnp.abs(psi))) < 1.0:
            flags.append("COLLAPSE")
        print(
            f"{i:>3} {p0:>10.2f} {float(psi.min()):>9.2f} {float(jnp.mean(psi)):>9.2f} "
            f"{float(jnp.median(psi)):>9.2f} {float(psi.max()):>9.2f} "
            f"{floor:>12.2f}  {' '.join(flags)}"
        )
