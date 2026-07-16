# Batch and JIT the KPI Residual Evaluation

`evaluate_plasma_kpis` looped over plasma configs in Python with un-jitted
per-config `vmap`s, and re-evaluated ψ on the identical Sobol sample a second
time inside `estimate_axis_for_config` just to normalize the residual. The
core is now `evaluate_residual_samples`: configs stacked and `vmap`ped
together, points chunked with `jax.lax.map(batch_size=GRID_EVAL_CHUNK)`, the
whole pass jitted once per architecture, and ψ computed once and reused for
the axis estimate. `evaluate_plasma_kpis` is a thin statistics wrapper over
it, so training-time tracking, post-training eval and HPO ranking share one
code path.

Benchmark on an RTX 3060 12 GB with JAX GPU, checkpoint
`2026_07_16_22_07_48_default_ba6ccac`, 20 validation configs:

```python
import time

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from src.engine.model_evaluation import evaluate_plasma_kpis
from src.engine.network import BASE_SEED, NetworkManager
from src.lib.network_config import DomainBounds, HyperParams

run = "data/benchmarks/2026_07_16_22_07_48_default_ba6ccac"
hp = HyperParams.from_json(f"{run}/config.json")
manager = NetworkManager(hp)
manager.state = manager.state.replace(params=manager.from_disk(f"{run}/network.flax"))

lower, upper = DomainBounds.get_bounds()
sobol = qmc.Sobol(d=len(lower), scramble=True, seed=BASE_SEED + 123)
val = jnp.array(qmc.scale(sobol.random(20), np.asarray(lower), np.asarray(upper)), jnp.float32)
inputs = manager.sampler.sample_flux_input(plasma_configs=val)
configs = [inputs.config[i] for i in range(20)]

evaluate_plasma_kpis(manager, configs, sample_size=2048)  # compile
start = time.perf_counter()
evaluate_plasma_kpis(manager, configs, sample_size=2048)
print(f"{time.perf_counter() - start:.3f}s")
```

| Variant | 20 configs × 2048 pts |
| --- | --- |
| per-config Python loop (before) | 6.28 s |
| batched + jitted, first call (compile) | 2.39 s |
| batched + jitted, cached | 0.037 s |

KPI values match the old implementation to ≤ 2.6e-5 relative (worst key
`loss_p05` at 2.2e-4 relative = 3e-7 absolute — float32 reassociation on the
smallest statistic); `boundary_leak_max` is bit-identical.
