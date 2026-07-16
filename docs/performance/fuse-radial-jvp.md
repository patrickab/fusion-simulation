# Fuse the Radial JVP

The GS operator previously evaluated a radial JVP for `psi` and `dpsi/dR`, then
ran a separate nested radial JVP for `d2psi/dR2`. Nesting the value-and-first-
derivative function returns all three terms in one traversal and removes the
duplicate radial network work.

Benchmark on an RTX 3060 12 GB with JAX GPU, after the preceding training
optimizations and using the direct-training defaults:

```python
import time

import jax

from src.engine.network import NetworkManager
from src.lib.network_config import HyperParams

config = HyperParams(
    hidden_dims=(128,) * 4,
    n_fourier_features=64,
    huber_delta=1.0,
    batch_size=64,
    n_rz_inner_samples=512,
    n_rz_boundary_samples=128,
    n_train=1024,
    warmup_epochs=1,
    decay_epochs=1,
)
manager = NetworkManager(config, test_mode=True, n_validation_size=16)
inputs = manager.sampler.sample_flux_input(manager.train_set[: config.batch_size])
args = (
    manager.state,
    inputs,
    config.weight_boundary_condition,
    config.huber_delta,
    config.weight_flux_scale,
    config.soft_bc,
)

output = manager.train_step(*args)
jax.block_until_ready(output)
start = time.perf_counter()
for _ in range(100):
    output = manager.train_step(*args)
jax.block_until_ready(output)
print((time.perf_counter() - start) * 10)
print(manager.train_step.lower(*args).compile().memory_analysis())
```

| Revision | Train step | Temporary memory |
|---|---:|---:|
| Separate radial passes | 32.893 ms | 855.6 MiB |
| Fused radial pass, run A | 28.471 ms | 822.8 MiB |
| Fused radial pass, run B | 28.438 ms | 822.8 MiB |

The fused pass is 13.5% faster incrementally and 33.3% faster than the original
42.695 ms baseline.
