# Cache the Boundary Fourier Fit

Every hard-BC flux query rebuilt the boundary-radius design matrix and solved
the same ridge system. `PlasmaBoundary` now stores coefficients computed once
at boundary construction, leaving only the point-dependent basis evaluation in
the differentiated training path.

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
```

| Revision | Train step | Temporary memory |
|---|---:|---:|
| Query-time fit | 33.399 ms | 855.6 MiB |
| Cached fit, run A | 32.851 ms | 855.6 MiB |
| Cached fit, run B | 32.934 ms | 855.6 MiB |

The two cached runs average 32.893 ms, a 1.5% incremental improvement.
