# Fuse Axis and Operator Passes

The PINN loss evaluated the network once to estimate `psi_axis`, then evaluated
the same interior points again while computing the Grad-Shafranov operator. The
operator already produces the primal flux, so the optimized loss reuses it for
axis estimation, collapse protection, and the profile source terms.

Benchmark on an RTX 3060 12 GB with JAX GPU, after removing the redundant outer
loss checkpoint and using the direct-training defaults:

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
for _ in range(30):
    output = manager.train_step(*args)
jax.block_until_ready(output)
print((time.perf_counter() - start) * 1000 / 30)
print(manager.train_step.lower(*args).compile().memory_analysis())
```

| Revision | Train step | Temporary memory |
|---|---:|---:|
| Separate axis pass | 36.320 ms | 861.0 MiB |
| Fused axis/operator pass | 33.572 ms | 855.6 MiB |

The fused pass is 7.6% faster incrementally and 21.4% faster than the original
42.695 ms baseline.
