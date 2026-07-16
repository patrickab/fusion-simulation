# Remove Redundant Loss Rematerialization

`train_step` rematerialized the complete loss even though the expensive
Grad-Shafranov residual already has a targeted checkpoint. Removing the outer
checkpoint avoids recomputing the loss during backpropagation without increasing
XLA's temporary-memory estimate.

Benchmark on an RTX 3060 12 GB with JAX GPU, using the direct-training defaults
(4x128 MLP, 64 Fourier features, batch 64, 512 interior points):

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
| Before | 42.695 ms | 863.0 MiB |
| After | 36.320 ms | 861.0 MiB |

The optimized step is 14.9% faster.
