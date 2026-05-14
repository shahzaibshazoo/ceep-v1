# GPU Acceleration

## Backend System

NeuroWave uses a backend abstraction layer. Switch between CPU and GPU with one call:

```python
from neurowave.core.backend import set_backend

set_backend('numpy')  # CPU (default)
set_backend('cupy')   # NVIDIA GPU
```

All arrays and computations automatically use the active backend.

## When to Use GPU

**GPU is faster when:**

- Grid is large (>500x500 for single sim)
- Running batched simulations (multiple TX antennas)
- 3D simulations (always use GPU)

**CPU is faster when:**

- Grid is small (<300x300 single sim)
- Running sequential per-antenna simulations on small grids
- No GPU available

## Batched Solver (Recommended for Arrays)

The key insight: a small grid (300x300 = 90K cells) doesn't saturate a GPU's thousands of cores. Running 16 sequential simulations means 16 underutilized kernel launches.

The batched solver stacks all simulations into shape `(batch, nx, ny)`:

```python
from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D

# 16 simulations run as ONE kernel with 1.44M cells
solver = BatchedFDTD2D(
    nx=300, ny=300, dx=1e-3,
    total_steps=400,
    cpml_thickness=10,
    source_positions=positions,  # 16 TX locations
    probe_positions=positions,
    frequency=1e9,
)
s_matrix = solver.run()
```

Performance comparison on T4 GPU:

| Method | Time | Why |
|--------|------|-----|
| Sequential GPU (16 × FDTD2D) | 91.4s | 90K cells/kernel = idle GPU |
| **Batched GPU** | **3.6s** | 1.44M cells/kernel = saturated |
| CPU (batched NumPy) | 15.2s | Baseline |

## Fused CUDA Kernels

NeuroWave includes custom `RawKernel` implementations that fuse multiple operations into single launches:

- `update_h_batched_2d` — H-field update for all batch elements
- `update_e_batched_2d` — E-field update for all batch elements
- `inject_sources_batched` — Source injection for all TX
- `record_probes_batched` — Probe recording for all RX

These are automatically used when CuPy is the active backend.

## Memory Estimation

```python
# Memory per simulation: ~7 arrays × nx × ny × 8 bytes (float64)
# Batched: multiply by batch size

batch = 16
nx, ny = 300, 300
mem_mb = batch * 7 * nx * ny * 8 / 1e6
print(f"GPU memory needed: ~{mem_mb:.0f} MB")
# ~72 MB — easily fits on any GPU
```

T4 has 16 GB — supports batches up to ~100 for 300x300 grids, or batch=16 for 1000x1000 grids.

## Checking GPU Status

```python
from neurowave.core.backend import print_backend_info
print_backend_info()
```

```python
# Direct CuPy check
import cupy as cp
print(cp.cuda.Device())
print(f"Free memory: {cp.cuda.Device().mem_info[0]/1e9:.1f} GB")
```
