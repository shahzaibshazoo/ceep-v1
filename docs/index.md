# NeuroWave

**GPU-Accelerated FDTD Electromagnetic Simulation for Biomedical Microwave Imaging**

---

## What is NeuroWave?

NeuroWave is a Python-native FDTD (Finite-Difference Time-Domain) solver that runs Maxwell's equations on NVIDIA GPUs. It is purpose-built for biomedical microwave imaging — simulating antenna arrays, tissue phantoms, and reconstructing dielectric contrast images.

## Key Features

- **Batched GPU Solver** — Run 16+ antenna simulations simultaneously in a single kernel launch
- **Biological Tissue Database** — Gabriel et al. 4-term Cole-Cole models for 50+ tissue types
- **Antenna Arrays** — Circular, planar, ULA, URA, L-shaped, random, conformal
- **Dispersive Materials** — Debye, Drude, Lorentz, Cole-Cole via ADE method
- **Image Reconstruction** — DAS beamforming from multistatic S-parameters
- **3D Engine** — Full 3D Yee grid with CPML boundaries
- **Validated** — Tested against analytical solutions, 36 unit tests passing

## Performance

On a T4 GPU (Google Colab free tier):

| Simulation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| 16-antenna batched (300x300, 400 steps) | 15.2s | 3.6s | **4.3x** |
| Single sim (1000x1000, 500 steps) | 6.4s | 3.0s | **2.1x** |

## Install

```bash
pip install neurowave

# With GPU support
pip install neurowave[gpu]
```

## Quick Example

```python
from neurowave.core.backend import set_backend
from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D
import numpy as np

set_backend('cupy')  # Enable GPU

# 16-antenna circular array
positions = [(int(150 + 100*np.cos(2*np.pi*i/16)),
              int(150 + 100*np.sin(2*np.pi*i/16))) for i in range(16)]

solver = BatchedFDTD2D(
    nx=300, ny=300, dx=1e-3, total_steps=400,
    cpml_thickness=10,
    source_positions=positions,
    probe_positions=positions,
    frequency=1e9,
)
solver.set_material_circle(150, 150, 60, eps_r=40.0, sigma_e=0.7)

s_matrix = solver.run()  # All 16 TX in parallel on GPU
```
