# Microwave Imaging Pipeline

This guide walks through a complete biomedical microwave imaging simulation — from phantom creation to image reconstruction.

## Overview

The imaging pipeline has 4 stages:

1. **Phantom** — Define the target (e.g., head with hemorrhage)
2. **Array** — Configure antenna positions
3. **Acquisition** — Run multistatic FDTD simulations
4. **Reconstruction** — Beamform the S-matrix into an image

## Step 1: Create a Head Phantom

### Simple Phantom (manual)

```python
from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D

solver = BatchedFDTD2D(nx=300, ny=300, dx=1e-3, ...)

# Brain tissue
solver.set_material_circle(150, 150, 60, eps_r=40.0, sigma_e=0.7)
```

### Realistic Multilayer Phantom

```python
from neurowave.phantoms.head_models import SimpleHeadPhantom

phantom = SimpleHeadPhantom(
    center=(150, 150),
    head_radius_mm=80,
    dx=1e-3,
    frequency=1.0e9
)

# Layers: skin (3mm), skull (7mm), brain
# Properties from Gabriel tissue database at 1 GHz
```

### Adding a Hemorrhage

```python
from neurowave.phantoms.head_models import DetailedBrainPhantom

phantom = DetailedBrainPhantom(
    center=(150, 150),
    head_radius_mm=80,
    dx=1e-3,
    frequency=1.0e9
)

# Blood clot at (170, 160), 10mm radius
phantom.add_hemorrhage(
    position=(170, 160),
    radius_mm=10,
    hemorrhage_type='intracerebral'
)
```

## Step 2: Configure Antenna Array

```python
from neurowave.antennas.arrays import CircularArray
import numpy as np

# 16 monopoles in a ring around the head
array = CircularArray(
    num_antennas=16,
    radius_mm=120,
    center=(150, 150),
    dx=1e-3,
    antenna_type='monopole',
    polarization='vertical'
)

positions = array.get_antenna_positions()
# Returns: [(250, 150), (243, 196), (220, 235), ...]
```

## Step 3: Multistatic Acquisition (Batched GPU)

```python
from neurowave.core.backend import set_backend
from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D

set_backend('cupy')

solver = BatchedFDTD2D(
    nx=300, ny=300, dx=1e-3,
    total_steps=400,
    cpml_thickness=10,
    source_positions=positions,  # Each TX is one batch element
    probe_positions=positions,   # All antennas also receive
    frequency=1.0e9,
)

# Apply phantom materials to the shared grid
solver.set_material_circle(150, 150, 60, eps_r=40.0, sigma_e=0.7)

# Run all 16 TX simultaneously
s_matrix = solver.run()

# s_matrix[tx_idx][rx_idx] = numpy array of shape (total_steps,)
# Contains 16 × 16 = 256 time-domain signals
```

## Step 4: Image Reconstruction

```python
from neurowave.imaging.beamforming import DASBeamformer, ImagingRegion
import numpy as np

# Define the area to image (inside the antenna ring)
region = ImagingRegion(
    x_range=(60, 240),
    y_range=(60, 240),
    pixel_size=2
)

# DAS beamformer
beamformer = DASBeamformer(
    antenna_positions=positions,
    imaging_region=region,
    dx=1e-3,
    c_background=3e8 / np.sqrt(40)  # Speed in brain tissue
)

# Reconstruct
image = beamformer.reconstruct(s_matrix, dt=solver.dt)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(image.T, cmap='hot', origin='lower', extent=[60,240,60,240])
plt.colorbar(label='Intensity (a.u.)')
plt.title('DAS Microwave Image')
plt.xlabel('X (cells)')
plt.ylabel('Y (cells)')
plt.show()
```

## Complete Example

```python
import numpy as np
from neurowave.core.backend import set_backend
from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D

set_backend('cupy')

# Parameters
nx, ny = 300, 300
dx = 1e-3
num_ant = 16
center = (150, 150)

# Antenna positions
positions = [
    (int(150 + 100*np.cos(2*np.pi*i/16)),
     int(150 + 100*np.sin(2*np.pi*i/16)))
    for i in range(16)
]

# Solver
solver = BatchedFDTD2D(
    nx=nx, ny=ny, dx=dx,
    total_steps=400,
    cpml_thickness=10,
    source_positions=positions,
    probe_positions=positions,
    frequency=1.0e9,
)

# Head (brain tissue)
solver.set_material_circle(150, 150, 60, eps_r=40.0, sigma_e=0.7)

# Run
s_matrix = solver.run()

print(f"Acquired {num_ant**2} channels")
print(f"Signal shape: {s_matrix[0][0].shape}")
```

## Tips

- **Grid size**: 1mm resolution at 1 GHz gives ~10 cells per wavelength in brain — adequate for imaging
- **Timesteps**: 400 steps at CFL limit covers ~2 domain traversals — enough for reflections
- **CPML**: 10 cells is minimal; use 15-20 for cleaner results
- **Larger arrays**: The batched solver handles 32 or 64 antennas; just increases GPU memory linearly
