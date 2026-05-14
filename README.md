# NeuroWave

**GPU-Accelerated Electromagnetic Simulation for Biomedical Microwave Imaging**

---

## What is NeuroWave?

NeuroWave is a GPU-native FDTD (Finite-Difference Time-Domain) electromagnetic solver built for biomedical microwave imaging. It runs Maxwell's equations on NVIDIA GPUs via CuPy, with specialized modules for tissue modeling, antenna arrays, and image reconstruction.

**Key capability:** Simulate a 16-element antenna array performing multistatic imaging of a head phantom in under 4 seconds on a T4 GPU.

## Installation

```bash
git clone https://github.com/shahzaibshazoo/ceep-v1.git
cd ceep-v1
pip install -e .

# For GPU support (requires CUDA 12.x)
pip install cupy-cuda12x
```

### Google Colab (GPU)

```python
!git clone https://github.com/shahzaibshazoo/ceep-v1.git
%cd ceep-v1
!pip install -e . -q
!pip install cupy-cuda12x -q
```

Select **Runtime > Change runtime type > T4 GPU**.

---

## Quick Start: Simulate Microwave Head Imaging

This is a complete example that:
1. Creates a 16-antenna circular array
2. Places a head phantom with realistic tissue properties
3. Runs multistatic acquisition (all TX-RX pairs)
4. Reconstructs a dielectric contrast image using DAS beamforming

```python
import numpy as np
from neurowave.core.backend import set_backend
from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D
from neurowave.imaging.beamforming import DASBeamformer, ImagingRegion

# Use GPU
set_backend('cupy')

# Grid: 300x300 at 1mm resolution (30cm x 30cm domain)
nx, ny = 300, 300
dx = 1e-3  # 1 mm

# 16-element circular antenna array
num_antennas = 16
center = (nx // 2, ny // 2)
radius_cells = 100  # 10 cm from center

positions = []
for i in range(num_antennas):
    angle = 2 * np.pi * i / num_antennas
    x = int(center[0] + radius_cells * np.cos(angle))
    y = int(center[1] + radius_cells * np.sin(angle))
    positions.append((x, y))

# Create batched solver (all 16 TX run in parallel on GPU)
solver = BatchedFDTD2D(
    nx=nx, ny=ny, dx=dx,
    total_steps=400,
    cpml_thickness=10,
    source_positions=positions,
    probe_positions=positions,
    frequency=1.0e9,
)

# Add head phantom
solver.set_material_circle(
    center_x=150, center_y=150, radius=60,
    eps_r=40.0, sigma_e=0.7
)

# Run (all 16 transmissions simultaneously)
s_matrix = solver.run()

# s_matrix[tx][rx] = time-domain signal at RX when TX transmits
print(f"Acquired {num_antennas}x{num_antennas} = {num_antennas**2} channels")
```

---

## Complete Imaging Pipeline

### Step 1: Define the Phantom

```python
from neurowave.phantoms.head_models import SimpleHeadPhantom

# 3-layer head: skin, skull, brain
phantom = SimpleHeadPhantom(
    center=(150, 150),
    head_radius_mm=80,
    dx=1e-3,
    frequency=1.0e9
)

# Apply to solver grid
phantom.apply_to_grid(solver)
```

For detailed models with hemorrhage:

```python
from neurowave.phantoms.head_models import DetailedBrainPhantom

phantom = DetailedBrainPhantom(
    center=(150, 150),
    head_radius_mm=80,
    dx=1e-3,
    frequency=1.0e9
)
# Add a simulated blood clot
phantom.add_hemorrhage(
    position=(170, 160),
    radius_mm=10,
    hemorrhage_type='intracerebral'
)
phantom.apply_to_grid(solver)
```

### Step 2: Configure the Antenna Array

```python
from neurowave.antennas.arrays import CircularArray

array = CircularArray(
    num_antennas=16,
    radius_mm=120,       # 12 cm from head center
    center=(150, 150),
    dx=1e-3,
    antenna_type='monopole',
    polarization='vertical'
)

positions = array.get_antenna_positions()
```

### Step 3: Run Multistatic Acquisition

The batched solver runs all transmissions simultaneously on GPU:

```python
from neurowave.core.backend import set_backend
from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D

set_backend('cupy')

solver = BatchedFDTD2D(
    nx=300, ny=300, dx=1e-3,
    total_steps=400,
    cpml_thickness=10,
    source_positions=positions,
    probe_positions=positions,
    frequency=1.0e9,
)

# Apply phantom materials...
# solver.set_material_circle(...)

s_matrix = solver.run()  # Returns dict: s_matrix[tx_idx][rx_idx] = signal
```

### Step 4: Image Reconstruction

```python
from neurowave.imaging.beamforming import DASBeamformer, ImagingRegion

# Define imaging region (inside the antenna ring)
region = ImagingRegion(
    x_range=(50, 250),
    y_range=(50, 250),
    pixel_size=2  # pixels per grid cell
)

# Reconstruct
beamformer = DASBeamformer(
    antenna_positions=positions,
    imaging_region=region,
    dx=1e-3,
    c_background=3e8 / np.sqrt(40)  # speed in brain tissue
)

image = beamformer.reconstruct(s_matrix, dt=solver.dt)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(image.T, cmap='hot', origin='lower')
plt.colorbar(label='Intensity')
plt.title('Microwave Imaging Reconstruction')
plt.show()
```

---

## CPU vs GPU Performance

The solver automatically uses fused CUDA kernels on GPU:

| Grid | Antennas | CPU Time | GPU Time | Speedup |
|------|----------|----------|----------|---------|
| 300x300 | 16 (batched) | 15.2s | 3.6s | 4.3x |
| 300x300 | 16 (sequential) | 18.3s | 91.4s | 0.2x |

**The batched solver is critical.** Sequential per-antenna GPU runs are slower than CPU because 90K cells don't saturate GPU cores. The batched solver stacks all 16 grids into shape (16, 300, 300) = 1.44M cells, properly utilizing all CUDA cores.

```python
# Always use BatchedFDTD2D for multi-antenna simulations
from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D

# For single simulations or CPU-only work
from neurowave.solvers.fdtd_2d import FDTD2D
```

---

## Architecture

```
src/neurowave/
├── core/
│   ├── backend.py          # NumPy/CuPy abstraction layer
│   ├── config.py           # Grid and simulation configuration
│   ├── grid.py             # 2D Yee grid (field arrays + materials)
│   └── grid_3d.py          # 3D Yee grid
├── solvers/
│   ├── fdtd_2d.py          # Single 2D FDTD solver
│   ├── fdtd_2d_batched.py  # Batched solver (multiple sims on GPU)
│   ├── fdtd_3d.py          # 3D FDTD solver
│   ├── dft.py              # DFT frequency extraction
│   ├── s_params.py         # S-parameter extraction
│   └── near_far.py         # Near-to-far field transformation
├── cuda/
│   └── kernels.py          # Fused CUDA RawKernels (2D, 3D, batched)
├── boundaries/
│   └── absorbing.py        # CPML + Mur ABC
├── sources/
│   ├── waveforms.py        # Gaussian, modulated Gaussian sources
│   └── plane_wave.py       # TF/SF plane wave injection
├── materials/
│   ├── dispersive.py       # Debye, Drude, Lorentz, Cole-Cole models
│   └── tissue_database.py  # Gabriel tissue properties (50+ tissues)
├── antennas/
│   └── arrays.py           # Circular, planar, ULA, URA, L-shaped arrays
├── phantoms/
│   └── head_models.py      # Multilayer head phantoms with hemorrhage
└── imaging/
    └── beamforming.py      # DAS image reconstruction
```

---

## Dispersive Materials

NeuroWave supports frequency-dependent materials via Auxiliary Differential Equations (ADE):

```python
from neurowave.materials.dispersive import DebyePole, DrudePole, LorentzPole

# Debye model (biological tissues, water)
grid.set_material_region(
    x_start=50, x_end=100, y_start=50, y_end=100,
    eps_r=4.0, eps_inf=4.0, sigma_e=0.02,
    debye_poles=[DebyePole(delta_eps=72.0, tau=9.4e-12)]
)

# Drude model (metals, plasmas)
grid.set_material_region(
    ...,
    debye_poles=[DrudePole(omega_p=1.37e16, gamma=4.05e13)]
)

# Cole-Cole model (broadband tissue dispersion)
from neurowave.materials.dispersive import ColeColePole
grid.set_material_circle(
    center_x=100, center_y=100, radius=30,
    eps_r=50.0, eps_inf=4.0, sigma_e=0.7,
    debye_poles=[ColeColePole(delta_eps=46.0, tau=7.96e-12, alpha=0.1)]
)
```

### Tissue Database (Gabriel et al.)

```python
from neurowave.materials.tissue_database import TissueDatabase

db = TissueDatabase()

# Get tissue properties at a specific frequency
brain = db.get_tissue('brain_grey_matter', frequency=1e9)
print(f"Brain at 1 GHz: eps_r={brain.eps_r:.1f}, sigma={brain.sigma:.2f} S/m")

# Available tissues
print(db.list_tissues())
```

---

## Antenna Arrays

```python
from neurowave.antennas.arrays import (
    CircularArray,      # Head imaging
    PlanarArray,        # Breast imaging
    UniformLinearArray, # Beamforming / radar
    UniformRectangularArray,  # Massive MIMO
    LShapedArray,       # 2D DOA estimation
    RandomArray,        # Compressed sensing
    ConformalArray,     # Body-fitted wearables
)

# Example: create standard head imaging array
from neurowave.antennas.arrays import create_imaging_array
array = create_imaging_array('circular', 'head', grid_shape=(300, 300), dx=1e-3)
```

---

## Boundary Conditions

```python
from neurowave.boundaries.absorbing import CPML, MurABC

# CPML (recommended — 10-20 cells thick)
cpml = CPML(thickness=10, order=3, sigma_max_factor=0.8)

# Mur 1st-order ABC (simpler, less reflections)
mur = MurABC()
```

---

## Running on Google Colab

Complete Colab cells are provided in `colab_cells/`:

| Cell | Description |
|------|-------------|
| `cell1_setup.py` | Clone repo, install dependencies |
| `cell5_gpu_cpu_comparison.py` | Basic GPU vs CPU comparison |
| `cell10_antenna_array_16x16.py` | Sequential 16-antenna sim |
| `cell11_batched_antenna_array.py` | Batched 16-antenna sim (fast) |

### Typical Colab workflow:

```python
# Cell 1: Setup
!git clone https://YOUR_TOKEN@github.com/shahzaibshazoo/ceep-v1.git
%cd ceep-v1
!pip install -e . -q && pip install cupy-cuda12x -q

# Cell 2: Run batched imaging simulation
%run colab_cells/cell11_batched_antenna_array.py
```

---

## Single Simulation (No Array)

For simpler problems without antenna arrays:

```python
from neurowave.core.backend import set_backend
from neurowave.core.config import GridConfig, SimulationConfig
from neurowave.sources.waveforms import GaussianSource
from neurowave.solvers.fdtd_2d import FDTD2D
from neurowave.boundaries.absorbing import CPML

set_backend('cupy')  # or 'numpy' for CPU

config = SimulationConfig(
    grid=GridConfig(nx=200, ny=200, dx=1e-3, dy=1e-3),
    total_steps=300
)

source = GaussianSource(
    x=100, y=100,
    frequency_max=2e9,
    field_component='Ez',
    delay_factor=5.0
)

solver = FDTD2D(
    config=config,
    sources=[source],
    boundaries=[CPML(thickness=10)],
    probe_points=[(50, 50), (150, 150)],
    record_field='Ez'
)

solver.run()

# Access results
import matplotlib.pyplot as plt
plt.imshow(solver.field_snapshots[-1].T, cmap='RdBu', origin='lower')
plt.show()
```

---

## 3D Simulations

```python
from neurowave.core.config import GridConfig, SimulationConfig
from neurowave.solvers.fdtd_3d import FDTD3D

config = SimulationConfig(
    grid=GridConfig(nx=100, ny=100, nz=100, dx=1e-3, dy=1e-3, dz=1e-3),
    total_steps=200
)

solver = FDTD3D(config=config, sources=[...], boundaries=[...])
solver.run()
```

---

## Tests

```bash
# Run all tests
python -m pytest tests/ -q

# Run specific test
python -m pytest tests/test_fdtd_2d.py -v
```

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | Done | Core 2D FDTD engine (TMz/TEz) |
| Phase 2A | Done | CPML absorbing boundaries |
| Phase 2B | Done | Debye/Drude/Lorentz/Cole-Cole dispersive materials |
| Phase 2C | Done | Near-to-far field, S-parameters, DFT monitors |
| Phase 3 | Done | 3D FDTD engine |
| Phase 4 | Done | Biomedical: tissue database, phantoms, beamforming |
| Phase 5 | Done | Antenna arrays (circular, planar, ULA, URA, etc.) |
| Phase 6 | Done | GPU acceleration (CuPy backend, fused CUDA kernels, batched solver) |

---

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{neurowave2026,
  title={NeuroWave: GPU-Accelerated Electromagnetic Simulation for Biomedical Imaging},
  year={2026},
  url={https://github.com/shahzaibshazoo/ceep-v1}
}
```
