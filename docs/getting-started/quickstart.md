# Quick Start

## Your First Simulation

A simple 2D FDTD simulation with a point source and CPML boundaries:

```python
from neurowave.core.config import GridConfig, SimulationConfig
from neurowave.sources.waveforms import GaussianSource
from neurowave.solvers.fdtd_2d import FDTD2D
from neurowave.boundaries.absorbing import CPML

# Configure: 200x200 grid, 1mm spacing
config = SimulationConfig(
    grid=GridConfig(nx=200, ny=200, dx=1e-3, dy=1e-3),
    total_steps=300
)

# Point source at center
source = GaussianSource(
    x=100, y=100,
    frequency_max=2e9,
    field_component='Ez',
    delay_factor=5.0
)

# Solve
solver = FDTD2D(
    config=config,
    sources=[source],
    boundaries=[CPML(thickness=10)],
    record_field='Ez'
)
solver.run()

# Visualize
import matplotlib.pyplot as plt
plt.imshow(solver.field_snapshots[-1].T, cmap='RdBu', origin='lower')
plt.colorbar(label='Ez (V/m)')
plt.title('Wave Propagation')
plt.show()
```

## Adding Materials

Place a dielectric object in the simulation:

```python
# After creating solver, before running:
solver.grid.set_material_circle(
    center_x=130, center_y=100,
    radius=20,
    eps_r=4.0,      # Relative permittivity
    sigma_e=0.01    # Conductivity (S/m)
)

solver.run()
```

## Recording at Points

Monitor field values at specific locations:

```python
solver = FDTD2D(
    config=config,
    sources=[source],
    boundaries=[CPML(thickness=10)],
    probe_points=[(50, 100), (150, 100)],
    record_field='Ez'
)
solver.run()

# Time-domain data at probe points
import numpy as np
t = np.arange(config.total_steps) * config.dt * 1e9  # ns
plt.plot(t, solver.probe_data[(50, 100)], label='Probe 1')
plt.plot(t, solver.probe_data[(150, 100)], label='Probe 2')
plt.xlabel('Time (ns)')
plt.ylabel('Ez')
plt.legend()
plt.show()
```

## Using GPU

```python
from neurowave.core.backend import set_backend

set_backend('cupy')  # Switch to GPU before creating solver

# Everything else stays the same — the solver automatically uses GPU
solver = FDTD2D(config=config, sources=[source], boundaries=[CPML(thickness=10)])
solver.run()
```
