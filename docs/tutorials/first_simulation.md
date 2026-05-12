# Tutorial: Your First FDTD Simulation

> Learn to run a 2D electromagnetic simulation with NeuroWave

---

## What You'll Build

A 2D FDTD simulation of a Gaussian pulse radiating in free space with PML absorption.

## Prerequisites

```bash
pip install numpy matplotlib
pip install -e .  # Install neurowave in dev mode
```

## Step 1: Import NeuroWave

```python
from neurowave.core.config import GridConfig, SimulationConfig, SimulationMode
from neurowave.sources.waveforms import GaussianSource
from neurowave.boundaries.absorbing import CPML
from neurowave.solvers.fdtd_2d import FDTD2D
from neurowave.visualization.field_plot import plot_field_2d
```

## Step 2: Configure the Grid

```python
# 200×200 grid with 1mm spacing
grid = GridConfig(nx=200, ny=200, dx=1e-3, dy=1e-3)

# TMz mode, Courant number 0.5, run for 300 timesteps
config = SimulationConfig(
    grid=grid,
    mode=SimulationMode.TMZ,
    courant=0.5,
    total_steps=300,
)

print(config.summary())
```

**What's happening?**
- The grid defines a 200mm × 200mm simulation domain
- TMz mode simulates Ez, Hx, Hy field components
- Courant number 0.5 is safely below the stability limit (1/√2 ≈ 0.707)
- The timestep is automatically computed from CFL: Δt ≈ 1.18 ps

## Step 3: Define a Source

```python
# Gaussian pulse at grid center, max frequency 5 GHz
source = GaussianSource(
    x=100, y=100,          # Center of grid
    frequency_max=5e9,     # Maximum frequency content
    amplitude=1.0,
)
```

**Physics notes:**
- At 5 GHz, λ_min = c/f = 60 mm
- With Δx = 1 mm, we have 60 points per wavelength (very good!)
- The pulse has delay t₀ = 6τ to ensure smooth start from zero

## Step 4: Set Up Boundaries

```python
# CPML absorbing boundary (10 cells thick)
boundary = CPML(thickness=10, order=3)
```

**Why CPML?**
- Absorbs outgoing waves with < -60 dB reflection
- Works across all angles and frequencies
- 10 cells is sufficient for most applications

## Step 5: Create and Run the Solver

```python
solver = FDTD2D(
    config=config,
    sources=[source],
    boundaries=[boundary],
    record_field="Ez",        # Record Ez snapshots
    record_interval=10,       # Every 10 steps
)

solver.run()
```

## Step 6: Visualize Results

```python
# Plot final field state
fig = plot_field_2d(
    solver.get_field("Ez"),
    title="Ez Field",
    dx=config.grid.dx,
    dy=config.grid.dy,
    save_path="ez_final.png",
)
```

## Step 7: Add Materials (Optional)

```python
# Add a dielectric slab (eps_r = 4)
solver.grid.set_material_region(
    x_start=80, x_end=120,
    y_start=0, y_end=200,
    eps_r=4.0,
)

# Or a circular scatterer
solver.grid.set_material_circle(
    center_x=150, center_y=100,
    radius=15,
    eps_r=2.5,
    sigma_e=0.01,  # Slightly lossy
)
```

## Full Example

See `examples/basic_2d_fdtd.py` for the complete working script.

## Next Steps

- Try different grid sizes and see how performance scales
- Add dielectric objects and observe reflection/transmission
- Use a `SinusoidalSource` for continuous-wave excitation
- Use probe points to record time-domain signals

---

*Part of the NeuroWave Tutorial series.*
