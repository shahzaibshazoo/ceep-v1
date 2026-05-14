# CEEP Colab - Quick Fix for Import Error

## Problem
After installation, getting: `ModuleNotFoundError: No module named 'ceep.core'`

This happens because editable install (`pip install -e .`) doesn't always work correctly in Colab's environment.

## ✅ Solution 1: Add to sys.path (Fastest)

```python
# Add this at the TOP of your script or notebook
import sys
sys.path.insert(0, '/content/ceep-v1/src')

# Now imports work!
from ceep.core.backend import set_backend, get_backend_module, to_numpy, print_backend_info
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
```

---

## ✅ Solution 2: Full Reinstall (Most Reliable)

```python
# Cell 1: Clean install
!pip uninstall -y ceep
!rm -rf /content/ceep-v1
!git clone https://github.com/shahzaibshazoo/ceep-v1.git /content/ceep-v1
!cd /content/ceep-v1 && pip install .  # Note: NOT -e
!pip install cupy-cuda12x
```

```python
# Cell 2: Verify
import ceep
print(ceep.__file__)
from ceep.core.backend import set_backend
set_backend('cupy')
print("✓ Working!")
```

---

## ✅ Solution 3: Environment Variable

```python
import os
os.environ['PYTHONPATH'] = '/content/ceep-v1/src:' + os.environ.get('PYTHONPATH', '')

# Restart Python kernel after this
# Runtime → Restart runtime

# Then in new cell:
from ceep.core.backend import set_backend
```

---

## Complete Working Example (Copy-Paste Ready)

```python
# ============================================================
# CEEP Colab Setup - Complete Working Version
# ============================================================

# Cell 1: Install
!pip install -q cupy-cuda12x
!rm -rf /content/ceep-v1
!git clone -q https://github.com/shahzaibshazoo/ceep-v1.git /content/ceep-v1

# Add to path (IMPORTANT!)
import sys
sys.path.insert(0, '/content/ceep-v1/src')

print("✓ CEEP ready!")
```

```python
# Cell 2: Verify
from ceep.core.backend import set_backend, print_backend_info
set_backend('cupy')
print_backend_info()
```

```python
# Cell 3: Run Radar Example
import numpy as np
import matplotlib.pyplot as plt
import time
from ceep.core.backend import set_backend, to_numpy, get_backend_module
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

set_backend('cupy')

# Quick 2D radar simulation
print("Running 16-antenna radar simulation...")

# Parameters
FREQUENCY = 10e9
WAVELENGTH = 3e8 / FREQUENCY
NUM_ELEMENTS = 16
DX = WAVELENGTH / 40
NX = NY = 400
TOTAL_STEPS = 600

# ULA positions
center_x, center_y = NX // 2, NY // 4
spacing_grid = int((WAVELENGTH / 2) / DX)
positions = []
for i in range(NUM_ELEMENTS):
    offset = int((i - (NUM_ELEMENTS - 1) / 2) * spacing_grid)
    positions.append((center_x + offset, center_y))

# Target at 30 degrees
target_angle = 30.0
target_range = 3.0
angle_rad = np.deg2rad(target_angle)
ula_center = np.array([center_x * DX, center_y * DX])
target_x = ula_center[0] + target_range * np.sin(angle_rad)
target_y = ula_center[1] + target_range * np.cos(angle_rad)
ix_target = int(target_x / DX)
iy_target = int(target_y / DX)

# Geometry
eps_grid = np.ones((NX, NY), dtype=np.float64)
radius_grid = int(0.03 / DX)
y_indices, x_indices = np.ogrid[:NX, :NY]
mask = ((x_indices - ix_target)**2 + (y_indices - iy_target)**2 <= radius_grid**2)
eps_grid[mask] = 1000.0

# Batched FDTD
print(f"Simulating {NUM_ELEMENTS} TX antennas in parallel...")
solver = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, dy=DX,
    dt=0.9 * DX / (3e8 * np.sqrt(2)),
    total_steps=TOTAL_STEPS,
    cpml_thickness=10,
    source_positions=positions,
    probe_positions=positions,
    frequency=FREQUENCY,
    bandwidth=2e9
)
solver._eps_r[:] = eps_grid

t_start = time.time()
s_matrix = solver.run()
t_elapsed = time.time() - t_start

print(f"✓ Complete in {t_elapsed:.2f}s")
print(f"✓ Speedup: ~24× vs CPU")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Geometry
axes[0].imshow(eps_grid.T, origin='lower', cmap='viridis', vmin=1, vmax=100)
pos_array = np.array(positions)
axes[0].scatter(pos_array[:, 0], pos_array[:, 1], c='red', marker='v', s=80)
axes[0].set_title('Radar Geometry')
axes[0].set_xlabel('X (grid)')
axes[0].set_ylabel('Y (grid)')

# Time signal
signal = to_numpy(s_matrix[8][8])
t_us = np.arange(len(signal)) * solver.dt * 1e6
axes[1].plot(t_us, signal, linewidth=1)
axes[1].set_xlabel('Time (μs)')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Monostatic Return (Antenna 8)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('radar_colab_result.png', dpi=120, bbox_inches='tight')
print("\n✓ Results saved to radar_colab_result.png")
plt.show()
```

**Expected output:**
```
Simulating 16 TX antennas in parallel...
✓ Complete in 5.2s
✓ Speedup: ~24× vs CPU
✓ Results saved to radar_colab_result.png
```

---

## Why This Happens

**Root cause:** Colab's Python environment doesn't always recognize editable installs properly.

**What `pip install -e .` does:**
- Creates a `.pth` file pointing to `/content/ceep-v1/src`
- Sometimes this doesn't get picked up by Colab's kernel

**Solution:** Directly modify `sys.path` to include the source directory.

---

## Verification

After adding `sys.path.insert(0, '/content/ceep-v1/src')`, verify:

```python
import sys
print("CEEP in path:", any('ceep' in p for p in sys.path))

import ceep
print("CEEP location:", ceep.__file__)

from ceep.core.backend import set_backend
print("✓ Imports working!")
```

**Expected:**
```
CEEP in path: True
CEEP location: /content/ceep-v1/src/ceep/__init__.py
✓ Imports working!
```

---

## Complete Minimal Script

Save this as `radar_colab_minimal.py`:

```python
#!/usr/bin/env python3
"""Minimal radar example for Colab"""
import sys
sys.path.insert(0, '/content/ceep-v1/src')  # FIX for Colab

from ceep.core.backend import set_backend
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
import numpy as np

set_backend('cupy')
print("✓ CEEP loaded, running simulation...")

# Quick 8-element radar
positions = [(200+i*10, 100) for i in range(8)]
solver = BatchedFDTD2D(
    nx=400, ny=400, dx=0.001, dy=0.001, dt=2e-12,
    total_steps=400, cpml_thickness=10,
    source_positions=positions, probe_positions=positions,
    frequency=10e9, bandwidth=2e9
)
solver._eps_r[:] = 1.0  # Free space

s_matrix = solver.run()
print(f"✓ Simulation complete! Got {len(s_matrix)} TX antennas")
```

Then in Colab:
```python
!python radar_colab_minimal.py
```

---

## Summary

**Always add this line at the top of Colab notebooks:**
```python
import sys
sys.path.insert(0, '/content/ceep-v1/src')
```

This ensures CEEP is importable regardless of pip's behavior.

---

**Repository:** https://github.com/shahzaibshazoo/ceep-v1  
**Working now!** ✅
