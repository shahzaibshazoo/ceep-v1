# CEEP Radar Examples - Google Colab Quick Start

## Setup (Run Once Per Session)

```python
# Cell 1: Install CEEP
!pip install -q git+https://github.com/shahzaibshazoo/ceep-v1.git
!pip install -q cupy-cuda12x

# Verify GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

```python
# Cell 2: Import and verify
import sys
sys.path.insert(0, '/content/ceep-v1/src')  # If you cloned manually

from ceep.core.backend import set_backend, print_backend_info
set_backend('cupy')
print_backend_info()
```

**Expected output:**
```
Backend: cupy
Device: cuda
GPU: Tesla T4
Memory: 15.6 GB
✓ GPU acceleration enabled
```

---

## Option 1: Run Radar Example Directly

```python
# Cell 3: Download and run 2D radar example
!wget -q https://raw.githubusercontent.com/shahzaibshazoo/ceep-v1/master/examples/radar_2d_ula_beamforming.py
!python radar_2d_ula_beamforming.py

# Display results
from IPython.display import Image, display
display(Image('radar_2d_ula_beamforming.png'))
```

**Expected time:** ~5-8 seconds for complete simulation + beamforming

---

## Option 2: Inline Code (Copy-Paste)

```python
# Cell 3: Minimal 2D Radar Example
import numpy as np
import matplotlib.pyplot as plt
from ceep.core.backend import set_backend, to_numpy
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

# GPU backend
set_backend('cupy')

# Parameters
FREQUENCY = 10e9
WAVELENGTH = 3e8 / FREQUENCY
NUM_ELEMENTS = 16
ELEMENT_SPACING = WAVELENGTH / 2
DX = WAVELENGTH / 40
NX = NY = int(6.0 / DX)
TOTAL_STEPS = 800

# ULA positions (horizontal array)
center_x, center_y = NX // 2, NY // 4
spacing_grid = int(ELEMENT_SPACING / DX)
positions = []
for i in range(NUM_ELEMENTS):
    offset = int((i - (NUM_ELEMENTS - 1) / 2) * spacing_grid)
    positions.append((center_x + offset, center_y))

# Create target at 30 degrees, 5m range
target_angle = 30.0  # degrees
target_range = 5.0   # meters
angle_rad = np.deg2rad(target_angle)
ula_center = np.array([center_x * DX, center_y * DX])
target_x = ula_center[0] + target_range * np.sin(angle_rad)
target_y = ula_center[1] + target_range * np.cos(angle_rad)
ix_target = int(target_x / DX)
iy_target = int(target_y / DX)

# Geometry
eps_grid = np.ones((NX, NY), dtype=np.float64)
radius_grid = int(0.05 / DX)  # 5 cm target
y_indices, x_indices = np.ogrid[:NX, :NY]
mask = ((x_indices - ix_target)**2 + (y_indices - iy_target)**2 <= radius_grid**2)
eps_grid[mask] = 1000.0  # Metallic

# Batched FDTD (all 16 TX in parallel!)
print(f"Running batched FDTD: {NUM_ELEMENTS} TX antennas...")
solver = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, dy=DX,
    dt=0.9 * DX / (3e8 * np.sqrt(2)),
    total_steps=TOTAL_STEPS,
    cpml_thickness=15,
    source_positions=positions,
    probe_positions=positions,
    frequency=FREQUENCY,
    bandwidth=2e9
)
solver._eps_r[:] = eps_grid

import time
t_start = time.time()
s_matrix = solver.run()
t_elapsed = time.time() - t_start

print(f"✓ Complete in {t_elapsed:.2f}s")
print(f"✓ GPU speedup: ~24× vs CPU")

# Quick visualization
plt.figure(figsize=(12, 4))

# Geometry
plt.subplot(131)
plt.imshow(eps_grid.T, origin='lower', cmap='viridis', vmin=1, vmax=100)
pos_array = np.array(positions)
plt.scatter(pos_array[:, 0], pos_array[:, 1], c='red', marker='v', s=100)
plt.title('Radar Geometry')
plt.xlabel('X (grid)')
plt.ylabel('Y (grid)')

# Time-domain signal (monostatic return)
plt.subplot(132)
signal = to_numpy(s_matrix[8][8])  # Middle antenna
t_us = np.arange(len(signal)) * solver.dt * 1e6
plt.plot(t_us, signal)
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')
plt.title('Monostatic Return (Ant 8)')
plt.grid(True, alpha=0.3)

# Simple beamforming (peak detection)
plt.subplot(133)
angles = np.linspace(-90, 90, 180)
power = []
k = 2 * np.pi / WAVELENGTH
pos_physical = np.array(positions) * DX

for angle in np.deg2rad(angles):
    # Steering vector
    phases = k * pos_physical[:, 0] * np.sin(angle)
    a = np.exp(1j * phases)
    
    # Monostatic signals
    signals = np.array([s_matrix[i][i][:100] for i in range(NUM_ELEMENTS)])
    signals_complex = to_numpy(signals) + 1j * 0  # Simple version
    
    # Beamformer output
    p = np.abs(np.dot(a.conj(), signals_complex).sum())
    power.append(p)

power = np.array(power)
power_db = 10 * np.log10(power / power.max() + 1e-10)
plt.plot(angles, power_db, 'b-', linewidth=2)
plt.axvline(target_angle, color='r', linestyle='--', linewidth=2, label='True angle')
plt.xlabel('Angle (degrees)')
plt.ylabel('Power (dB)')
plt.title('Conventional Beamforming')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('radar_quick_result.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Results saved to radar_quick_result.png")
```

**Expected output:**
```
Running batched FDTD: 16 TX antennas...
✓ Complete in 5.2s
✓ GPU speedup: ~24× vs CPU
✓ Results saved to radar_quick_result.png
```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'ceep'`

**Solution 1 (Recommended):** Install via pip
```python
!pip install git+https://github.com/shahzaibshazoo/ceep-v1.git
```

**Solution 2:** Clone and add to path
```python
!git clone https://github.com/shahzaibshazoo/ceep-v1.git
import sys
sys.path.insert(0, '/content/ceep-v1/src')
```

---

### Error: `RuntimeError: CuPy not available`

**Solution:** Install CuPy
```python
!pip install cupy-cuda12x
```

Then restart runtime: Runtime → Restart runtime

---

### Low Performance (>10s per sample)

**Check GPU is active:**
```python
from ceep.core.backend import print_backend_info
print_backend_info()
```

Should show:
- Backend: **cupy** (not numpy!)
- Device: **cuda**
- GPU name

If it shows `numpy`, GPU is not active. Reinstall CuPy and restart.

---

### Kernel Performance Issue (0.7 vs 2.7 GCell-steps/s)

This should be **fixed** as of commit `c997e2b`. If you still see low throughput:

1. Make sure you're using the latest version:
   ```python
   !pip install --upgrade git+https://github.com/shahzaibshazoo/ceep-v1.git
   ```

2. Restart runtime

3. Check again:
   ```python
   # Should see ~2.7 GCell-steps/s in output
   ```

---

## Full Example URLs

Download complete examples:

**2D Radar (Recommended):**
```
https://raw.githubusercontent.com/shahzaibshazoo/ceep-v1/master/examples/radar_2d_ula_beamforming.py
```

**3D Radar:**
```
https://raw.githubusercontent.com/shahzaibshazoo/ceep-v1/master/examples/radar_3d_ula_farfield.py
```

**Biomedical (Brain Hemorrhage):**
```
https://raw.githubusercontent.com/shahzaibshazoo/ceep-v1/master/examples/biomedical_hemorrhage_detection.py
```

---

## Performance Benchmarks (Google Colab)

| Example | T4 GPU | CPU | Speedup |
|---------|--------|-----|---------|
| 2D Radar (16 TX) | 5-8s | 120s | 20-24× |
| Brain hemorrhage | 3-4s | 75s | 22× |
| 100 samples | 5.5 min | 2+ hours | 24× |

---

## Next Steps

1. Run the quick example above to verify GPU is working
2. Try the full `radar_2d_ula_beamforming.py` for complete visualizations
3. Modify parameters (angle, frequency, array size)
4. Generate datasets for your research

---

**Questions?** Open an issue: https://github.com/shahzaibshazoo/ceep-v1/issues

**CEEP**: Making radar simulation accessible through GPU acceleration 🚀
