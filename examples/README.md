# CEEP Examples

This directory contains example scripts demonstrating CEEP's GPU-accelerated FDTD capabilities.

## Available Examples

### Biomedical Imaging

| Example | Description | Performance | Status |
|---------|-------------|-------------|--------|
| `biomedical_hemorrhage_detection.py` | Brain hemorrhage detection with 16-element circular array | 3.3s per sample | ✅ |
| `basic_2d_fdtd.py` | Simple 2D free-space propagation | <1s | ✅ |
| `dielectric_slab.py` | Wave interaction with dielectric materials | <1s | ✅ |
| `waveguide.py` | Waveguide mode propagation | <2s | ✅ |

### Radar & Beamforming ⭐ **NEW!**

| Example | Description | Performance | Status |
|---------|-------------|-------------|--------|
| `radar_2d_ula_beamforming.py` | 2D radar with ULA + Conventional/Capon/MUSIC beamforming | 5s (16 TX) | ✅ |
| `radar_3d_ula_farfield.py` | 3D far-field radar with azimuth/elevation estimation | 15s (8 TX) | ✅ |

See **[README_RADAR.md](README_RADAR.md)** for detailed radar documentation.

## Quick Start

### 1. Install CEEP with GPU support
```bash
pip install -e ".[gpu]"
```

### 2. Run biomedical example (fastest!)
```bash
python examples/biomedical_hemorrhage_detection.py
```

**Expected output:**
```
✓ Simulated 16×16 = 256 channels in 3.3 seconds
✓ GPU speedup: 22× vs MEEP
```

### 3. Run radar beamforming example
```bash
python examples/radar_2d_ula_beamforming.py
```

**Expected output:**
```
✓ Batched FDTD Complete! (5.0s for 16 TX)
✓ Beamforming algorithms: Conventional, Capon, MUSIC
✓ Results saved to radar_2d_ula_beamforming.png
```

## GPU vs CPU Performance

| Example | GPU (T4) | CPU | Speedup |
|---------|----------|-----|---------|
| Hemorrhage detection | 3.3s | 75s | 22× |
| 2D radar (16 TX) | 5s | 120s | 24× |
| 3D radar (8 TX) | 15s | 300s | 20× |

## Customization

### Change backend (GPU/CPU)
```python
from ceep.core.backend import set_backend

set_backend('cupy')   # GPU (requires CuPy)
set_backend('numpy')  # CPU
```

### Modify simulation parameters
```python
# Larger domain
DOMAIN_SIZE = 10.0  # meters

# Higher resolution
GRID_RESOLUTION = 60  # points per wavelength

# More timesteps
TOTAL_STEPS = 2000
```

## Documentation

- **Biomedical examples**: See code comments in `biomedical_*.py`
- **Radar examples**: See [README_RADAR.md](README_RADAR.md) for comprehensive guide
- **API reference**: https://ceep.readthedocs.io

## Troubleshooting

### GPU not detected
```bash
# Check CUDA installation
nvidia-smi

# Install CuPy
pip install cupy-cuda12x
```

### Out of memory
Reduce domain size or grid resolution:
```python
DOMAIN_SIZE = 4.0      # Smaller
GRID_RESOLUTION = 30   # Coarser
```

### Slow performance
Make sure GPU backend is active:
```python
from ceep.core.backend import print_backend_info
print_backend_info()  # Should show "cupy" and GPU name
```

## Contributing

Have an interesting example? Submit a PR!

Guidelines:
- Include docstring with description and expected runtime
- Add GPU/CPU performance comparison
- Follow existing code style
- Test on both NumPy and CuPy backends
