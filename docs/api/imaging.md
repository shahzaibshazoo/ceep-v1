# Imaging API

## DASBeamformer

Delay-And-Sum beamforming for microwave image reconstruction.

```python
from neurowave.imaging.beamforming import DASBeamformer, ImagingRegion
```

### ImagingRegion

Defines the pixel grid for reconstruction:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x_range` | tuple | (x_min, x_max) in grid cells |
| `y_range` | tuple | (y_min, y_max) in grid cells |
| `pixel_size` | int | Output resolution (cells per pixel) |
| `z_slice` | int | Z-index for 3D (optional) |

### DASBeamformer

| Parameter | Type | Description |
|-----------|------|-------------|
| `antenna_positions` | list | (x, y) positions of all antennas |
| `imaging_region` | ImagingRegion | Target region |
| `dx` | float | Grid spacing (meters) |
| `c_background` | float | Wave speed in background medium (m/s) |

### Methods

```python
image = beamformer.reconstruct(s_matrix, dt)
```

- `s_matrix` — Dict: `s_matrix[tx][rx] = signal_array`
- `dt` — Timestep (seconds)
- Returns: 2D numpy array (imaging intensity)

### Example

```python
import numpy as np
from neurowave.imaging.beamforming import DASBeamformer, ImagingRegion

region = ImagingRegion(x_range=(60, 240), y_range=(60, 240), pixel_size=2)

beamformer = DASBeamformer(
    antenna_positions=positions,
    imaging_region=region,
    dx=1e-3,
    c_background=3e8 / np.sqrt(40)
)

image = beamformer.reconstruct(s_matrix, dt=solver.dt)
```

---

## Head Phantoms

```python
from neurowave.phantoms.head_models import (
    SimpleHeadPhantom,
    DetailedBrainPhantom,
)
```

### SimpleHeadPhantom

3-layer model: skin, skull, brain.

| Parameter | Type | Description |
|-----------|------|-------------|
| `center` | tuple | Center position |
| `head_radius_mm` | float | Outer radius |
| `dx` | float | Grid spacing |
| `frequency` | float | Operating frequency |

### DetailedBrainPhantom

Multilayer with grey/white matter, CSF, and hemorrhage support.

```python
phantom = DetailedBrainPhantom(center=(150,150), head_radius_mm=80, dx=1e-3, frequency=1e9)
phantom.add_hemorrhage(position=(170, 160), radius_mm=10, hemorrhage_type='intracerebral')
```
