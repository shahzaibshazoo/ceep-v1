# Antennas API

```python
from neurowave.antennas.arrays import (
    CircularArray,
    PlanarArray,
    UniformLinearArray,
    UniformRectangularArray,
    LShapedArray,
    RandomArray,
    ConformalArray,
    create_imaging_array,
)
```

## Common Interface

All array classes provide:

- `array.get_antenna_positions()` — Returns list of `(x, y)` tuples
- `array.elements` — List of `AntennaElement` objects

## CircularArray

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_antennas` | int | — | Number of elements |
| `radius_mm` | float | — | Ring radius (mm) |
| `center` | tuple | — | Center (x, y) in grid cells |
| `dx` | float | — | Grid spacing (m) |
| `antenna_type` | str | 'monopole' | Element type |
| `polarization` | str | 'vertical' | 'vertical' or 'horizontal' |

Extra methods:
- `array.compute_array_factor(frequency, angles, weights=None)` — Radiation pattern
- `array.get_transmit_receive_pairs()` — All (TX, RX) index pairs

## UniformLinearArray

Extra methods:
- `ula.compute_steering_vector(frequency, angle_deg)` — Beamforming weights

## UniformRectangularArray

Extra methods:
- `ura.compute_2d_steering_vector(frequency, azimuth_deg, elevation_deg)` — 2D steering

## create_imaging_array

Factory for standard clinical configurations:

```python
array = create_imaging_array(
    array_type='circular',     # 'circular' or 'planar'
    imaging_target='head',     # 'head' or 'breast'
    grid_shape=(300, 300),
    dx=1e-3
)
```
