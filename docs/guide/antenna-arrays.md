# Antenna Arrays

NeuroWave provides 7 antenna array geometries for different imaging and communication scenarios.

## Circular Array (Head Imaging)

The standard configuration for microwave brain imaging:

```python
from neurowave.antennas.arrays import CircularArray

array = CircularArray(
    num_antennas=16,
    radius_mm=120,
    center=(150, 150),
    dx=1e-3,
    antenna_type='monopole',
    polarization='vertical'
)

positions = array.get_antenna_positions()
```

## Planar Array (Breast Imaging)

Rectangular grid above the imaging region:

```python
from neurowave.antennas.arrays import PlanarArray

array = PlanarArray(
    nx_antennas=8,
    ny_antennas=8,
    spacing_mm=20,
    corner=(50, 50),
    dx=1e-3,
    antenna_type='patch'
)
```

## Uniform Linear Array (ULA)

For beamforming and radar:

```python
from neurowave.antennas.arrays import UniformLinearArray

ula = UniformLinearArray(
    num_elements=8,
    spacing_mm=50,       # lambda/2 at 3 GHz
    orientation='horizontal',
    start_position=(50, 100),
    dx=1e-3,
)

# Compute steering vector for beamforming
sv = ula.compute_steering_vector(frequency=3e9, angle_deg=30)
```

## Uniform Rectangular Array (URA)

For massive MIMO and 2D beamforming:

```python
from neurowave.antennas.arrays import UniformRectangularArray

ura = UniformRectangularArray(
    nx_elements=4, ny_elements=4,
    spacing_x_mm=50, spacing_y_mm=50,
    corner=(50, 50),
    dx=1e-3,
)

# 2D steering
sv = ura.compute_2d_steering_vector(frequency=3e9, azimuth_deg=30, elevation_deg=15)
```

## L-Shaped Array

For 2D direction-of-arrival estimation:

```python
from neurowave.antennas.arrays import LShapedArray

l_array = LShapedArray(
    num_x_elements=8,
    num_y_elements=8,
    spacing_mm=50,
    corner=(100, 100),
    dx=1e-3,
)
```

## Random/Sparse Array

For compressed sensing and super-resolution:

```python
from neurowave.antennas.arrays import RandomArray

array = RandomArray(
    num_elements=16,
    region=(50, 250, 50, 250),
    dx=1e-3,
    min_spacing_mm=20,
    seed=42  # Reproducible
)
```

## Conformal Array

Body-fitted arrays for wearable systems:

```python
from neurowave.antennas.arrays import ConformalArray
import numpy as np

# Elliptical contour
angles = np.linspace(0, 2*np.pi, 200)
contour = [(int(150 + 80*np.cos(a)), int(150 + 60*np.sin(a)))
           for a in angles]

array = ConformalArray(
    contour_points=contour,
    num_antennas=12,
    dx=1e-3,
    offset_mm=5.0
)
```

## Factory Function

Quick creation of standard imaging configurations:

```python
from neurowave.antennas.arrays import create_imaging_array

# Standard 16-element head imaging
array = create_imaging_array('circular', 'head', grid_shape=(300, 300), dx=1e-3)

# 12-element breast imaging
array = create_imaging_array('circular', 'breast', grid_shape=(200, 200), dx=1e-3)
```

## Multistatic Pairs

Get all TX-RX combinations:

```python
pairs = array.get_transmit_receive_pairs()
# [(0,0), (0,1), (0,2), ..., (15,15)] — 256 pairs for 16 antennas
```

## Array Factor

Compute radiation pattern:

```python
import numpy as np

angles = np.linspace(0, 2*np.pi, 360)
af = array.compute_array_factor(frequency=1e9, angles=angles)

import matplotlib.pyplot as plt
plt.polar(angles, np.abs(af))
plt.title('Array Factor')
plt.show()
```
