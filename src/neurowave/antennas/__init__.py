"""
Antenna Array Configurations
=============================

Comprehensive antenna array geometries for:
- Biomedical microwave imaging (circular, conformal)
- MIMO communications (ULA, URA, L-shaped)
- Radar and beamforming (phased arrays)
- General EM experiments (random, sparse arrays)
"""

from neurowave.antennas.arrays import (
    AntennaElement,
    CircularArray,
    PlanarArray,
    ConformalArray,
    UniformLinearArray,
    UniformRectangularArray,
    LShapedArray,
    RandomArray,
    visualize_array_geometry,
    create_imaging_array
)

__all__ = [
    'AntennaElement',
    'CircularArray',
    'PlanarArray',
    'ConformalArray',
    'UniformLinearArray',
    'UniformRectangularArray',
    'LShapedArray',
    'RandomArray',
    'visualize_array_geometry',
    'create_imaging_array'
]
