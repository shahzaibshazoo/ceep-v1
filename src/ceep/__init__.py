"""
NeuroWave — GPU-Accelerated FDTD Electromagnetic Simulation
============================================================

A GPU-native Maxwell equation solver for biomedical microwave imaging.

Quick Start::

    from ceep.core.backend import set_backend
    from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

    set_backend('cupy')  # Use GPU
    solver = BatchedFDTD2D(nx=300, ny=300, dx=1e-3, ...)
    s_matrix = solver.run()

Modules
-------
- ``core``        : Backend, grid, configuration
- ``solvers``     : FDTD2D, FDTD3D, BatchedFDTD2D
- ``boundaries``  : CPML, Mur ABC
- ``sources``     : Gaussian, modulated Gaussian, plane wave
- ``materials``   : Dispersive models, tissue database
- ``antennas``    : Array geometries (circular, planar, ULA, etc.)
- ``phantoms``    : Head models with hemorrhage
- ``imaging``     : DAS beamforming reconstruction
- ``cuda``        : Fused CUDA kernels
"""

__version__ = "0.1.0"
__author__ = "Shahzaib Elbert"
__license__ = "MIT"
