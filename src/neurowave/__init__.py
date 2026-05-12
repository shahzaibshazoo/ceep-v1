"""
NeuroWave — GPU-Accelerated Electromagnetic Simulation Framework
================================================================

A modern, GPU-native alternative to Meep/PyMeep for computational
electromagnetics, biomedical microwave imaging, and AI-integrated
physics simulation.

Core Modules
------------
- ``core``       : Grid, field arrays, simulation engine
- ``cuda``       : CUDA kernels, GPU memory management
- ``materials``  : Dielectric models, tissue libraries
- ``boundaries`` : ABC, PML, periodic boundaries
- ``sources``    : Gaussian, sinusoidal, custom sources
- ``solvers``    : FDTD solvers, frequency-domain extractors
- ``visualization`` : Real-time & post-processing visualization
- ``ai``         : PyTorch integration, differentiable physics

Philosophy
----------
Correctness → Reproducibility → Performance → Elegance

License
-------
MIT License. See LICENSE file for details.
"""

__version__ = "0.1.0-dev"
__author__ = "NeuroWave Contributors"
__license__ = "MIT"

# Lazy imports will be added as modules are implemented
