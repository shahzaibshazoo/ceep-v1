"""
ceep.core — Core Data Structures and Simulation Engine
======================================================

This module provides the fundamental building blocks for electromagnetic
simulation:

- **Grid**: Yee grid implementations (2D and 3D) with staggered field positions
- **Fields**: Electric and magnetic field arrays with proper memory layout
- **Config**: Simulation configuration (grid size, resolution, timestep)
- **Engine**: Main simulation loop orchestrator
- **Constants**: Physical constants (c, ε₀, μ₀, etc.)
- **Backend**: NumPy/CuPy abstraction for GPU/CPU switching

Mathematical Foundation
-----------------------
The Yee grid places E and H field components at staggered positions in both
space and time, enabling second-order accurate central differences for the
curl operations in Maxwell's equations.

For 2D TMz mode:
    Ez is at integer grid points (i, j)
    Hx is at (i, j+1/2)
    Hy is at (i+1/2, j)

References
----------
.. [1] K. S. Yee, "Numerical solution of initial boundary value problems
       involving Maxwell's equations in isotropic media," IEEE Trans.
       Antennas Propag., vol. 14, no. 3, pp. 302-307, 1966.
.. [2] A. Taflove and S. C. Hagness, "Computational Electrodynamics:
       The Finite-Difference Time-Domain Method," 3rd ed., Artech House, 2005.
"""

# Import core classes
from .grid import Grid2D
from .grid_3d import Grid3D
from .config import GridConfig, SimulationConfig, SimulationMode, Backend
from .backend import set_backend, get_backend, is_backend_available
from .constants import PHYSICAL_CONSTANTS

# Legacy aliases for backward compatibility
Config2D = GridConfig
Config3D = GridConfig

__all__ = [
    'Grid2D',
    'Grid3D',
    'GridConfig',
    'SimulationConfig',
    'SimulationMode',
    'Backend',
    'Config2D',  # Legacy
    'Config3D',  # Legacy
    'set_backend',
    'get_backend',
    'is_backend_available',
    'PHYSICAL_CONSTANTS',
]
