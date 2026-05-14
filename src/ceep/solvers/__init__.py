"""
ceep.solvers — FDTD Solvers and Analysis Tools
===================================================

This module provides the main simulation solvers:

- **FDTD2D**: 2D Finite-Difference Time-Domain solver
- **BatchedFDTD2D**: Batched 2D FDTD for multistatic arrays (GPU-accelerated)
- **FDTD3D**: 3D Finite-Difference Time-Domain solver
- **DFTMonitor**: Frequency-domain field extraction via running DFT
- **SParameters**: S-parameter extraction from port fields
- **NearToFar**: Near-to-far field transformation

FDTD Algorithm
--------------
The FDTD method solves Maxwell's curl equations by:
1. Discretizing space on a Yee grid (staggered E/H positions)
2. Discretizing time with a leapfrog scheme (E at n, H at n+1/2)
3. Applying explicit update equations at each timestep
4. Enforcing boundary conditions after each update

Time-stepping (leapfrog):
    H^{n+1/2} = H^{n-1/2} + (Δt/μ) ∇×E^n
    E^{n+1}   = E^n       + (Δt/ε) ∇×H^{n+1/2} − (σΔt/ε) E^n

Computational Complexity
------------------------
- Time:  O(N × T) where N = grid cells, T = timesteps
- Space: O(N) for field storage
- GPU:   O(T) kernel launches, each O(N/P) with P GPU threads
"""

from ceep.solvers.fdtd_2d import FDTD2D
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
from ceep.solvers.fdtd_3d import FDTD3D

__all__ = [
    'FDTD2D',
    'BatchedFDTD2D',
    'FDTD3D',
]
