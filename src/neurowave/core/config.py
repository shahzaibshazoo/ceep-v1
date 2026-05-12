"""
Simulation configuration and parameter management.

This module defines the core configuration dataclasses used to set up
and parameterize FDTD simulations. All physical quantities are in SI units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

from neurowave.core.constants import (
    C_0,
    DEFAULT_COURANT_2D,
    DEFAULT_COURANT_3D,
    cfl_timestep_2d,
    cfl_timestep_3d,
)


class SimulationMode(Enum):
    """Supported FDTD simulation modes.

    Attributes
    ----------
    TMZ : Transverse Magnetic (Ez, Hx, Hy) — 2D
    TEZ : Transverse Electric (Hz, Ex, Ey) — 2D
    FULL_3D : Full 3D vectorial simulation
    """

    TMZ = auto()
    TEZ = auto()
    FULL_3D = auto()


class Backend(Enum):
    """Computation backend selection.

    Attributes
    ----------
    NUMPY : Pure NumPy (CPU reference implementation)
    CUPY : CuPy GPU arrays
    CUDA : Raw CUDA kernels via pybind11
    TORCH : PyTorch tensors (for differentiable simulation)
    """

    NUMPY = auto()
    CUPY = auto()
    CUDA = auto()
    TORCH = auto()


@dataclass
class GridConfig:
    """Configuration for the simulation grid.

    Parameters
    ----------
    nx : int
        Number of grid cells in x-direction.
    ny : int
        Number of grid cells in y-direction.
    nz : int, optional
        Number of grid cells in z-direction (3D only). Default is 1.
    dx : float
        Grid spacing in x-direction [m].
    dy : float
        Grid spacing in y-direction [m].
    dz : float, optional
        Grid spacing in z-direction [m] (3D only). Default matches dx.

    Notes
    -----
    For accurate simulation, the grid spacing should satisfy:
        Δx ≤ λ_min / 10

    where λ_min is the minimum wavelength of interest. For dispersive
    or high-contrast materials, finer resolution may be needed (λ/20 or more).
    """

    nx: int
    ny: int
    nz: int = 1
    dx: float = 1e-3  # 1 mm default
    dy: float = 1e-3
    dz: float = 1e-3

    def __post_init__(self) -> None:
        """Validate grid configuration."""
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("Grid dimensions must be positive integers.")
        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ValueError("Grid spacings must be positive.")

    @property
    def is_3d(self) -> bool:
        """Check if grid is 3D."""
        return self.nz > 1

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return grid shape as tuple."""
        if self.is_3d:
            return (self.nx, self.ny, self.nz)
        return (self.nx, self.ny)

    @property
    def total_cells(self) -> int:
        """Total number of grid cells."""
        return self.nx * self.ny * self.nz

    @property
    def physical_size(self) -> Tuple[float, ...]:
        """Physical size of the simulation domain [m]."""
        if self.is_3d:
            return (self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)
        return (self.nx * self.dx, self.ny * self.dy)


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    Parameters
    ----------
    grid : GridConfig
        Grid configuration.
    mode : SimulationMode
        Simulation mode (TMz, TEz, or Full 3D).
    courant : float
        Courant number for CFL stability.
    total_time : float, optional
        Total simulation time [s]. Either this or total_steps must be given.
    total_steps : int, optional
        Total number of timesteps. Either this or total_time must be given.
    backend : Backend
        Computation backend to use.

    Notes
    -----
    The timestep is automatically calculated from the CFL condition:
        Δt = S / (c₀ · √(1/Δx² + 1/Δy²))

    where S is the Courant number. For stability:
        - 2D: S ≤ 1/√2 ≈ 0.7071
        - 3D: S ≤ 1/√3 ≈ 0.5774
    """

    grid: GridConfig
    mode: SimulationMode = SimulationMode.TMZ
    courant: float = DEFAULT_COURANT_2D
    total_time: Optional[float] = None
    total_steps: Optional[int] = None
    backend: Backend = Backend.NUMPY

    def __post_init__(self) -> None:
        """Validate configuration and compute derived parameters."""
        # Validate Courant number
        if self.grid.is_3d:
            max_courant = 1.0 / math.sqrt(3.0)
        else:
            max_courant = 1.0 / math.sqrt(2.0)

        if self.courant <= 0 or self.courant > max_courant:
            raise ValueError(
                f"Courant number must be in (0, {max_courant:.4f}] "
                f"for {'3D' if self.grid.is_3d else '2D'} simulation. "
                f"Got {self.courant}."
            )

        # Require either total_time or total_steps
        if self.total_time is None and self.total_steps is None:
            raise ValueError("Must specify either total_time or total_steps.")

    @property
    def dt(self) -> float:
        """Timestep computed from CFL condition [s]."""
        if self.grid.is_3d:
            return cfl_timestep_3d(
                self.grid.dx, self.grid.dy, self.grid.dz, self.courant
            )
        return cfl_timestep_2d(self.grid.dx, self.grid.dy, self.courant)

    @property
    def num_steps(self) -> int:
        """Total number of simulation timesteps."""
        if self.total_steps is not None:
            return self.total_steps
        assert self.total_time is not None
        return int(math.ceil(self.total_time / self.dt))

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            "=" * 60,
            "NeuroWave Simulation Configuration",
            "=" * 60,
            f"Mode:           {self.mode.name}",
            f"Backend:        {self.backend.name}",
            f"Grid:           {self.grid.nx} × {self.grid.ny}"
            + (f" × {self.grid.nz}" if self.grid.is_3d else ""),
            f"Grid spacing:   Δx={self.grid.dx:.2e} m, Δy={self.grid.dy:.2e} m"
            + (f", Δz={self.grid.dz:.2e} m" if self.grid.is_3d else ""),
            f"Physical size:  {' × '.join(f'{s:.4f} m' for s in self.grid.physical_size)}",
            f"Total cells:    {self.grid.total_cells:,}",
            f"Courant number: {self.courant:.4f}",
            f"Timestep:       {self.dt:.4e} s",
            f"Total steps:    {self.num_steps:,}",
            f"Total time:     {self.num_steps * self.dt:.4e} s",
            "=" * 60,
        ]
        return "\n".join(lines)
