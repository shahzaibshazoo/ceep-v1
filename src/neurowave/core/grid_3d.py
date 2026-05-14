"""
3D Yee Grid implementation for FDTD simulation.

The 3D Yee grid places all 6 electromagnetic field components at staggered
positions in space.

Field component positions on the 3D Yee cell (i, j, k):
    Ex at (i+1/2, j, k)
    Ey at (i, j+1/2, k)
    Ez at (i, j, k+1/2)
    Hx at (i, j+1/2, k+1/2)
    Hy at (i+1/2, j, k+1/2)
    Hz at (i+1/2, j+1/2, k)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from neurowave.core.config import SimulationConfig, SimulationMode
from neurowave.core.constants import EPS_0, MU_0
from neurowave.core import backend as xpb


@dataclass
class Grid3D:
    """3D Yee grid with staggered electric and magnetic field arrays."""

    grid_config: "GridConfig"
    dt: float = 1e-12  # Default timestep

    # 3D fields
    ex: npt.NDArray[np.float64] = field(init=False, repr=False)
    ey: npt.NDArray[np.float64] = field(init=False, repr=False)
    ez: npt.NDArray[np.float64] = field(init=False, repr=False)
    hx: npt.NDArray[np.float64] = field(init=False, repr=False)
    hy: npt.NDArray[np.float64] = field(init=False, repr=False)
    hz: npt.NDArray[np.float64] = field(init=False, repr=False)

    # Material properties
    eps_r: npt.NDArray[np.float64] = field(init=False, repr=False)
    mu_r: npt.NDArray[np.float64] = field(init=False, repr=False)
    sigma_e: npt.NDArray[np.float64] = field(init=False, repr=False)
    sigma_m: npt.NDArray[np.float64] = field(init=False, repr=False)

    # Update coefficients
    _ca: npt.NDArray[np.float64] = field(init=False, repr=False)
    _cb: npt.NDArray[np.float64] = field(init=False, repr=False)
    _da: npt.NDArray[np.float64] = field(init=False, repr=False)
    _db: npt.NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Allocate field arrays and initialize material properties."""
        if not self.grid_config.is_3d:
            raise ValueError("Grid3D requires nz > 1 in GridConfig.")

        nx = self.grid_config.nx
        ny = self.grid_config.ny
        nz = self.grid_config.nz
        shape = (nx, ny, nz)

        # Initialize field arrays (backend-aware: GPU if cupy active)
        self.ex = xpb.zeros(shape)
        self.ey = xpb.zeros(shape)
        self.ez = xpb.zeros(shape)
        self.hx = xpb.zeros(shape)
        self.hy = xpb.zeros(shape)
        self.hz = xpb.zeros(shape)

        # Material properties
        self.eps_r = xpb.ones(shape)
        self.mu_r = xpb.ones(shape)
        self.sigma_e = xpb.zeros(shape)
        self.sigma_m = xpb.zeros(shape)

        self._precompute_coefficients()

    def _precompute_coefficients(self) -> None:
        """Precompute FDTD update coefficients."""
        dt = self.dt
        eps = self.eps_r * EPS_0
        mu = self.mu_r * MU_0

        sigma_dt_2eps = self.sigma_e * dt / (2.0 * eps)
        self._ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
        self._cb = (dt / eps) / (1.0 + sigma_dt_2eps)

        sigma_m_dt_2mu = self.sigma_m * dt / (2.0 * mu)
        self._da = (1.0 - sigma_m_dt_2mu) / (1.0 + sigma_m_dt_2mu)
        self._db = (dt / mu) / (1.0 + sigma_m_dt_2mu)

    def set_material_region(
        self,
        x_start: int, x_end: int,
        y_start: int, y_end: int,
        z_start: int, z_end: int,
        eps_r: float = 1.0,
        mu_r: float = 1.0,
        sigma_e: float = 0.0,
        sigma_m: float = 0.0,
    ) -> None:
        """Set material properties in a 3D rectangular region."""
        region = (slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end))
        self.eps_r[region] = eps_r
        self.mu_r[region] = mu_r
        self.sigma_e[region] = sigma_e
        self.sigma_m[region] = sigma_m
        self._precompute_coefficients()

    @property
    def nx(self) -> int:
        """Grid size in x-direction."""
        return self.grid_config.nx

    @property
    def ny(self) -> int:
        """Grid size in y-direction."""
        return self.grid_config.ny

    @property
    def nz(self) -> int:
        """Grid size in z-direction."""
        return self.grid_config.nz

    @property
    def dx(self) -> float:
        """Grid spacing in x-direction (meters)."""
        return self.grid_config.dx

    @property
    def dy(self) -> float:
        """Grid spacing in y-direction (meters)."""
        return self.grid_config.dy

    @property
    def dz(self) -> float:
        """Grid spacing in z-direction (meters)."""
        return self.grid_config.dz
