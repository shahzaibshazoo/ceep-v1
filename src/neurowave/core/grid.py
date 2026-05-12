"""
2D Yee Grid implementation for FDTD simulation.

The Yee grid is the fundamental spatial discretization for FDTD. It places
electric and magnetic field components at staggered positions, enabling
second-order accurate central differences for Maxwell's curl equations.

TMz Mode (Ez, Hx, Hy):
-----------------------
Field component positions on the Yee cell:

    (i,j+1) ---- Hy(i,j+1) ---- (i+1,j+1)
       |                            |
       |                            |
    Hx(i,j)      Ez(i,j)        Hx(i+1,j)
       |                            |
       |                            |
    (i,j) ------ Hy(i,j) ------- (i+1,j)

    Ez is at integer grid points:      (i, j)
    Hx is at half-integer y positions: (i, j+1/2)
    Hy is at half-integer x positions: (i+1/2, j)

TEz Mode (Hz, Ex, Ey):
-----------------------
    Hz is at (i+1/2, j+1/2)
    Ex is at (i+1/2, j)
    Ey is at (i, j+1/2)

Memory Layout
-------------
All field arrays use the same shape (nx, ny) for simplicity and future
GPU compatibility (uniform memory access patterns). The staggering is
handled by the update equation indexing, not by different array sizes.

References
----------
.. [1] K. S. Yee, "Numerical solution of initial boundary value problems
       involving Maxwell's equations in isotropic media," IEEE Trans.
       Antennas Propag., vol. 14, no. 3, pp. 302-307, 1966.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from neurowave.core.config import GridConfig, SimulationConfig, SimulationMode
from neurowave.core.constants import EPS_0, MU_0


@dataclass
class Grid2D:
    """2D Yee grid with staggered electric and magnetic field arrays.

    This class manages the spatial discretization and field storage for
    2D FDTD simulation. It supports both TMz and TEz polarization modes.

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration including grid, mode, and backend.

    Attributes
    ----------
    ez, hx, hy : numpy.ndarray
        TMz field components (Ez at integer points, Hx/Hy at half-integer).
    hz, ex, ey : numpy.ndarray
        TEz field components (Hz at half-integer, Ex/Ey at integer/half).
    eps_r : numpy.ndarray
        Relative permittivity at each grid cell.
    mu_r : numpy.ndarray
        Relative permeability at each grid cell.
    sigma_e : numpy.ndarray
        Electric conductivity [S/m] at each grid cell.
    sigma_m : numpy.ndarray
        Magnetic conductivity [Ω/m] at each grid cell (typically 0).

    Notes
    -----
    All arrays have shape (nx, ny) regardless of the actual staggered
    positions. This simplifies memory management and GPU kernel design.
    The half-cell offset is accounted for in the update equations.

    Computational Complexity
    -----------------------
    Memory: O(N) where N = nx * ny. For TMz, 3 field arrays + 4 material
    arrays = 7 * N * 8 bytes (float64). A 1000×1000 grid requires ~56 MB.
    """

    config: SimulationConfig

    # TMz fields
    ez: npt.NDArray[np.float64] = field(init=False, repr=False)
    hx: npt.NDArray[np.float64] = field(init=False, repr=False)
    hy: npt.NDArray[np.float64] = field(init=False, repr=False)

    # TEz fields
    hz: npt.NDArray[np.float64] = field(init=False, repr=False)
    ex: npt.NDArray[np.float64] = field(init=False, repr=False)
    ey: npt.NDArray[np.float64] = field(init=False, repr=False)

    # Material properties
    eps_r: npt.NDArray[np.float64] = field(init=False, repr=False)
    eps_inf: npt.NDArray[np.float64] = field(init=False, repr=False)
    mu_r: npt.NDArray[np.float64] = field(init=False, repr=False)
    sigma_e: npt.NDArray[np.float64] = field(init=False, repr=False)
    sigma_m: npt.NDArray[np.float64] = field(init=False, repr=False)

    # Dispersive material manager
    dispersive: "DispersiveManager" = field(init=False, repr=False)

    # Update coefficients (precomputed for performance)
    _ca: npt.NDArray[np.float64] = field(init=False, repr=False)
    _cb: npt.NDArray[np.float64] = field(init=False, repr=False)
    _da: npt.NDArray[np.float64] = field(init=False, repr=False)
    _db: npt.NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Allocate field arrays and initialize material properties."""
        nx = self.config.grid.nx
        ny = self.config.grid.ny
        shape = (nx, ny)

        # Initialize field arrays to zero
        if self.config.mode == SimulationMode.TMZ:
            self.ez = np.zeros(shape, dtype=np.float64)
            self.hx = np.zeros(shape, dtype=np.float64)
            self.hy = np.zeros(shape, dtype=np.float64)
            # TEz fields not used but initialized for safety
            self.hz = np.empty(0)
            self.ex = np.empty(0)
            self.ey = np.empty(0)
        elif self.config.mode == SimulationMode.TEZ:
            self.hz = np.zeros(shape, dtype=np.float64)
            self.ex = np.zeros(shape, dtype=np.float64)
            self.ey = np.zeros(shape, dtype=np.float64)
            # TMz fields not used
            self.ez = np.empty(0)
            self.hx = np.empty(0)
            self.hy = np.empty(0)
        else:
            raise ValueError(f"Grid2D does not support mode: {self.config.mode}")

        # Material properties — default: free space (ε_r=1, μ_r=1, σ=0)
        self.eps_r = np.ones(shape, dtype=np.float64)
        self.eps_inf = np.ones(shape, dtype=np.float64)
        self.mu_r = np.ones(shape, dtype=np.float64)
        self.sigma_e = np.zeros(shape, dtype=np.float64)
        self.sigma_m = np.zeros(shape, dtype=np.float64)

        from neurowave.materials.dispersive import DispersiveManager
        self.dispersive = DispersiveManager(nx, ny, max_poles=4)

        # Precompute update coefficients
        self._precompute_coefficients()

    def _precompute_coefficients(self) -> None:
        """Precompute FDTD update coefficients from material properties.

        For the E-field update with lossy dielectrics:

            Ca = (1 - σ_e·Δt / (2·ε)) / (1 + σ_e·Δt / (2·ε))
            Cb = (Δt / ε) / (1 + σ_e·Δt / (2·ε))

        For the H-field update with magnetic losses:

            Da = (1 - σ_m·Δt / (2·μ)) / (1 + σ_m·Δt / (2·μ))
            Db = (Δt / μ) / (1 + σ_m·Δt / (2·μ))

        These coefficients encode all material effects and need only be
        computed once (or when materials change).

        Notes
        -----
        Numerical stability: The denominators (1 + σΔt/2ε) are always
        positive for physical materials (σ ≥ 0, ε > 0), so division is safe.
        """
        dt = self.config.dt
        
        # Add effective permittivity from dispersive materials
        eps_eff_add = self.dispersive.compute_coefficients(dt)
        eps = self.eps_inf * EPS_0 + eps_eff_add
        
        mu = self.mu_r * MU_0

        # E-field coefficients
        sigma_dt_2eps = self.sigma_e * dt / (2.0 * eps)
        self._ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
        self._cb = (dt / eps) / (1.0 + sigma_dt_2eps)

        # H-field coefficients
        sigma_m_dt_2mu = self.sigma_m * dt / (2.0 * mu)
        self._da = (1.0 - sigma_m_dt_2mu) / (1.0 + sigma_m_dt_2mu)
        self._db = (dt / mu) / (1.0 + sigma_m_dt_2mu)

    def set_material_region(
        self,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
        eps_r: float = 1.0,
        mu_r: float = 1.0,
        sigma_e: float = 0.0,
        sigma_m: float = 0.0,
        eps_inf: Optional[float] = None,
        debye_poles: Optional[list] = None,
    ) -> None:
        """Set material properties in a rectangular region.

        Parameters
        ----------
        x_start, x_end : int
            Start and end indices in x-direction (exclusive end).
        y_start, y_end : int
            Start and end indices in y-direction (exclusive end).
        eps_r : float
            Relative permittivity (default: 1.0 = free space).
        mu_r : float
            Relative permeability (default: 1.0).
        sigma_e : float
            Electric conductivity [S/m] (default: 0.0).
        sigma_m : float
            Magnetic conductivity [Ω/m] (default: 0.0).

        Notes
        -----
        After calling this method, update coefficients are automatically
        recomputed to reflect the new material distribution.

        Physical constraints:
        - eps_r must be ≥ 1.0 for physical materials
        - mu_r must be ≥ 1.0 for non-magnetic materials
        - sigma_e must be ≥ 0.0
        - sigma_m must be ≥ 0.0
        """
        if eps_r < 1.0:
            raise ValueError(f"Relative permittivity must be ≥ 1.0, got {eps_r}")
        if mu_r < 1.0:
            raise ValueError(f"Relative permeability must be ≥ 1.0, got {mu_r}")
        if sigma_e < 0.0:
            raise ValueError(f"Electric conductivity must be ≥ 0.0, got {sigma_e}")

        if eps_inf is None:
            eps_inf = eps_r

        self.eps_r[x_start:x_end, y_start:y_end] = eps_r
        self.eps_inf[x_start:x_end, y_start:y_end] = eps_inf
        self.mu_r[x_start:x_end, y_start:y_end] = mu_r
        self.sigma_e[x_start:x_end, y_start:y_end] = sigma_e
        self.sigma_m[x_start:x_end, y_start:y_end] = sigma_m

        if debye_poles:
            region = (slice(x_start, x_end), slice(y_start, y_end))
            self.dispersive.add_poles(region, debye_poles)

        # Recompute coefficients with updated materials
        self._precompute_coefficients()

    def set_material_circle(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        eps_r: float = 1.0,
        mu_r: float = 1.0,
        sigma_e: float = 0.0,
        sigma_m: float = 0.0,
        eps_inf: Optional[float] = None,
        debye_poles: Optional[list] = None,
    ) -> None:
        """Set material properties in a circular region.

        Parameters
        ----------
        center_x, center_y : int
            Center of the circle in grid indices.
        radius : int
            Radius in grid cells.
        eps_r, mu_r, sigma_e, sigma_m : float
            Material properties (see set_material_region).
        """
        if eps_r < 1.0:
            raise ValueError(f"Relative permittivity must be ≥ 1.0, got {eps_r}")
            
        if eps_inf is None:
            eps_inf = eps_r

        nx, ny = self.config.grid.nx, self.config.grid.ny
        y_grid, x_grid = np.meshgrid(np.arange(ny), np.arange(nx))
        mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius ** 2

        self.eps_r[mask] = eps_r
        self.eps_inf[mask] = eps_inf
        self.mu_r[mask] = mu_r
        self.sigma_e[mask] = sigma_e
        self.sigma_m[mask] = sigma_m
        
        if debye_poles:
            self.dispersive.add_poles(mask, debye_poles)

        self._precompute_coefficients()

    def reset_fields(self) -> None:
        """Reset all field arrays to zero.

        Useful for restarting a simulation without reallocating memory
        or changing material properties.
        """
        if self.config.mode == SimulationMode.TMZ:
            self.ez[:] = 0.0
            self.hx[:] = 0.0
            self.hy[:] = 0.0
        elif self.config.mode == SimulationMode.TEZ:
            self.hz[:] = 0.0
            self.ex[:] = 0.0
            self.ey[:] = 0.0

    @property
    def memory_usage_bytes(self) -> int:
        """Estimate total memory usage for field and material arrays."""
        n = self.config.grid.total_cells
        # 3 field arrays + 4 material arrays + 4 coefficient arrays = 11
        return 11 * n * 8  # float64 = 8 bytes

    @property
    def memory_usage_mb(self) -> float:
        """Memory usage in megabytes."""
        return self.memory_usage_bytes / (1024 * 1024)

    def summary(self) -> str:
        """Return human-readable grid summary."""
        lines = [
            f"Grid2D [{self.config.mode.name}]",
            f"  Shape:    {self.config.grid.nx} × {self.config.grid.ny}",
            f"  Spacing:  Δx={self.config.grid.dx:.2e} m, "
            f"Δy={self.config.grid.dy:.2e} m",
            f"  Memory:   {self.memory_usage_mb:.1f} MB",
            f"  Timestep: {self.config.dt:.4e} s",
            f"  Materials: ε_r ∈ [{self.eps_r.min():.1f}, {self.eps_r.max():.1f}], "
            f"σ ∈ [{self.sigma_e.min():.2e}, {self.sigma_e.max():.2e}] S/m",
        ]
        return "\n".join(lines)
