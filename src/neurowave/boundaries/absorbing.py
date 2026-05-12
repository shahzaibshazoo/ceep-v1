"""
Boundary conditions for FDTD simulation.

Implements:
- PEC: Perfect Electric Conductor (simplest, zero tangential E)
- MurABC: 1st order Mur absorbing boundary (simple, moderate absorption)
- CPML: Convolutional PML (gold standard, excellent wideband absorption)

Theory
------
PEC: Sets tangential E-field to zero at boundaries. Simple but creates
total reflection — useful for waveguide walls and resonant cavities.

Mur ABC: Uses the 1D wave equation at boundaries to approximate outgoing
wave propagation. First-order accurate, works well for normal incidence
but degrades at oblique angles.

CPML: Extends the computational domain with a lossy, impedance-matched
layer that absorbs outgoing waves with minimal reflection. Uses recursive
convolution for efficient implementation. Works well across all angles
and frequencies.

References
----------
.. [1] G. Mur, "Absorbing boundary conditions for the finite-difference
       approximation of the time-domain electromagnetic-field equations,"
       IEEE Trans. EMC, vol. 23, pp. 377-382, 1981.
.. [2] J. A. Roden and S. D. Gedney, "Convolution PML (CPML),"
       Microwave Opt. Technol. Lett., vol. 27, pp. 334-339, 2000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from neurowave.core.base import BaseBoundary
from neurowave.core.config import SimulationConfig, SimulationMode
from neurowave.core.constants import C_0, EPS_0, MU_0


@dataclass
class PEC(BaseBoundary):
    """Perfect Electric Conductor boundary condition.

    Sets tangential E-field components to zero at all grid boundaries.
    This creates perfect reflection — useful for metallic walls,
    waveguide simulations, and resonant cavities.

    Computational cost: O(perimeter) per timestep.
    """

    def apply_e_field(self, e_fields, h_fields, config):
        """Zero out E-field at boundaries."""
        if config.mode == SimulationMode.TMZ:
            ez = e_fields["Ez"]
            ez[0, :] = 0.0
            ez[-1, :] = 0.0
            ez[:, 0] = 0.0
            ez[:, -1] = 0.0
        elif config.mode == SimulationMode.TEZ:
            ex = e_fields["Ex"]
            ey = e_fields["Ey"]
            ex[0, :] = 0.0
            ex[-1, :] = 0.0
            ey[:, 0] = 0.0
            ey[:, -1] = 0.0

    def apply_h_field(self, e_fields, h_fields, config):
        """No action needed for PEC on H-fields."""
        pass


@dataclass
class MurABC(BaseBoundary):
    """First-order Mur absorbing boundary condition.

    Approximates the 1D wave equation at each boundary face:
        (∂/∂t ± c·∂/∂n) E = 0

    where n is the outward normal direction.

    Discretized (1st order):
        E_boundary^{n+1} = E_interior^n + ((cΔt - Δx)/(cΔt + Δx))
                           × (E_interior^{n+1} - E_boundary^n)

    Limitations
    -----------
    - Only first-order accurate
    - Reflection increases at oblique incidence angles
    - Typical reflection coefficient: -20 to -30 dB at normal incidence
    - For production use, prefer CPML

    Computational cost: O(perimeter) per timestep.
    """

    # Previous boundary values for the time-stepping formula
    _prev_x0: Optional[npt.NDArray[np.float64]] = field(
        init=False, default=None, repr=False
    )
    _prev_xn: Optional[npt.NDArray[np.float64]] = field(
        init=False, default=None, repr=False
    )
    _prev_y0: Optional[npt.NDArray[np.float64]] = field(
        init=False, default=None, repr=False
    )
    _prev_yn: Optional[npt.NDArray[np.float64]] = field(
        init=False, default=None, repr=False
    )
    _initialized: bool = field(init=False, default=False)

    def _initialize(self, config: SimulationConfig) -> None:
        """Allocate storage for previous boundary values."""
        nx, ny = config.grid.nx, config.grid.ny
        self._prev_x0 = np.zeros(ny, dtype=np.float64)
        self._prev_xn = np.zeros(ny, dtype=np.float64)
        self._prev_y0 = np.zeros(nx, dtype=np.float64)
        self._prev_yn = np.zeros(nx, dtype=np.float64)
        self._initialized = True

    def apply_e_field(self, e_fields, h_fields, config):
        """Apply Mur ABC to E-field boundaries."""
        if not self._initialized:
            self._initialize(config)

        if config.mode == SimulationMode.TMZ:
            ez = e_fields["Ez"]
        elif config.mode == SimulationMode.TEZ:
            # For TEz, apply to Hz instead
            ez = h_fields.get("Hz", e_fields.get("Hz"))
            if ez is None:
                return
        else:
            return

        # Mur coefficient: (c·Δt - Δx) / (c·Δt + Δx)
        c_dt = C_0 * config.dt
        coeff_x = (c_dt - config.grid.dx) / (c_dt + config.grid.dx)
        coeff_y = (c_dt - config.grid.dy) / (c_dt + config.grid.dy)

        # Store current interior values before update
        new_prev_x0 = ez[1, :].copy()
        new_prev_xn = ez[-2, :].copy()
        new_prev_y0 = ez[:, 1].copy()
        new_prev_yn = ez[:, -2].copy()

        # Apply Mur ABC at each face
        # x = 0 face
        ez[0, :] = self._prev_x0 + coeff_x * (ez[1, :] - ez[0, :])
        # x = nx-1 face
        ez[-1, :] = self._prev_xn + coeff_x * (ez[-2, :] - ez[-1, :])
        # y = 0 face
        ez[:, 0] = self._prev_y0 + coeff_y * (ez[:, 1] - ez[:, 0])
        # y = ny-1 face
        ez[:, -1] = self._prev_yn + coeff_y * (ez[:, -2] - ez[:, -1])

        # Update stored values
        self._prev_x0 = new_prev_x0
        self._prev_xn = new_prev_xn
        self._prev_y0 = new_prev_y0
        self._prev_yn = new_prev_yn

    def apply_h_field(self, e_fields, h_fields, config):
        """No action needed for Mur ABC on H-fields."""
        pass


@dataclass
class CPML(BaseBoundary):
    """Convolutional Perfectly Matched Layer (CPML).

    The gold standard absorbing boundary for FDTD. Uses a graded
    conductivity profile within a layer of cells at each boundary
    to absorb outgoing waves with minimal reflection.

    CPML Theory
    -----------
    The PML stretches the spatial coordinates into the complex plane:
        x → x + (1/jω) ∫ σ_x(x') dx'

    The CPML formulation avoids split fields by using recursive
    convolution with auxiliary variables ψ (psi):
        ψ_e and ψ_h accumulate the convolutional terms

    The conductivity profile is polynomial-graded:
        σ(d) = σ_max · (d / thickness)^order

    where d is the distance into the PML from the interior boundary.

    Parameters
    ----------
    thickness : int
        Number of PML cells (typically 8-20). Default: 10.
    order : int
        Polynomial grading order (typically 3-4). Default: 3.
    sigma_factor : float
        σ_max scaling factor. Default: 1.5 (optimized for most cases).
    alpha_max : float
        Maximum α value for CFS-PML (improves evanescent wave absorption).
    kappa_max : float
        Maximum κ value for CFS-PML stretching.

    Computational Cost
    ------------------
    Memory: O(thickness × perimeter) for auxiliary ψ arrays
    Time: O(thickness × perimeter) per timestep

    Typical Performance
    -------------------
    - 10-cell PML: reflection < -60 dB
    - 15-cell PML: reflection < -80 dB
    - 20-cell PML: reflection < -100 dB

    References
    ----------
    .. [1] Roden & Gedney, "Convolution PML (CPML)," MOTL, 2000.
    .. [2] Taflove, Ch. 7, "Perfectly Matched Layer."
    """

    thickness: int = 10
    order: int = 4
    sigma_factor: float = 2.0
    alpha_max: float = 0.05
    kappa_max: float = 1.0

    # Internal state
    _psi_ezx_xlo: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_ezx_xhi: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_ezy_ylo: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_ezy_yhi: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_hyx_xlo: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_hyx_xhi: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_hxy_ylo: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_hxy_yhi: Optional[npt.NDArray] = field(init=False, default=None, repr=False)

    # TEz Psi arrays
    _psi_hzx_xlo: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_hzx_xhi: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_hzy_ylo: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_hzy_yhi: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_eyx_xlo: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_eyx_xhi: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_exy_ylo: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _psi_exy_yhi: Optional[npt.NDArray] = field(init=False, default=None, repr=False)

    _be_x: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _ce_x: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _bh_x: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _ch_x: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _be_y: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _ce_y: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _bh_y: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _ch_y: Optional[npt.NDArray] = field(init=False, default=None, repr=False)
    _initialized: bool = field(init=False, default=False)

    def _compute_pml_params(
        self, n_cells: int, d: float, dt: float
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Compute CPML b and c coefficients for a given direction.

        The polynomial grading profile:
            σ(ρ) = σ_max · ρ^order
            α(ρ) = α_max · (1 - ρ)
            κ(ρ) = 1 + (κ_max - 1) · ρ^order

        where ρ = distance/thickness (normalized depth into PML).

        Returns be, ce for E-field positions and bh, ch for H-field positions.
        """
        # Optimal σ_max from Taflove: σ_opt = (order+1) / (150π·Δx)
        sigma_max = self.sigma_factor * (self.order + 1) / (150.0 * math.pi * d)

        # E-field positions: at integer points 0, 1, ..., thickness-1
        rho_e = np.arange(n_cells, dtype=np.float64) / n_cells
        sigma_e = sigma_max * rho_e ** self.order
        kappa_e = 1.0 + (self.kappa_max - 1.0) * rho_e ** self.order
        alpha_e = self.alpha_max * (1.0 - rho_e)

        # H-field positions: at half-integer points 0.5, 1.5, ...
        rho_h = (np.arange(n_cells, dtype=np.float64) + 0.5) / n_cells
        sigma_h = sigma_max * rho_h ** self.order
        kappa_h = 1.0 + (self.kappa_max - 1.0) * rho_h ** self.order
        alpha_h = self.alpha_max * (1.0 - rho_h)

        # CPML coefficients: b = exp(-(σ/κ + α)·Δt/ε₀)
        be = np.exp(-(sigma_e / kappa_e + alpha_e) * dt / EPS_0)
        ce = np.where(
            np.abs(sigma_e) < 1e-20,
            0.0,
            sigma_e / (sigma_e * kappa_e + kappa_e ** 2 * alpha_e) * (be - 1.0),
        )

        bh = np.exp(-(sigma_h / kappa_h + alpha_h) * dt / EPS_0)
        ch = np.where(
            np.abs(sigma_h) < 1e-20,
            0.0,
            sigma_h / (sigma_h * kappa_h + kappa_h ** 2 * alpha_h) * (bh - 1.0),
        )

        return be, ce, bh, ch

    def _initialize(self, config: SimulationConfig) -> None:
        """Allocate ψ arrays and compute PML coefficients."""
        nx, ny = config.grid.nx, config.grid.ny
        dt = config.dt
        t = self.thickness

        # Compute coefficients for x and y directions
        self._be_x, self._ce_x, self._bh_x, self._ch_x = self._compute_pml_params(
            t, config.grid.dx, dt
        )
        self._be_y, self._ce_y, self._bh_y, self._ch_y = self._compute_pml_params(
            t, config.grid.dy, dt
        )

        # Allocate ψ arrays for each PML region
        # x-low and x-high PML regions: thickness × ny
        self._psi_ezx_xlo = np.zeros((t, ny), dtype=np.float64)
        self._psi_ezx_xhi = np.zeros((t, ny), dtype=np.float64)
        self._psi_hyx_xlo = np.zeros((t, ny), dtype=np.float64)
        self._psi_hyx_xhi = np.zeros((t, ny), dtype=np.float64)

        # y-low and y-high PML regions: nx × thickness
        self._psi_ezy_ylo = np.zeros((nx, t), dtype=np.float64)
        self._psi_ezy_yhi = np.zeros((nx, t), dtype=np.float64)
        self._psi_hxy_ylo = np.zeros((nx, t), dtype=np.float64)
        self._psi_hxy_yhi = np.zeros((nx, t), dtype=np.float64)

        # TEz psi arrays
        self._psi_hzx_xlo = np.zeros((t, ny), dtype=np.float64)
        self._psi_hzx_xhi = np.zeros((t, ny), dtype=np.float64)
        self._psi_eyx_xlo = np.zeros((t, ny), dtype=np.float64)
        self._psi_eyx_xhi = np.zeros((t, ny), dtype=np.float64)

        self._psi_hzy_ylo = np.zeros((nx, t), dtype=np.float64)
        self._psi_hzy_yhi = np.zeros((nx, t), dtype=np.float64)
        self._psi_exy_ylo = np.zeros((nx, t), dtype=np.float64)
        self._psi_exy_yhi = np.zeros((nx, t), dtype=np.float64)

        self._initialized = True

    def apply_h_field(self, e_fields, h_fields, config):
        """Apply CPML corrections to H-field update in PML regions."""
        if not self._initialized:
            self._initialize(config)

        t = self.thickness
        nx, ny = config.grid.nx, config.grid.ny
        dx, dy = config.grid.dx, config.grid.dy
        dt = config.dt

        if config.mode == SimulationMode.TEZ:
            hz = h_fields["Hz"]
            ex = e_fields["Ex"]
            ey = e_fields["Ey"]

            # X-direction PML for Hz (∂Ey/∂x term) -> hz -= dt/mu * dEy/dx
            for i in range(t):
                idx = t - 1 - i
                self._psi_hzx_xlo[i, :] = (
                    self._bh_x[idx] * self._psi_hzx_xlo[i, :]
                    + self._ch_x[idx] * (ey[i + 1, :] - ey[i, :]) / dx
                )
                hz[i, :] -= dt / MU_0 * self._psi_hzx_xlo[i, :]

            for i in range(t):
                idx = i
                gi = nx - t + i
                if gi < nx - 1:
                    self._psi_hzx_xhi[i, :] = (
                        self._bh_x[idx] * self._psi_hzx_xhi[i, :]
                        + self._ch_x[idx] * (ey[gi + 1, :] - ey[gi, :]) / dx
                    )
                    hz[gi, :] -= dt / MU_0 * self._psi_hzx_xhi[i, :]

            # Y-direction PML for Hz (∂Ex/∂y term) -> hz += dt/mu * dEx/dy
            for j in range(t):
                idx = t - 1 - j
                self._psi_hzy_ylo[:, j] = (
                    self._bh_y[idx] * self._psi_hzy_ylo[:, j]
                    + self._ch_y[idx] * (ex[:, j + 1] - ex[:, j]) / dy
                )
                hz[:, j] += dt / MU_0 * self._psi_hzy_ylo[:, j]

            for j in range(t):
                idx = j
                gj = ny - t + j
                if gj < ny - 1:
                    self._psi_hzy_yhi[:, j] = (
                        self._bh_y[idx] * self._psi_hzy_yhi[:, j]
                        + self._ch_y[idx] * (ex[:, gj + 1] - ex[:, gj]) / dy
                    )
                    hz[:, gj] += dt / MU_0 * self._psi_hzy_yhi[:, j]
            return

        if config.mode != SimulationMode.TMZ:
            return

        ez = e_fields["Ez"]
        hx = h_fields["Hx"]
        hy = h_fields["Hy"]

        # --- X-direction PML for Hy ---
        # x-low PML (indices 0..t-1)
        for i in range(t):
            idx = t - 1 - i  # PML depth index (0=deepest, t-1=interface)
            self._psi_hyx_xlo[i, :] = (
                self._bh_x[idx] * self._psi_hyx_xlo[i, :]
                + self._ch_x[idx] * (ez[i + 1, :] - ez[i, :]) / dx
            )
            hy[i, :] += dt / MU_0 * self._psi_hyx_xlo[i, :]

        # x-high PML (indices nx-t..nx-1)
        for i in range(t):
            idx = i  # PML depth index
            gi = nx - t + i
            if gi < nx - 1:
                self._psi_hyx_xhi[i, :] = (
                    self._bh_x[idx] * self._psi_hyx_xhi[i, :]
                    + self._ch_x[idx] * (ez[gi + 1, :] - ez[gi, :]) / dx
                )
                hy[gi, :] += dt / MU_0 * self._psi_hyx_xhi[i, :]

        # --- Y-direction PML for Hx ---
        # y-low PML
        for j in range(t):
            idx = t - 1 - j
            self._psi_hxy_ylo[:, j] = (
                self._bh_y[idx] * self._psi_hxy_ylo[:, j]
                + self._ch_y[idx] * (ez[:, j + 1] - ez[:, j]) / dy
            )
            hx[:, j] -= dt / MU_0 * self._psi_hxy_ylo[:, j]

        # y-high PML
        for j in range(t):
            idx = j
            gj = ny - t + j
            if gj < ny - 1:
                self._psi_hxy_yhi[:, j] = (
                    self._bh_y[idx] * self._psi_hxy_yhi[:, j]
                    + self._ch_y[idx] * (ez[:, gj + 1] - ez[:, gj]) / dy
                )
                hx[:, gj] -= dt / MU_0 * self._psi_hxy_yhi[:, j]

    def apply_e_field(self, e_fields, h_fields, config):
        """Apply CPML corrections to E-field update in PML regions."""
        if not self._initialized:
            self._initialize(config)

        t = self.thickness
        nx, ny = config.grid.nx, config.grid.ny
        dx, dy = config.grid.dx, config.grid.dy
        dt = config.dt

        if config.mode == SimulationMode.TEZ:
            hz = h_fields["Hz"]
            ex = e_fields["Ex"]
            ey = e_fields["Ey"]

            # Y-direction PML for Ex (∂Hz/∂y term) -> ex += dt/eps * dHz/dy
            for j in range(t):
                idx = t - 1 - j
                if j > 0:
                    self._psi_exy_ylo[:, j] = (
                        self._be_y[idx] * self._psi_exy_ylo[:, j]
                        + self._ce_y[idx] * (hz[:, j] - hz[:, j - 1]) / dy
                    )
                    ex[:, j] += dt / EPS_0 * self._psi_exy_ylo[:, j]

            for j in range(t):
                idx = j
                gj = ny - t + j
                self._psi_exy_yhi[:, j] = (
                    self._be_y[idx] * self._psi_exy_yhi[:, j]
                    + self._ce_y[idx] * (hz[:, gj] - hz[:, gj - 1]) / dy
                )
                ex[:, gj] += dt / EPS_0 * self._psi_exy_yhi[:, j]

            # X-direction PML for Ey (∂Hz/∂x term) -> ey -= dt/eps * dHz/dx
            for i in range(t):
                idx = t - 1 - i
                if i > 0:
                    self._psi_eyx_xlo[i, :] = (
                        self._be_x[idx] * self._psi_eyx_xlo[i, :]
                        + self._ce_x[idx] * (hz[i, :] - hz[i - 1, :]) / dx
                    )
                    ey[i, :] -= dt / EPS_0 * self._psi_eyx_xlo[i, :]

            for i in range(t):
                idx = i
                gi = nx - t + i
                self._psi_eyx_xhi[i, :] = (
                    self._be_x[idx] * self._psi_eyx_xhi[i, :]
                    + self._ce_x[idx] * (hz[gi, :] - hz[gi - 1, :]) / dx
                )
                ey[gi, :] -= dt / EPS_0 * self._psi_eyx_xhi[i, :]
            return

        if config.mode != SimulationMode.TMZ:
            return

        ez = e_fields["Ez"]
        hx = h_fields["Hx"]
        hy = h_fields["Hy"]

        # --- X-direction PML for Ez (∂Hy/∂x term) ---
        # x-low PML
        for i in range(t):
            idx = t - 1 - i
            if i > 0:
                self._psi_ezx_xlo[i, :] = (
                    self._be_x[idx] * self._psi_ezx_xlo[i, :]
                    + self._ce_x[idx] * (hy[i, :] - hy[i - 1, :]) / dx
                )
                ez[i, :] += dt / EPS_0 * self._psi_ezx_xlo[i, :]

        # x-high PML
        for i in range(t):
            idx = i
            gi = nx - t + i
            self._psi_ezx_xhi[i, :] = (
                self._be_x[idx] * self._psi_ezx_xhi[i, :]
                + self._ce_x[idx] * (hy[gi, :] - hy[gi - 1, :]) / dx
            )
            ez[gi, :] += dt / EPS_0 * self._psi_ezx_xhi[i, :]

        # --- Y-direction PML for Ez (∂Hx/∂y term) ---
        # y-low PML
        for j in range(t):
            idx = t - 1 - j
            if j > 0:
                self._psi_ezy_ylo[:, j] = (
                    self._be_y[idx] * self._psi_ezy_ylo[:, j]
                    + self._ce_y[idx] * (hx[:, j] - hx[:, j - 1]) / dy
                )
                ez[:, j] -= dt / EPS_0 * self._psi_ezy_ylo[:, j]

        # y-high PML
        for j in range(t):
            idx = j
            gj = ny - t + j
            self._psi_ezy_yhi[:, j] = (
                self._be_y[idx] * self._psi_ezy_yhi[:, j]
                + self._ce_y[idx] * (hx[:, gj] - hx[:, gj - 1]) / dy
            )
            ez[:, gj] -= dt / EPS_0 * self._psi_ezy_yhi[:, j]
