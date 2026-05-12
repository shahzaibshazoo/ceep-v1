"""
Dispersive material models for FDTD simulation.

Implements Auxiliary Differential Equation (ADE) methods for modeling
materials with frequency-dependent permittivity, such as biological tissues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from neurowave.core.constants import EPS_0


@dataclass
class DebyePole:
    """A single pole for the Debye dispersive material model.
    
    The relative permittivity of a multi-pole Debye medium is:
        ε_r(ω) = ε_inf + Σ [ Δε_p / (1 + j·ω·τ_p) ]
        
    Parameters
    ----------
    delta_eps : float
        Change in relative permittivity (ε_s - ε_inf) for this pole.
    tau : float
        Relaxation time (seconds).
    """
    delta_eps: float
    tau: float


class DispersiveManager:
    """Manages Auxiliary Differential Equations (ADE) for dispersive materials.
    
    This manager tracks the auxiliary polarization currents (J_p) and 
    update coefficients for a generic multi-pole Debye medium across the grid.
    
    Memory footprint: For N poles, requires 4 arrays of shape (N, nx, ny)
    plus coefficients. For 4 poles on a 400x400 grid, this is roughly 25 MB.
    """
    
    def __init__(self, nx: int, ny: int, max_poles: int = 4):
        self.nx = nx
        self.ny = ny
        self.max_poles = max_poles
        
        # Physical parameters per pole
        self.delta_eps = np.zeros((max_poles, nx, ny), dtype=np.float64)
        self.tau = np.zeros((max_poles, nx, ny), dtype=np.float64)
        
        # Precomputed ADE coefficients
        self.k1 = np.zeros((max_poles, nx, ny), dtype=np.float64)
        self.k2 = np.zeros((max_poles, nx, ny), dtype=np.float64)
        self.gamma = np.zeros((max_poles, nx, ny), dtype=np.float64)
        
        # Polarization current arrays (J_p) for each E-field component
        # J = ∂P/∂t
        self.j_ez = np.zeros((max_poles, nx, ny), dtype=np.float64)
        self.j_ex = np.zeros((max_poles, nx, ny), dtype=np.float64)
        self.j_ey = np.zeros((max_poles, nx, ny), dtype=np.float64)
        
        self.active_poles = 0

    def add_poles(self, region: tuple | np.ndarray, poles: List[DebyePole]) -> None:
        """Assign Debye poles to a specific spatial region.
        
        Parameters
        ----------
        region : tuple or ndarray
            Grid slices or boolean mask defining the region.
        poles : list of DebyePole
            List of poles to assign to the region.
        """
        for i, pole in enumerate(poles):
            if i >= self.max_poles:
                raise ValueError(f"Exceeded max_poles ({self.max_poles}). Increase max_poles in Grid2D.")
            
            # Using tuple unpacking to construct the 3D index: (i, region_x, region_y)
            # If region is a mask, we need to apply it correctly.
            # Easiest way: self.delta_eps[i][region] = ...
            self.delta_eps[i][region] = pole.delta_eps
            self.tau[i][region] = pole.tau
            self.active_poles = max(self.active_poles, i + 1)
            
    def compute_coefficients(self, dt: float) -> npt.NDArray[np.float64]:
        """Precompute ADE update coefficients and return effective permittivity.
        
        This must be called before the main FDTD loop begins.
        
        Returns
        -------
        eps_eff_add : ndarray
            The effective permittivity addition (Δε_eff) that must be added 
            to the static ε_inf before computing standard Ca/Cb coefficients.
        """
        valid = self.tau > 0
        
        # k1 = (2*tau - dt) / (2*tau + dt)
        self.k1[valid] = (2 * self.tau[valid] - dt) / (2 * self.tau[valid] + dt)
        
        # k2 = 2 * ε₀ * Δε * dt / (2*tau + dt)
        self.k2[valid] = (2 * EPS_0 * self.delta_eps[valid] * dt) / (2 * self.tau[valid] + dt)
        
        # gamma = (1 + k1) / 2
        self.gamma[valid] = (1.0 + self.k1[valid]) / 2.0
        
        # Effective permittivity addition: sum_p (k2_p / 2)
        eps_eff_add = np.sum(self.k2 / 2.0, axis=0)
        return eps_eff_add
        
    def get_sum_gamma_j(self, component: str) -> npt.NDArray[np.float64]:
        """Get the sum(gamma * J) term for the E-field update.
        
        Parameters
        ----------
        component : str
            Field component ('Ez', 'Ex', 'Ey').
            
        Returns
        -------
        ndarray
            The summation term to be subtracted in the E-field update.
        """
        if self.active_poles == 0:
            return np.zeros((self.nx, self.ny), dtype=np.float64)
            
        j_array = getattr(self, f"j_{component.lower()}")
        return np.sum(self.gamma[:self.active_poles] * j_array[:self.active_poles], axis=0)
        
    def update_j_fields(self, component: str, e_new: np.ndarray, e_old: np.ndarray, dt: float) -> None:
        """Update the polarization currents for a specific field component.
        
        J^{n+1}_p = k_{1,p} J^n_p + (k_{2,p} / dt) * (E^{n+1} - E^n)
        
        Parameters
        ----------
        component : str
            Field component ('Ez', 'Ex', 'Ey').
        e_new : ndarray
            The newly updated E-field (E^{n+1}).
        e_old : ndarray
            The previous E-field (E^n).
        dt : float
            Timestep size.
        """
        if self.active_poles == 0:
            return
            
        j_array = getattr(self, f"j_{component.lower()}")
        de_dt = (e_new - e_old) / dt
        
        for p in range(self.active_poles):
            j_array[p] = self.k1[p] * j_array[p] + self.k2[p] * de_dt
