"""
Dispersive material models for FDTD simulation.

Implements Auxiliary Differential Equation (ADE) methods for modeling
materials with frequency-dependent permittivity:
  - Debye: Biological tissues, water (microwave frequencies)
  - Drude: Metals, plasmas (DC-optical)
  - Lorentz: Optical resonances, metamaterials

References
----------
.. [1] Taflove & Hagness, "Computational Electrodynamics," 3rd ed., Ch. 9.
.. [2] Gedney, "Introduction to the FDTD Method," Ch. 9, Morgan & Claypool, 2011.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import numpy.typing as npt

from neurowave.core.constants import EPS_0
from neurowave.core import backend as xpb


# ---------------------------------------------------------------------------
# Pole dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DebyePole:
    """A single pole for the Debye dispersive material model.
    
    χ(ω) = Δε / (1 + jωτ)
        
    Parameters
    ----------
    delta_eps : float
        Change in relative permittivity (ε_s − ε_∞) for this pole.
    tau : float
        Relaxation time (seconds).
    """
    delta_eps: float
    tau: float


@dataclass
class DrudePole:
    """A single pole for the Drude dispersive material model.
    
    χ(ω) = −ω_p² / (ω² − jωγ)
    
    Common for metals (gold, silver) and cold plasmas.
        
    Parameters
    ----------
    omega_p : float
        Plasma frequency (rad/s).
    gamma : float
        Collision frequency / damping rate (rad/s).
    """
    omega_p: float
    gamma: float


@dataclass
class LorentzPole:
    """A single pole for the Lorentz dispersive material model.

    χ(ω) = Δε · ω_0² / (ω_0² + 2jωδ − ω²)

    Models resonant absorption in dielectrics and metamaterials.

    Parameters
    ----------
    delta_eps : float
        Oscillator strength (dimensionless).
    omega_0 : float
        Resonance frequency (rad/s).
    delta : float
        Damping coefficient (rad/s).
    """
    delta_eps: float
    omega_0: float
    delta: float


@dataclass
class ColeColePole:
    """A single pole for the Cole-Cole dispersive material model.

    χ(ω) = Δε / (1 + (jωτ)^(1-α))

    The Cole-Cole model extends the Debye model with fractional-order
    relaxation (α parameter) to model the broad dispersion observed in
    biological tissues. The standard 4-term Cole-Cole model (Gabriel et al.)
    accurately represents tissue dielectric properties from MHz to GHz.

    Reference
    ---------
    .. [1] Gabriel, S., et al. "The dielectric properties of biological
           tissues," Phys. Med. Biol. 41, 2231 (1996).
    .. [2] Kelley & Luebbers, "Piecewise linear recursive convolution for
           dispersive media," IEEE TAP 44(6), 792-797 (1996).

    Parameters
    ----------
    delta_eps : float
        Change in relative permittivity (ε_s − ε_∞) for this relaxation.
    tau : float
        Relaxation time constant (seconds).
    alpha : float
        Fractional distribution parameter (0 ≤ α < 1).
        α = 0 → standard Debye model.
        α > 0 → broader distribution of relaxation times.
    """
    delta_eps: float
    tau: float
    alpha: float

    def __post_init__(self):
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(f"Cole-Cole alpha must be in [0,1), got {self.alpha}")


# Type alias for any pole type
DispersivePole = Union[DebyePole, DrudePole, LorentzPole, ColeColePole]


# ---------------------------------------------------------------------------
# Dispersive Manager
# ---------------------------------------------------------------------------

class DispersiveManager:
    """Manages Auxiliary Differential Equations (ADE) for dispersive materials.
    
    Supports Debye, Drude, and Lorentz poles through a unified coefficient
    interface. All three reduce to the same update form:
    
        J^{n+1} = k1 · J^n + k2 · (E^{n+1} − E^n) / Δt
    
    for the Debye and Drude models (1st-order ADE). The Lorentz model uses
    a 2nd-order ADE that additionally tracks previous-step polarization.
    
    Memory footprint: For N poles on (nx, ny) grid ≈ 7·N·nx·ny·8 bytes.
    """
    
    def __init__(self, nx: int, ny: int, max_poles: int = 4, max_rc_order: int = 3):
        self.nx = nx
        self.ny = ny
        self.max_poles = max_poles
        self.max_rc_order = max_rc_order  # For Cole-Cole recursive convolution

        # Pole type tracking: 'debye', 'drude', 'lorentz', or 'colecole'
        self.pole_types: list[str] = ['none'] * max_poles

        # Physical parameters per pole (Debye: delta_eps, tau)
        self.delta_eps = xpb.zeros((max_poles, nx, ny))
        self.tau = xpb.zeros((max_poles, nx, ny))

        # Cole-Cole alpha parameter
        self.alpha = xpb.zeros((max_poles, nx, ny))

        # Drude/Lorentz extra parameters
        self.omega_p = xpb.zeros((max_poles, nx, ny))
        self.omega_0 = xpb.zeros((max_poles, nx, ny))
        self.damping = xpb.zeros((max_poles, nx, ny))

        # Precomputed ADE coefficients (unified for all pole types)
        self.k1 = xpb.zeros((max_poles, nx, ny))
        self.k2 = xpb.zeros((max_poles, nx, ny))
        self.gamma = xpb.zeros((max_poles, nx, ny))

        # Lorentz 2nd-order: extra coefficients and previous-step storage
        self.k3 = xpb.zeros((max_poles, nx, ny))
        self.k4 = xpb.zeros((max_poles, nx, ny))

        # Cole-Cole recursive convolution coefficients (Kelley & Luebbers method)
        # We store max_rc_order terms per pole
        self.cc_a = xpb.zeros((max_poles, max_rc_order, nx, ny))
        self.cc_b = xpb.zeros((max_poles, max_rc_order, nx, ny))
        self.cc_psi_ez = xpb.zeros((max_poles, max_rc_order, nx, ny))
        self.cc_psi_ex = xpb.zeros((max_poles, max_rc_order, nx, ny))
        self.cc_psi_ey = xpb.zeros((max_poles, max_rc_order, nx, ny))

        # Polarization current arrays (J_p) for each E-field component
        self.j_ez = xpb.zeros((max_poles, nx, ny))
        self.j_ex = xpb.zeros((max_poles, nx, ny))
        self.j_ey = xpb.zeros((max_poles, nx, ny))

        # Previous-step J for Lorentz (2nd-order)
        self.j_ez_prev = xpb.zeros((max_poles, nx, ny))
        self.j_ex_prev = xpb.zeros((max_poles, nx, ny))
        self.j_ey_prev = xpb.zeros((max_poles, nx, ny))

        self.active_poles = 0

    def add_poles(self, region: tuple | np.ndarray, poles: List[DispersivePole]) -> None:
        """Assign dispersive poles to a specific spatial region.
        
        Parameters
        ----------
        region : tuple or ndarray
            Grid slices or boolean mask defining the region.
        poles : list of DebyePole, DrudePole, or LorentzPole
            Poles to assign. Can mix types within a single call.
        """
        for i, pole in enumerate(poles):
            if i >= self.max_poles:
                raise ValueError(
                    f"Exceeded max_poles ({self.max_poles}). "
                    f"Increase max_poles in Grid2D."
                )
            
            if isinstance(pole, DebyePole):
                self.pole_types[i] = 'debye'
                self.delta_eps[i][region] = pole.delta_eps
                self.tau[i][region] = pole.tau
            elif isinstance(pole, DrudePole):
                self.pole_types[i] = 'drude'
                self.omega_p[i][region] = pole.omega_p
                self.damping[i][region] = pole.gamma
            elif isinstance(pole, LorentzPole):
                self.pole_types[i] = 'lorentz'
                self.delta_eps[i][region] = pole.delta_eps
                self.omega_0[i][region] = pole.omega_0
                self.damping[i][region] = pole.delta
            elif isinstance(pole, ColeColePole):
                self.pole_types[i] = 'colecole'
                self.delta_eps[i][region] = pole.delta_eps
                self.tau[i][region] = pole.tau
                self.alpha[i][region] = pole.alpha
            else:
                raise TypeError(f"Unknown pole type: {type(pole)}")

            self.active_poles = max(self.active_poles, i + 1)
            
    def compute_coefficients(self, dt: float) -> npt.NDArray[np.float64]:
        """Precompute ADE update coefficients and return effective permittivity.

        Returns
        -------
        eps_eff_add : ndarray of shape (nx, ny)
            Effective permittivity addition to be added to ε_∞·ε₀.
        """
        eps_eff_add = xpb.zeros((self.nx, self.ny))
        
        for p in range(self.active_poles):
            ptype = self.pole_types[p]
            
            if ptype == 'debye':
                valid = self.tau[p] > 0
                denom = 2.0 * self.tau[p][valid] + dt
                
                self.k1[p][valid] = (2.0 * self.tau[p][valid] - dt) / denom
                self.k2[p][valid] = (2.0 * EPS_0 * self.delta_eps[p][valid] * dt) / denom
                self.gamma[p][valid] = (1.0 + self.k1[p][valid]) / 2.0
                
                eps_eff_add[valid] += self.k2[p][valid] / 2.0
                
            elif ptype == 'drude':
                valid = self.omega_p[p] > 0
                g = self.damping[p][valid]
                wp2 = self.omega_p[p][valid] ** 2
                denom = 2.0 + g * dt
                
                # Drude ADE: J^{n+1} = k1·J^n + k2·E^{n+1}
                # k1 = (2 - γΔt) / (2 + γΔt)
                # k2 = ε₀·ωp²·2Δt / (2 + γΔt)
                self.k1[p][valid] = (2.0 - g * dt) / denom
                self.k2[p][valid] = EPS_0 * wp2 * 2.0 * dt / denom
                self.gamma[p][valid] = (1.0 + self.k1[p][valid]) / 2.0
                
                eps_eff_add[valid] += self.k2[p][valid] / 2.0
                
            elif ptype == 'lorentz':
                valid = self.omega_0[p] > 0
                d = self.damping[p][valid]
                w0_2 = self.omega_0[p][valid] ** 2
                de = self.delta_eps[p][valid]
                dt2 = dt * dt
                
                # Lorentz 2nd-order ADE:
                # denom = 4 + 2δΔt + ω₀²Δt²
                denom = 4.0 + 2.0 * d * dt + w0_2 * dt2
                
                # J^{n+1} = k1·J^n + k3·J^{n-1} + k2·(E^{n+1} - E^{n-1})
                # But we reformulate to use (E^{n+1} - E^n)/dt like Debye:
                # k1 = (8 - 2ω₀²Δt²) / denom
                # k3 = -(4 - 2δΔt + ω₀²Δt²) / denom
                # k2 = 2ε₀·Δε·ω₀²·Δt / denom  (effective for E-field coupling)
                self.k1[p][valid] = (8.0 - 2.0 * w0_2 * dt2) / denom
                self.k3[p][valid] = -(4.0 - 2.0 * d * dt + w0_2 * dt2) / denom
                self.k2[p][valid] = 2.0 * EPS_0 * de * w0_2 * dt2 / denom
                self.gamma[p][valid] = 1.0  # Full subtraction of J

                eps_eff_add[valid] += self.k2[p][valid] / 2.0

            elif ptype == 'colecole':
                # Cole-Cole: recursive convolution method (Kelley & Luebbers, 1996)
                # χ(ω) = Δε / (1 + (jωτ)^(1-α))
                # Approximated as a sum of M Debye poles:
                # χ(ω) ≈ Σ[m=1 to M] (Δε_m / (1 + jωτ_m))
                valid = self.tau[p] > 0
                alpha_val = self.alpha[p][valid]
                tau_val = self.tau[p][valid]
                de_val = self.delta_eps[p][valid]

                # Compute M exponentially-spaced poles
                # Use logarithmic spacing from τ/10 to τ*10 (2-decade range)
                tau_min = tau_val / 10.0
                tau_max = tau_val * 10.0

                for m in range(self.max_rc_order):
                    # Logarithmically spaced relaxation times
                    if self.max_rc_order > 1:
                        log_tau_m = (np.log(tau_min) +
                                     m * (np.log(tau_max) - np.log(tau_min)) /
                                     (self.max_rc_order - 1))
                        tau_m = np.exp(log_tau_m)
                    else:
                        tau_m = tau_val

                    # Weight for this pole (normalized so sum = Δε)
                    # For Cole-Cole, weight decreases as power-law
                    # w_m ∝ τ_m^(-α) / Σ(τ_k^(-α))
                    # Simplified: equal weights for first-order approximation
                    delta_eps_m = de_val / self.max_rc_order

                    # Standard Debye RC coefficients
                    denom_m = 2.0 * tau_m + dt
                    self.cc_a[p, m][valid] = (2.0 * tau_m - dt) / denom_m
                    self.cc_b[p, m][valid] = (2.0 * EPS_0 * delta_eps_m * dt) / denom_m

                    eps_eff_add[valid] += self.cc_b[p, m][valid] / 2.0

        return eps_eff_add
        
    def get_sum_gamma_j(self, component: str) -> npt.NDArray[np.float64]:
        """Get the Σ(γ·J) term for the E-field update.
        
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
            return xpb.zeros((self.nx, self.ny))

        xp = xpb.get_backend_module()
        j_array = getattr(self, f"j_{component.lower()}")
        return xp.sum(
            self.gamma[:self.active_poles] * j_array[:self.active_poles],
            axis=0,
        )
        
    def update_j_fields(
        self,
        component: str,
        e_new: np.ndarray,
        e_old: np.ndarray,
        dt: float,
    ) -> None:
        """Update the polarization currents after E-field update.
        
        Parameters
        ----------
        component : str
            Field component ('Ez', 'Ex', 'Ey').
        e_new, e_old : ndarray
            New and previous E-field arrays.
        dt : float
            Timestep size.
        """
        if self.active_poles == 0:
            return
            
        j_array = getattr(self, f"j_{component.lower()}")
        j_prev = getattr(self, f"j_{component.lower()}_prev")
        de_dt = (e_new - e_old) / dt
        
        for p in range(self.active_poles):
            ptype = self.pole_types[p]

            if ptype in ('debye', 'drude'):
                # 1st-order: J^{n+1} = k1·J^n + k2·dE/dt
                j_array[p] = self.k1[p] * j_array[p] + self.k2[p] * de_dt

            elif ptype == 'lorentz':
                # 2nd-order: J^{n+1} = k1·J^n + k3·J^{n-1} + k2·dE/dt
                j_new = (
                    self.k1[p] * j_array[p]
                    + self.k3[p] * j_prev[p]
                    + self.k2[p] * de_dt
                )
                j_prev[p] = j_array[p].copy()
                j_array[p] = j_new

            elif ptype == 'colecole':
                # Cole-Cole: sum of recursive convolution terms
                # Each term: ψ_m^{n+1} = a_m·ψ_m^n + b_m·E^{n+1}
                # J = Σ ψ_m
                j_total = xpb.zeros_like(e_new)

                if component == 'Ez':
                    psi_array = self.cc_psi_ez
                elif component == 'Ex':
                    psi_array = self.cc_psi_ex
                else:  # Ey
                    psi_array = self.cc_psi_ey

                for m in range(self.max_rc_order):
                    psi_array[p, m] = (
                        self.cc_a[p, m] * psi_array[p, m] +
                        self.cc_b[p, m] * e_new
                    )
                    j_total += psi_array[p, m]

                j_array[p] = j_total
