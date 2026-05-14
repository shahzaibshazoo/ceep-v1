"""
Total-Field / Scattered-Field (TF/SF) plane wave source.

Injects a 1D auxiliary plane wave into the 2D FDTD grid along a specified
boundary, enabling clean plane wave illumination without edge diffraction
artifacts from soft point sources.

References
----------
.. [1] Taflove & Hagness, "Computational Electrodynamics," 3rd ed., Ch. 5.6.
.. [2] S. D. Gedney, "Introduction to the FDTD Method," Ch. 7, 2011.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from neurowave.core.base import BaseSource
from neurowave.core.constants import C_0, EPS_0, MU_0, ETA_0, TWO_PI


@dataclass
class PlaneWaveSource(BaseSource):
    """Total-Field / Scattered-Field (TF/SF) plane wave source.
    
    Injects a uniform plane wave propagating in the +x direction into
    a rectangular TF/SF boundary. The fields outside the boundary remain
    in a scattered-field region (ideally zero for a homogeneous medium).
    
    Parameters
    ----------
    x_start, x_end : int
        TF/SF boundary x-range (the wave enters at x_start, exits at x_end).
    y_start, y_end : int
        TF/SF boundary y-range.
    frequency_max : float
        Maximum frequency for the Gaussian pulse envelope (Hz).
    amplitude : float
        Peak amplitude of the incident E-field.
    field_component : str
        Which E-field component carries the wave ('Ez' for TMz).
    """
    
    x_start: int = 10
    x_end: int = 90
    y_start: int = 10
    y_end: int = 90
    frequency_max: float = 5e9
    amplitude: float = 1.0
    delay_factor: float = 6.0
    field_component: str = "Ez"
    
    # Internal 1D auxiliary grid
    _aux_ez: np.ndarray = field(init=False, repr=False, default=None)
    _aux_hy: np.ndarray = field(init=False, repr=False, default=None)
    _tau: float = field(init=False, default=0.0)
    _t0: float = field(init=False, default=0.0)
    _initialized: bool = field(init=False, default=False)
    
    def __post_init__(self) -> None:
        self._tau = 1.0 / (math.pi * self.frequency_max)
        self._t0 = self.delay_factor * self._tau

    def _initialize(self, nx: int, dx: float, dt: float) -> None:
        """Initialize the 1D auxiliary propagation grid."""
        # 1D grid length covers from x_start to x_end plus PML margin
        self._aux_len = nx
        self._aux_ez = np.zeros(self._aux_len, dtype=np.float64)
        self._aux_hy = np.zeros(self._aux_len, dtype=np.float64)
        self._dx = dx
        self._dt = dt
        self._courant_1d = C_0 * dt / dx
        self._initialized = True
        
    def _step_1d_h(self) -> None:
        """Advance 1D auxiliary H-field to n+1/2."""
        ch = self._dt / (MU_0 * self._dx)
        self._aux_hy[:-1] += ch * (self._aux_ez[1:] - self._aux_ez[:-1])
        
    def _step_1d_e(self, step: int) -> None:
        """Advance 1D auxiliary E-field to n+1."""
        ce = self._dt / (EPS_0 * self._dx)
        self._aux_ez[1:] += ce * (self._aux_hy[1:] - self._aux_hy[:-1])
        
        # Inject hard source at x=0
        t = step * self._dt
        self._aux_ez[0] = self.amplitude * math.exp(
            -((t - self._t0) / self._tau) ** 2
        )
        
        # Simple ABC at right boundary
        self._aux_ez[-1] = self._aux_ez[-2]
    
    def apply_tfsf_h(self, hx: np.ndarray, hy: np.ndarray,
                     dx: float, dy: float, dt: float) -> None:
        """Apply TF/SF corrections to 2D H-fields using 1D E^n fields.
        Must be called AFTER 2D H update and BEFORE 1D H update."""
        if not self._initialized:
            self._initialize(hx.shape[0], dx, dt)
            
        xs, xe = self.x_start, self.x_end
        ys, ye = self.y_start, self.y_end
        
        hx[xs:xe, ys - 1] += dt / (MU_0 * dy) * self._aux_ez[xs:xe]
        hx[xs:xe, ye - 1] -= dt / (MU_0 * dy) * self._aux_ez[xs:xe]
        hy[xs - 1, ys:ye] -= dt / (MU_0 * dx) * self._aux_ez[xs]
        hy[xe - 1, ys:ye] += dt / (MU_0 * dx) * self._aux_ez[xe]
        
        # Now advance 1D H to n+1/2
        self._step_1d_h()
        
    def apply_tfsf_e(self, ez: np.ndarray, dx: float, dt: float, step: int) -> None:
        """Apply TF/SF corrections to 2D E-fields using 1D H^{n+1/2} fields.
        Must be called AFTER 2D E update and BEFORE 1D E update."""
        xs, xe = self.x_start, self.x_end
        ys, ye = self.y_start, self.y_end
        
        ez[xs, ys:ye] -= dt / (EPS_0 * dx) * self._aux_hy[xs - 1]
        ez[xe, ys:ye] += dt / (EPS_0 * dx) * self._aux_hy[xe - 1]
        
        # Now advance 1D E to n+1
        self._step_1d_e(step)

    # BaseSource interface (not used for TF/SF, but required)
    def value_at(self, timestep: int, dt: float) -> float:
        return 0.0
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.x_start, self.y_start)
    
    @property
    def component(self) -> str:
        return self.field_component
