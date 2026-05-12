"""
Physical and mathematical constants for electromagnetic simulation.

All constants are in SI units unless otherwise noted.

References
----------
NIST CODATA 2018 recommended values:
https://physics.nist.gov/cuu/Constants/
"""

import math
from typing import Final


# =============================================================
# Fundamental Physical Constants (SI)
# =============================================================

#: Speed of light in vacuum [m/s]
C_0: Final[float] = 299_792_458.0

#: Permittivity of free space [F/m]
EPS_0: Final[float] = 8.854187817e-12

#: Permeability of free space [H/m]
MU_0: Final[float] = 4.0 * math.pi * 1e-7

#: Impedance of free space [Ω]
ETA_0: Final[float] = math.sqrt(MU_0 / EPS_0)

#: Boltzmann constant [J/K]
K_B: Final[float] = 1.380649e-23

# =============================================================
# Derived Constants
# =============================================================

#: 1 / (speed of light squared) [s²/m²]
INV_C0_SQ: Final[float] = 1.0 / (C_0 * C_0)

#: 2π
TWO_PI: Final[float] = 2.0 * math.pi

# =============================================================
# Default Simulation Parameters
# =============================================================

#: Default Courant number for 2D simulations (must be ≤ 1/√2 ≈ 0.7071)
DEFAULT_COURANT_2D: Final[float] = 0.5

#: Default Courant number for 3D simulations (must be ≤ 1/√3 ≈ 0.5774)
DEFAULT_COURANT_3D: Final[float] = 0.5

#: Default PML thickness in grid cells
DEFAULT_PML_THICKNESS: Final[int] = 10

#: Default PML polynomial grading order
DEFAULT_PML_ORDER: Final[int] = 3

#: Default PML maximum conductivity scaling factor
DEFAULT_PML_SIGMA_MAX_FACTOR: Final[float] = 0.8


def wavelength_to_frequency(wavelength: float) -> float:
    """Convert wavelength [m] to frequency [Hz].

    Parameters
    ----------
    wavelength : float
        Wavelength in meters.

    Returns
    -------
    float
        Frequency in Hz.

    Notes
    -----
    Uses the relation: f = c₀ / λ
    """
    return C_0 / wavelength


def frequency_to_wavelength(frequency: float) -> float:
    """Convert frequency [Hz] to wavelength [m].

    Parameters
    ----------
    frequency : float
        Frequency in Hz.

    Returns
    -------
    float
        Wavelength in meters.

    Notes
    -----
    Uses the relation: λ = c₀ / f
    """
    return C_0 / frequency


def cfl_timestep_2d(dx: float, dy: float, courant: float = DEFAULT_COURANT_2D) -> float:
    """Calculate the maximum stable timestep for 2D FDTD.

    Parameters
    ----------
    dx : float
        Grid spacing in x-direction [m].
    dy : float
        Grid spacing in y-direction [m].
    courant : float, optional
        Courant number (must be ≤ 1/√2 for stability).
        Default is 0.5.

    Returns
    -------
    float
        Maximum stable timestep [s].

    Notes
    -----
    CFL condition for 2D FDTD:
        Δt ≤ 1 / (c₀ · √(1/Δx² + 1/Δy²))

    The Courant number S relates the actual timestep to the maximum:
        Δt = S · Δt_max

    Stability requires S ≤ 1/√2 ≈ 0.7071 in 2D.
    A typical choice is S = 0.5 for a good balance of stability and accuracy.

    References
    ----------
    .. [1] A. Taflove and S. C. Hagness, "Computational Electrodynamics,"
           3rd ed., Artech House, 2005, Chapter 4.
    """
    dt_max = 1.0 / (C_0 * math.sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy)))
    return courant * dt_max


def cfl_timestep_3d(
    dx: float, dy: float, dz: float, courant: float = DEFAULT_COURANT_3D
) -> float:
    """Calculate the maximum stable timestep for 3D FDTD.

    Parameters
    ----------
    dx : float
        Grid spacing in x-direction [m].
    dy : float
        Grid spacing in y-direction [m].
    dz : float
        Grid spacing in z-direction [m].
    courant : float, optional
        Courant number (must be ≤ 1/√3 for stability).
        Default is 0.5.

    Returns
    -------
    float
        Maximum stable timestep [s].

    Notes
    -----
    CFL condition for 3D FDTD:
        Δt ≤ 1 / (c₀ · √(1/Δx² + 1/Δy² + 1/Δz²))

    Stability requires S ≤ 1/√3 ≈ 0.5774 in 3D.
    """
    dt_max = 1.0 / (
        C_0 * math.sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))
    )
    return courant * dt_max
