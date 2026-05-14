"""
Near-to-Far Field (NTFF) Transformation
========================================

Implements the equivalence principle for computing far-field radiation patterns
from near-field FDTD data. This is essential for antenna analysis, scattering
problems, and radiation pattern characterization.

The method uses a virtual closed surface surrounding the radiating source,
records tangential E and H fields on this surface, then applies the equivalence
principle to compute the far-field pattern.

Theory
------
The equivalence principle states that fields outside a closed surface can be
computed from equivalent electric and magnetic currents on that surface:

    J_s = n × H    (equivalent electric surface current)
    M_s = -n × E   (equivalent magnetic surface current)

The far-field (r → ∞) radiated by these currents is:

    E_far(θ,φ) = (e^(-jkr) / r) · [jωμ₀/(4π)] · ∫∫_S [J_s - (n·J_s)n] e^(jk·r') dS'

References
----------
.. [1] Taflove & Hagness, "Computational Electrodynamics," 3rd ed., Ch. 8.
.. [2] Balanis, "Antenna Theory: Analysis and Design," 3rd ed., Ch. 12.
.. [3] Sullivan, "Electromagnetic Simulation Using the FDTD Method," Ch. 10.

Author: NeuroWave Development Team
Date: 2026-05-13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt

from neurowave.core.constants import C_0, MU_0, EPS_0


@dataclass
class NTFFSurface:
    """Defines a virtual surface for near-to-far field transformation.

    Parameters
    ----------
    x_min, x_max : int
        X-direction bounds (grid indices).
    y_min, y_max : int
        Y-direction bounds (grid indices).
    z_min, z_max : int, optional
        Z-direction bounds for 3D (None for 2D).
    """
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: Optional[int] = None
    z_max: Optional[int] = None

    @property
    def is_2d(self) -> bool:
        """True if this is a 2D surface (z bounds are None)."""
        return self.z_min is None and self.z_max is None


class NearToFarField:
    """Near-to-far field transformer using equivalence principle.

    This class records E and H fields on a virtual closed surface during FDTD
    simulation, then computes the far-field radiation pattern at arbitrary
    observation angles.

    For 2D simulations (TMz or TEz), the surface is a closed rectangle.
    For 3D simulations, the surface is a closed box (6 faces).

    Parameters
    ----------
    surface : NTFFSurface
        Virtual surface definition.
    dx, dy, dz : float
        Grid spacing in each direction (meters).
    frequency : float
        Operating frequency for far-field computation (Hz).
    mode : str
        Simulation mode ('TMz', 'TEz', or '3D').

    Examples
    --------
    >>> # 2D TMz antenna radiation pattern
    >>> surface = NTFFSurface(x_min=20, x_max=180, y_min=20, y_max=180)
    >>> ntff = NearToFarField(surface, dx=1e-3, dy=1e-3, frequency=10e9, mode='TMz')
    >>>
    >>> # Record fields during FDTD
    >>> for step in range(num_steps):
    >>>     solver.step()
    >>>     ntff.record_fields(solver.grid, step, dt)
    >>>
    >>> # Compute far-field pattern
    >>> angles = np.linspace(0, 2*np.pi, 360)
    >>> E_far = ntff.compute_far_field_2d(angles)
    """

    def __init__(
        self,
        surface: NTFFSurface,
        dx: float,
        dy: float,
        dz: Optional[float],
        frequency: float,
        mode: str = 'TMz'
    ):
        self.surface = surface
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.frequency = frequency
        self.mode = mode
        self.omega = 2 * np.pi * frequency
        self.k0 = self.omega / C_0  # Free-space wavenumber
        self.wavelength = C_0 / frequency

        # Storage for time-domain surface fields (DFT accumulation)
        self._init_storage()

        self.num_samples = 0

    def _init_storage(self):
        """Initialize storage arrays for surface fields."""
        s = self.surface

        if self.mode in ('TMz', 'TEz'):
            # 2D: 4 edges (x_min, x_max, y_min, y_max)
            nx = s.x_max - s.x_min + 1
            ny = s.y_max - s.y_min + 1

            # Tangential E and H on each edge (complex for DFT)
            self.e_xmin = np.zeros(ny, dtype=complex)  # E_z on x=x_min edge
            self.h_xmin = np.zeros(ny, dtype=complex)  # H_y on x=x_min edge

            self.e_xmax = np.zeros(ny, dtype=complex)
            self.h_xmax = np.zeros(ny, dtype=complex)

            self.e_ymin = np.zeros(nx, dtype=complex)  # E_z on y=y_min edge
            self.h_ymin = np.zeros(nx, dtype=complex)  # H_x on y=y_min edge

            self.e_ymax = np.zeros(nx, dtype=complex)
            self.h_ymax = np.zeros(nx, dtype=complex)

        else:  # 3D
            # 6 faces (x_min, x_max, y_min, y_max, z_min, z_max)
            # Each face stores Ex, Ey, Ez, Hx, Hy, Hz
            nx = s.x_max - s.x_min + 1
            ny = s.y_max - s.y_min + 1
            nz = s.z_max - s.z_min + 1 if s.z_max is not None else 1

            # x-normal faces (yz plane)
            self.ex_xmin = np.zeros((ny, nz), dtype=complex)
            self.ey_xmin = np.zeros((ny, nz), dtype=complex)
            self.ez_xmin = np.zeros((ny, nz), dtype=complex)
            self.hx_xmin = np.zeros((ny, nz), dtype=complex)
            self.hy_xmin = np.zeros((ny, nz), dtype=complex)
            self.hz_xmin = np.zeros((ny, nz), dtype=complex)

            self.ex_xmax = np.zeros((ny, nz), dtype=complex)
            self.ey_xmax = np.zeros((ny, nz), dtype=complex)
            self.ez_xmax = np.zeros((ny, nz), dtype=complex)
            self.hx_xmax = np.zeros((ny, nz), dtype=complex)
            self.hy_xmax = np.zeros((ny, nz), dtype=complex)
            self.hz_xmax = np.zeros((ny, nz), dtype=complex)

            # y-normal faces (xz plane)
            self.ex_ymin = np.zeros((nx, nz), dtype=complex)
            self.ey_ymin = np.zeros((nx, nz), dtype=complex)
            self.ez_ymin = np.zeros((nx, nz), dtype=complex)
            self.hx_ymin = np.zeros((nx, nz), dtype=complex)
            self.hy_ymin = np.zeros((nx, nz), dtype=complex)
            self.hz_ymin = np.zeros((nx, nz), dtype=complex)

            self.ex_ymax = np.zeros((nx, nz), dtype=complex)
            self.ey_ymax = np.zeros((nx, nz), dtype=complex)
            self.ez_ymax = np.zeros((nx, nz), dtype=complex)
            self.hx_ymax = np.zeros((nx, nz), dtype=complex)
            self.hy_ymax = np.zeros((nx, nz), dtype=complex)
            self.hz_ymax = np.zeros((nx, nz), dtype=complex)

            # z-normal faces (xy plane)
            self.ex_zmin = np.zeros((nx, ny), dtype=complex)
            self.ey_zmin = np.zeros((nx, ny), dtype=complex)
            self.ez_zmin = np.zeros((nx, ny), dtype=complex)
            self.hx_zmin = np.zeros((nx, ny), dtype=complex)
            self.hy_zmin = np.zeros((nx, ny), dtype=complex)
            self.hz_zmin = np.zeros((nx, ny), dtype=complex)

            self.ex_zmax = np.zeros((nx, ny), dtype=complex)
            self.ey_zmax = np.zeros((nx, ny), dtype=complex)
            self.ez_zmax = np.zeros((nx, ny), dtype=complex)
            self.hx_zmax = np.zeros((nx, ny), dtype=complex)
            self.hy_zmax = np.zeros((nx, ny), dtype=complex)
            self.hz_zmax = np.zeros((nx, ny), dtype=complex)

    def record_fields(self, grid, timestep: int, dt: float):
        """Record fields on the virtual surface (running DFT).

        This should be called every timestep during FDTD simulation.

        Parameters
        ----------
        grid : Grid2D or Grid3D
            FDTD grid containing field arrays.
        timestep : int
            Current timestep number.
        dt : float
            Timestep size (seconds).
        """
        t = timestep * dt
        phase = np.exp(-1j * self.omega * t) * dt  # DFT phase factor

        s = self.surface

        if self.mode == 'TMz':
            # TMz: Ez, Hx, Hy are the active fields
            # x_min edge (outward normal: -x)
            self.e_xmin += grid.ez[s.x_min, s.y_min:s.y_max+1] * phase
            self.h_xmin += grid.hy[s.x_min, s.y_min:s.y_max+1] * phase

            # x_max edge (outward normal: +x)
            self.e_xmax += grid.ez[s.x_max, s.y_min:s.y_max+1] * phase
            self.h_xmax += grid.hy[s.x_max, s.y_min:s.y_max+1] * phase

            # y_min edge (outward normal: -y)
            self.e_ymin += grid.ez[s.x_min:s.x_max+1, s.y_min] * phase
            self.h_ymin += grid.hx[s.x_min:s.x_max+1, s.y_min] * phase

            # y_max edge (outward normal: +y)
            self.e_ymax += grid.ez[s.x_min:s.x_max+1, s.y_max] * phase
            self.h_ymax += grid.hx[s.x_min:s.x_max+1, s.y_max] * phase

        elif self.mode == 'TEz':
            # TEz: Hz, Ex, Ey are the active fields
            # Similar recording for TEz mode
            self.e_xmin += grid.ey[s.x_min, s.y_min:s.y_max+1] * phase
            self.h_xmin += grid.hz[s.x_min, s.y_min:s.y_max+1] * phase

            self.e_xmax += grid.ey[s.x_max, s.y_min:s.y_max+1] * phase
            self.h_xmax += grid.hz[s.x_max, s.y_min:s.y_max+1] * phase

            self.e_ymin += grid.ex[s.x_min:s.x_max+1, s.y_min] * phase
            self.h_ymin += grid.hz[s.x_min:s.x_max+1, s.y_min] * phase

            self.e_ymax += grid.ex[s.x_min:s.x_max+1, s.y_max] * phase
            self.h_ymax += grid.hz[s.x_min:s.x_max+1, s.y_max] * phase

        else:  # 3D
            # Record all 6 field components on each of 6 faces
            # x_min face
            self.ex_xmin += grid.ex[s.x_min, s.y_min:s.y_max+1, s.z_min:s.z_max+1] * phase
            self.ey_xmin += grid.ey[s.x_min, s.y_min:s.y_max+1, s.z_min:s.z_max+1] * phase
            self.ez_xmin += grid.ez[s.x_min, s.y_min:s.y_max+1, s.z_min:s.z_max+1] * phase
            self.hx_xmin += grid.hx[s.x_min, s.y_min:s.y_max+1, s.z_min:s.z_max+1] * phase
            self.hy_xmin += grid.hy[s.x_min, s.y_min:s.y_max+1, s.z_min:s.z_max+1] * phase
            self.hz_xmin += grid.hz[s.x_min, s.y_min:s.y_max+1, s.z_min:s.z_max+1] * phase

            # Similar for other 5 faces (x_max, y_min, y_max, z_min, z_max)
            # ... (omitted for brevity, same pattern)

        self.num_samples += 1

    def compute_far_field_2d(
        self,
        angles: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]:
        """Compute 2D far-field pattern (TMz or TEz mode).

        Parameters
        ----------
        angles : ndarray
            Observation angles (radians, 0 to 2π).

        Returns
        -------
        E_far : ndarray (complex)
            Far-field pattern (complex amplitude) at each angle.
            For TMz: E_z component.
            For TEz: H_z component (scaled to field units).
        """
        E_far = np.zeros_like(angles, dtype=complex)

        s = self.surface

        # Convert indices to physical positions
        x_positions = np.arange(s.x_min, s.x_max + 1) * self.dx
        y_positions = np.arange(s.y_min, s.y_max + 1) * self.dy

        # Far-field integral: sum contributions from all edges
        for i, phi in enumerate(angles):
            # Unit vector in observation direction
            k_hat = np.array([np.cos(phi), np.sin(phi)])

            # Contribution from x_min edge (outward normal: [-1, 0])
            n_hat = np.array([-1.0, 0.0])
            for j, y in enumerate(y_positions):
                r_prime = np.array([s.x_min * self.dx, y])
                # J_s = n × H, M_s = -n × E
                # For TMz: H is in xy-plane, E_z is out-of-plane
                # Contribution: (n × H) - n·(n × H)·n term
                # Simplified for TMz 2D case:
                phase_shift = np.exp(1j * self.k0 * np.dot(k_hat, r_prime))

                # Equivalent current contribution
                # J_s = n × H = [-1, 0] × [H_x, H_y, 0] = [0, 0, H_y]
                # For far field: cross product with propagation direction
                j_eff = self.h_xmin[j] * n_hat[1]  # Effective current
                E_far[i] += j_eff * phase_shift * self.dy

            # Contribution from x_max edge (outward normal: [+1, 0])
            n_hat = np.array([1.0, 0.0])
            for j, y in enumerate(y_positions):
                r_prime = np.array([s.x_max * self.dx, y])
                phase_shift = np.exp(1j * self.k0 * np.dot(k_hat, r_prime))
                j_eff = self.h_xmax[j] * n_hat[1]
                E_far[i] += j_eff * phase_shift * self.dy

            # Contribution from y_min edge (outward normal: [0, -1])
            n_hat = np.array([0.0, -1.0])
            for j, x in enumerate(x_positions):
                r_prime = np.array([x, s.y_min * self.dy])
                phase_shift = np.exp(1j * self.k0 * np.dot(k_hat, r_prime))
                j_eff = -self.h_ymin[j] * n_hat[0]  # -H_x component
                E_far[i] += j_eff * phase_shift * self.dx

            # Contribution from y_max edge (outward normal: [0, +1])
            n_hat = np.array([0.0, 1.0])
            for j, x in enumerate(x_positions):
                r_prime = np.array([x, s.y_max * self.dy])
                phase_shift = np.exp(1j * self.k0 * np.dot(k_hat, r_prime))
                j_eff = -self.h_ymax[j] * n_hat[0]
                E_far[i] += j_eff * phase_shift * self.dx

        # Apply prefactor: jωμ₀/(4π)
        prefactor = 1j * self.omega * MU_0 / (4 * np.pi)
        E_far *= prefactor

        return E_far

    def compute_far_field_3d(
        self,
        theta: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        """Compute 3D far-field pattern (theta and phi components).

        Parameters
        ----------
        theta : ndarray
            Polar angle (radians, 0 to π).
        phi : ndarray
            Azimuthal angle (radians, 0 to 2π).

        Returns
        -------
        E_theta, E_phi : ndarray (complex)
            Far-field components in spherical coordinates.
        """
        # 3D far-field computation using equivalence principle
        # This is more complex - requires integration over all 6 faces
        # and proper vector transformations to spherical coordinates

        E_theta = np.zeros_like(theta, dtype=complex)
        E_phi = np.zeros_like(phi, dtype=complex)

        # TODO: Implement full 3D far-field integration
        # For now, return placeholder
        raise NotImplementedError("3D far-field computation coming soon")

        return E_theta, E_phi

    def get_radiation_pattern_db(
        self,
        angles: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Get radiation pattern in dB (normalized to peak).

        Parameters
        ----------
        angles : ndarray
            Observation angles (radians).

        Returns
        -------
        pattern_db : ndarray
            Radiation pattern in dB relative to peak.
        """
        E_far = self.compute_far_field_2d(angles)
        power = np.abs(E_far) ** 2
        power_db = 10 * np.log10(power / np.max(power))
        return power_db

    def get_directivity(self) -> float:
        """Compute directivity (peak gain relative to isotropic).

        Returns
        -------
        directivity : float
            Directivity in linear scale (use 10*log10 for dBi).
        """
        angles = np.linspace(0, 2*np.pi, 360)
        E_far = self.compute_far_field_2d(angles)
        power = np.abs(E_far) ** 2

        # Directivity = 4π * max_power / total_radiated_power
        # For 2D: 2π instead of 4π
        max_power = np.max(power)
        avg_power = np.mean(power)
        directivity = max_power / avg_power  # 2D case

        return directivity


def plot_radiation_pattern(
    ntff: NearToFarField,
    angles: Optional[npt.NDArray[np.float64]] = None,
    polar: bool = True
):
    """Plot radiation pattern from NTFF data.

    Parameters
    ----------
    ntff : NearToFarField
        Near-to-far field transformer with recorded data.
    angles : ndarray, optional
        Observation angles (default: 0 to 2π, 360 points).
    polar : bool
        Use polar plot (True) or Cartesian (False).
    """
    import matplotlib.pyplot as plt

    if angles is None:
        angles = np.linspace(0, 2*np.pi, 360)

    pattern_db = ntff.get_radiation_pattern_db(angles)

    if polar:
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
        ax.plot(angles, pattern_db - np.min(pattern_db), linewidth=2)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim([0, np.max(pattern_db - np.min(pattern_db))])
        ax.set_title('Radiation Pattern (dB)', fontsize=14, pad=20)
        ax.grid(True)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.degrees(angles), pattern_db, linewidth=2)
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Normalized Pattern (dB)', fontsize=12)
        ax.set_title('Radiation Pattern', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 360])

    return fig
