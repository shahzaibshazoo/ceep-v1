"""
Discrete Fourier Transform (DFT) monitors.

Calculates the running DFT of time-domain fields at specified frequencies
during the FDTD simulation loop. This is critical for extracting S-parameters,
impedance, and near-to-far-field transformations without storing massive
time-history arrays.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple
import numpy as np
import numpy.typing as npt


class DFTMonitor:
    """Computes running Discrete Fourier Transform of fields.
    
    The monitor runs concurrently with the FDTD time-stepping, computing:
        E(ω) = Σ E(nΔt) exp(-jω nΔt) Δt
        
    Parameters
    ----------
    frequencies : array_like
        List of frequencies (Hz) to extract.
    region : slice or tuple of slices or tuple of ints
        The spatial region of the grid to monitor. Can be a single point
        like (50, 50) or a slice like (slice(10, 20), slice(30, 40)).
    component : str
        The field component to monitor (e.g., 'Ez', 'Hx', 'Ex').
    """

    def __init__(
        self,
        frequencies: Union[float, list[float], npt.NDArray[np.float64]],
        region: Union[Tuple[int, int], Tuple[slice, slice]],
        component: str = "Ez",
    ):
        self.frequencies = np.atleast_1d(frequencies).astype(np.float64)
        self.region = region
        self.component = component
        self.omega = 2.0 * np.pi * self.frequencies
        
        self.dft_real: Optional[np.ndarray] = None
        self.dft_imag: Optional[np.ndarray] = None
        self._initialized = False

    def _initialize(self, shape: tuple) -> None:
        num_f = len(self.frequencies)
        self.dft_real = np.zeros((num_f, *shape), dtype=np.float64)
        self.dft_imag = np.zeros((num_f, *shape), dtype=np.float64)
        self._initialized = True

    def update(self, field: np.ndarray, step: int, dt: float) -> None:
        """Update the running DFT with the latest field values.
        
        Parameters
        ----------
        field : ndarray
            The full grid array for the monitored component.
        step : int
            Current timestep number (n).
        dt : float
            Timestep size (Δt).
        """
        # Extract the region. If single point, field_region is scalar,
        # but we handle it uniformly via broadcasting.
        field_region = field[self.region]
        
        if not self._initialized:
            shape = np.shape(field_region)
            self._initialize(shape)
            
        time = step * dt
        
        # exp(-jωt) = cos(ωt) - j sin(ωt)
        cos_wt = np.cos(self.omega * time)
        sin_wt = -np.sin(self.omega * time)
        
        # Reshape for broadcasting depending on spatial dimensions
        # If region is a single point, shape is ()
        # If region is 1D slice, shape is (N,)
        # If region is 2D slice, shape is (Nx, Ny)
        target_shape = [len(self.frequencies)] + [1] * len(np.shape(field_region))
        cos_wt = cos_wt.reshape(target_shape)
        sin_wt = sin_wt.reshape(target_shape)
        
        self.dft_real += cos_wt * field_region * dt
        self.dft_imag += sin_wt * field_region * dt
        
    def get_complex_field(self) -> np.ndarray:
        """Get the accumulated complex field array.
        
        Returns
        -------
        ndarray
            Complex array of shape (num_frequencies, *region_shape).
            If a single frequency was given, the first dimension is 1.
        """
        if not self._initialized:
            raise RuntimeError("Monitor has not received any updates yet.")
        return self.dft_real + 1j * self.dft_imag
