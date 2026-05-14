"""
S-Parameter Extraction for Multistatic Microwave Imaging
=========================================================

Comprehensive utilities for computing scattering parameters (S-parameters)
from FDTD simulations. Supports both traditional two-port measurements
and large N×N multistatic antenna arrays for imaging applications.

S-parameters characterize how electromagnetic waves scatter from objects:
- S11, S22, etc.: Reflection coefficients (monostatic)
- S12, S21, etc.: Transmission coefficients (bistatic)

For imaging, we collect the full N×N S-matrix where N is the number of
antennas. Each element S_ij represents transmission from antenna j to
antenna i.

Author: NeuroWave Development Team
Date: 2026-05-13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import numpy.typing as npt


def extract_s_parameters(
    incident_dft: np.ndarray,
    total_dft: np.ndarray,
    transmitted_dft: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute S11 (reflection) and S21 (transmission) from complex DFT fields.
    
    Parameters
    ----------
    incident_dft : ndarray
        Complex field DFT at port 1 from a reference (free space) simulation.
    total_dft : ndarray
        Complex field DFT at port 1 from the main simulation (scatterer present).
    transmitted_dft : ndarray
        Complex field DFT at port 2 from the main simulation.
        
    Returns
    -------
    s11, s21 : tuple of ndarray
        The complex scattering parameters across the frequencies.
        S11 is the reflection coefficient.
        S21 is the transmission coefficient.
    """
    # S11 = E_reflected / E_incident = (E_tot - E_inc) / E_inc
    reflected = total_dft - incident_dft
    
    # Avoid division by zero for frequencies with negligible energy
    safe_inc = np.where(np.abs(incident_dft) > 1e-15, incident_dft, 1.0)
    
    s11 = reflected / safe_inc
    s21 = transmitted_dft / safe_inc
    
    # Mask out values where incident energy was effectively zero
    mask = np.abs(incident_dft) <= 1e-15
    s11[mask] = 0.0
    s21[mask] = 0.0
    
    return s11, s21


@dataclass
class MultistaticSParameters:
    """Container for multistatic S-parameter measurements.

    Parameters
    ----------
    frequencies : ndarray
        Frequency points (Hz).
    s_matrix : ndarray (num_freq, N, N) complex
        Full S-matrix at each frequency.
        S[f, i, j] = transmission from antenna j to antenna i.
    antenna_positions : list of tuples
        Physical positions of antennas.
    background_subtracted : bool
        Whether background has been subtracted (differential imaging).

    Examples
    --------
    >>> # Extract S-parameters for 16-antenna array
    >>> s_params = MultistaticSParameters(
    ...     frequencies=np.array([1e9, 2e9, 3e9]),
    ...     s_matrix=s_data,  # shape (3, 16, 16)
    ...     antenna_positions=array.get_antenna_positions()
    ... )
    >>> # Get monostatic responses (diagonal)
    >>> monostatic = s_params.get_monostatic()
    >>> # Get bistatic for specific pair
    >>> s_12 = s_params.get_element(tx=1, rx=2, freq_index=0)
    """
    frequencies: npt.NDArray[np.float64]
    s_matrix: npt.NDArray[np.complex128]
    antenna_positions: List[Tuple[int, ...]]
    background_subtracted: bool = False

    @property
    def num_frequencies(self) -> int:
        """Number of frequency points."""
        return len(self.frequencies)

    @property
    def num_antennas(self) -> int:
        """Number of antennas."""
        return self.s_matrix.shape[1]

    def get_element(
        self,
        tx: int,
        rx: int,
        freq_index: Optional[int] = None
    ) -> npt.NDArray[np.complex128]:
        """Get S-parameter for specific TX-RX pair.

        Parameters
        ----------
        tx : int
            Transmit antenna index.
        rx : int
            Receive antenna index.
        freq_index : int, optional
            Frequency index (if None, returns all frequencies).

        Returns
        -------
        s_param : complex or ndarray
            S-parameter value(s).
        """
        if freq_index is None:
            return self.s_matrix[:, rx, tx]
        return self.s_matrix[freq_index, rx, tx]

    def get_monostatic(
        self,
        freq_index: Optional[int] = None
    ) -> npt.NDArray[np.complex128]:
        """Get monostatic S-parameters (diagonal of S-matrix).

        Parameters
        ----------
        freq_index : int, optional
            Frequency index (if None, returns all frequencies).

        Returns
        -------
        monostatic : ndarray
            Diagonal elements S_ii (reflection coefficients).
        """
        if freq_index is None:
            # Return (num_freq, N) array
            return np.array([np.diag(self.s_matrix[f])
                            for f in range(self.num_frequencies)])
        return np.diag(self.s_matrix[freq_index])

    def get_bistatic(
        self,
        exclude_monostatic: bool = True
    ) -> List[Tuple[int, int, npt.NDArray[np.complex128]]]:
        """Get all bistatic S-parameters (off-diagonal elements).

        Parameters
        ----------
        exclude_monostatic : bool
            If True, exclude S_ii terms.

        Returns
        -------
        bistatic : list of (tx, rx, s_param_array)
            All bistatic pairs with their S-parameters across frequencies.
        """
        bistatic = []
        for tx in range(self.num_antennas):
            for rx in range(self.num_antennas):
                if exclude_monostatic and tx == rx:
                    continue
                s_param = self.s_matrix[:, rx, tx]
                bistatic.append((tx, rx, s_param))
        return bistatic

    def to_db(self) -> 'MultistaticSParameters':
        """Convert S-parameters to dB (20*log10|S|).

        Returns
        -------
        s_params_db : MultistaticSParameters
            New object with magnitude in dB.
        """
        s_matrix_db = 20 * np.log10(np.abs(self.s_matrix) + 1e-20)
        return MultistaticSParameters(
            frequencies=self.frequencies,
            s_matrix=s_matrix_db,
            antenna_positions=self.antenna_positions,
            background_subtracted=self.background_subtracted
        )

    def subtract_background(
        self,
        background: 'MultistaticSParameters'
    ) -> 'MultistaticSParameters':
        """Subtract background measurement for differential imaging.

        This removes the direct coupling between antennas and clutter,
        leaving only the response from the target of interest.

        Parameters
        ----------
        background : MultistaticSParameters
            Background measurement (empty chamber or reference).

        Returns
        -------
        differential : MultistaticSParameters
            Differential S-parameters (target response only).
        """
        if not np.array_equal(self.frequencies, background.frequencies):
            raise ValueError("Frequency points must match")

        diff_matrix = self.s_matrix - background.s_matrix

        return MultistaticSParameters(
            frequencies=self.frequencies,
            s_matrix=diff_matrix,
            antenna_positions=self.antenna_positions,
            background_subtracted=True
        )


class MultistaticDataCollector:
    """Orchestrates multistatic S-parameter collection from FDTD.

    This class manages the sequential excitation of each antenna in an array
    and collection of responses at all other antennas, building up the full
    N×N S-matrix.

    Parameters
    ----------
    antenna_array : CircularArray, PlanarArray, or ConformalArray
        Antenna array configuration.
    frequencies : ndarray
        Frequency points for DFT (Hz).
    dx : float
        Grid spacing (meters).

    Examples
    --------
    >>> from ceep.antennas import CircularArray
    >>> array = CircularArray(num_antennas=8, radius_mm=100,
    ...                       center=(100,100), dx=1e-3)
    >>> collector = MultistaticDataCollector(array, frequencies=np.linspace(1e9, 3e9, 20))
    >>>
    >>> # Run FDTD with each antenna as transmitter
    >>> for tx_idx in range(8):
    ...     solver = setup_fdtd_with_tx(tx_idx)
    ...     solver.run()
    ...     collector.record_transmission(tx_idx, solver)
    >>>
    >>> s_params = collector.get_s_parameters()
    """

    def __init__(
        self,
        antenna_array,
        frequencies: npt.NDArray[np.float64],
        dx: float
    ):
        self.antenna_array = antenna_array
        self.frequencies = frequencies
        self.dx = dx
        self.num_antennas = len(antenna_array.get_antenna_positions())
        self.num_frequencies = len(frequencies)

        # Storage for S-matrix: (num_freq, N_rx, N_tx)
        self.s_matrix = np.zeros(
            (self.num_frequencies, self.num_antennas, self.num_antennas),
            dtype=complex
        )

        # Track which transmitters have been measured
        self.measured_tx: List[bool] = [False] * self.num_antennas

    def record_transmission(
        self,
        tx_index: int,
        dft_data: Dict[Tuple[int, ...], npt.NDArray[np.complex128]]
    ) -> None:
        """Record DFT data from one transmit antenna to all receivers.

        Parameters
        ----------
        tx_index : int
            Index of transmitting antenna.
        dft_data : dict
            Dictionary mapping receiver positions to complex DFT values.
            Keys are (x, y) or (x, y, z) tuples.
            Values are complex arrays of shape (num_frequencies,).
        """
        rx_positions = self.antenna_array.get_antenna_positions()

        for rx_index, rx_pos in enumerate(rx_positions):
            if rx_pos in dft_data:
                # Record S[rx, tx] across all frequencies
                self.s_matrix[:, rx_index, tx_index] = dft_data[rx_pos]
            else:
                # Position not found - zero response
                self.s_matrix[:, rx_index, tx_index] = 0.0

        self.measured_tx[tx_index] = True

    def get_s_parameters(self) -> MultistaticSParameters:
        """Get the complete S-parameter dataset.

        Returns
        -------
        s_params : MultistaticSParameters
            Complete multistatic S-parameters.

        Raises
        ------
        RuntimeError
            If not all transmitters have been measured.
        """
        if not all(self.measured_tx):
            missing = [i for i, measured in enumerate(self.measured_tx)
                      if not measured]
            raise RuntimeError(
                f"Not all transmitters measured. Missing: {missing}"
            )

        return MultistaticSParameters(
            frequencies=self.frequencies,
            s_matrix=self.s_matrix,
            antenna_positions=self.antenna_array.get_antenna_positions()
        )

    def is_complete(self) -> bool:
        """Check if all measurements are complete."""
        return all(self.measured_tx)

    def get_progress(self) -> Tuple[int, int]:
        """Get measurement progress.

        Returns
        -------
        completed, total : tuple
            Number of completed measurements and total required.
        """
        return sum(self.measured_tx), self.num_antennas


def calibrate_s_parameters(
    measured: MultistaticSParameters,
    calibration_standard: Optional[MultistaticSParameters] = None,
    method: str = 'short_open_load'
) -> MultistaticSParameters:
    """Apply calibration to measured S-parameters.

    Corrects for systematic errors: cable delays, antenna mismatch,
    mutual coupling, and system gain variations.

    Parameters
    ----------
    measured : MultistaticSParameters
        Raw measured S-parameters.
    calibration_standard : MultistaticSParameters, optional
        Known reference measurement.
    method : str
        Calibration method: 'short_open_load', 'thru', or 'background'.

    Returns
    -------
    calibrated : MultistaticSParameters
        Calibrated S-parameters.
    """
    if method == 'background' and calibration_standard is not None:
        # Simple background subtraction
        return measured.subtract_background(calibration_standard)

    # For more advanced calibration (TRL, SOLT), implement here
    # For now, return measured data
    return measured


def visualize_s_matrix(
    s_params: MultistaticSParameters,
    freq_index: int = 0,
    magnitude: bool = True,
    db_scale: bool = True
):
    """Visualize S-parameter matrix as heatmap.

    Parameters
    ----------
    s_params : MultistaticSParameters
        S-parameter data.
    freq_index : int
        Frequency index to plot.
    magnitude : bool
        Plot magnitude (True) or phase (False).
    db_scale : bool
        Use dB scale for magnitude.
    """
    import matplotlib.pyplot as plt

    s_matrix_freq = s_params.s_matrix[freq_index]

    if magnitude:
        if db_scale:
            data = 20 * np.log10(np.abs(s_matrix_freq) + 1e-20)
            label = '|S| (dB)'
            cmap = 'viridis'
        else:
            data = np.abs(s_matrix_freq)
            label = '|S|'
            cmap = 'hot'
    else:
        data = np.angle(s_matrix_freq)
        label = 'Phase (rad)'
        cmap = 'twilight'

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data, cmap=cmap, aspect='equal', origin='lower')
    plt.colorbar(im, ax=ax, label=label)

    ax.set_xlabel('Transmit Antenna Index', fontsize=12)
    ax.set_ylabel('Receive Antenna Index', fontsize=12)
    ax.set_title(
        f'S-Parameter Matrix at {s_params.frequencies[freq_index]/1e9:.2f} GHz\n'
        f'{s_params.num_antennas}×{s_params.num_antennas} Array',
        fontsize=14, fontweight='bold'
    )

    # Add grid
    ax.set_xticks(np.arange(s_params.num_antennas))
    ax.set_yticks(np.arange(s_params.num_antennas))
    ax.grid(which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_channel_impulse_response(
    s_params: MultistaticSParameters,
    tx: int,
    rx: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute time-domain channel impulse response via inverse FFT.

    Parameters
    ----------
    s_params : MultistaticSParameters
        Frequency-domain S-parameters.
    tx : int
        Transmit antenna index.
    rx : int
        Receive antenna index.

    Returns
    -------
    time, response : tuple of ndarrays
        Time vector (seconds) and impulse response.
    """
    # Get S-parameter across frequency
    s_freq = s_params.get_element(tx, rx)

    # Inverse FFT to time domain
    response_time = np.fft.ifft(s_freq)
    response_time = np.fft.ifftshift(response_time)

    # Time vector
    df = s_params.frequencies[1] - s_params.frequencies[0]
    dt = 1.0 / (len(s_freq) * df)
    time = np.arange(len(s_freq)) * dt

    return time, np.abs(response_time)
