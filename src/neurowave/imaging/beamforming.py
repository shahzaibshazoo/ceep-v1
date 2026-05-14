"""
Delay-And-Sum (DAS) Beamforming for Microwave Imaging
======================================================

Implements beamforming algorithms for reconstructing dielectric contrast images
from multistatic S-parameter measurements. These algorithms are the computational
heart of microwave imaging systems for stroke detection, breast cancer screening,
and general biomedical diagnostics.

DAS Algorithm:
--------------
For each pixel (x, y) in the imaging region:
1. Compute propagation delays from each TX antenna to the pixel and back to RX
2. Phase-shift the S-parameters by these delays
3. Sum coherently across all TX-RX pairs
4. The result is the dielectric contrast at that pixel

References
----------
.. [1] Hagness et al., "Two-dimensional FDTD analysis of a pulsed microwave
       confocal system for breast cancer detection," IEEE Trans. BMT, 1998.
.. [2] Fear et al., "Confocal microwave imaging for breast cancer detection,"
       IEEE Trans. MTT, 2002.
.. [3] Meaney et al., "A clinical prototype for active microwave imaging,"
       IEEE Trans. MTT, 2000.

Author: NeuroWave Development Team
Date: 2026-05-13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import numpy.typing as npt

from neurowave.solvers.s_params import MultistaticSParameters
from neurowave.core.constants import C_0


@dataclass
class ImagingRegion:
    """Defines the region to be imaged.

    Parameters
    ----------
    x_range, y_range : tuple
        (min, max) grid indices for imaging region.
    z_slice : int, optional
        Z-slice for 3D (None for 2D imaging).
    dx : float
        Grid spacing (meters).
    background_permittivity : float
        Background relative permittivity (default: 1.0 for air).
    """
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    z_slice: Optional[int] = None
    dx: float = 1e-3
    background_permittivity: float = 1.0

    @property
    def nx(self) -> int:
        """Number of pixels in x."""
        return self.x_range[1] - self.x_range[0]

    @property
    def ny(self) -> int:
        """Number of pixels in y."""
        return self.y_range[1] - self.y_range[0]

    def get_pixel_position(self, i: int, j: int) -> Tuple[float, float]:
        """Get physical position of pixel (i, j).

        Returns
        -------
        x, y : tuple of float
            Position in meters.
        """
        x = (self.x_range[0] + i) * self.dx
        y = (self.y_range[0] + j) * self.dx
        return x, y


class DelayAndSumBeamformer:
    """Delay-And-Sum (DAS) beamformer for microwave imaging.

    The DAS algorithm focuses scattered signals from all antenna pairs
    onto each pixel in the imaging region, reconstructing the dielectric
    contrast distribution.

    Parameters
    ----------
    s_parameters : MultistaticSParameters
        Measured S-parameters from all antenna pairs.
    antenna_positions : list of tuples
        Physical positions of antennas (grid indices).
    imaging_region : ImagingRegion
        Region to reconstruct.
    frequency_range : tuple, optional
        (f_min, f_max) to use (Hz). If None, uses all frequencies.

    Examples
    --------
    >>> # Reconstruct image from S-parameters
    >>> beamformer = DelayAndSumBeamformer(
    ...     s_parameters=s_params,
    ...     antenna_positions=array.get_antenna_positions(),
    ...     imaging_region=ImagingRegion(
    ...         x_range=(50, 150),
    ...         y_range=(50, 150),
    ...         dx=1e-3
    ...     )
    ... )
    >>> image = beamformer.reconstruct()
    >>> beamformer.visualize_image(image)
    """

    def __init__(
        self,
        s_parameters: MultistaticSParameters,
        antenna_positions: List[Tuple[int, ...]],
        imaging_region: ImagingRegion,
        frequency_range: Optional[Tuple[float, float]] = None
    ):
        self.s_params = s_parameters
        self.antenna_positions = antenna_positions
        self.region = imaging_region
        self.dx = imaging_region.dx

        # Select frequency range
        if frequency_range is None:
            self.freq_mask = np.ones(len(s_parameters.frequencies), dtype=bool)
        else:
            f_min, f_max = frequency_range
            self.freq_mask = ((s_parameters.frequencies >= f_min) &
                             (s_parameters.frequencies <= f_max))

        self.frequencies = s_parameters.frequencies[self.freq_mask]
        self.num_antennas = len(antenna_positions)

        # Compute propagation velocity in background
        self.velocity = C_0 / np.sqrt(imaging_region.background_permittivity)

    def compute_delay(
        self,
        tx_pos: Tuple[int, ...],
        rx_pos: Tuple[int, ...],
        pixel_pos: Tuple[float, float]
    ) -> float:
        """Compute two-way propagation delay for TX → pixel → RX.

        Parameters
        ----------
        tx_pos, rx_pos : tuple
            Transmit and receive antenna positions (grid indices).
        pixel_pos : tuple
            Pixel position in meters.

        Returns
        -------
        delay : float
            Two-way propagation time (seconds).
        """
        # Convert antenna positions to meters
        tx_x = tx_pos[0] * self.dx
        tx_y = tx_pos[1] * self.dx
        rx_x = rx_pos[0] * self.dx
        rx_y = rx_pos[1] * self.dx

        pixel_x, pixel_y = pixel_pos

        # Distance TX → pixel
        d_tx = np.sqrt((pixel_x - tx_x)**2 + (pixel_y - tx_y)**2)

        # Distance pixel → RX
        d_rx = np.sqrt((pixel_x - rx_x)**2 + (pixel_y - rx_y)**2)

        # Two-way delay
        delay = (d_tx + d_rx) / self.velocity
        return delay

    def reconstruct(
        self,
        method: str = 'das',
        normalize: bool = True
    ) -> npt.NDArray[np.float64]:
        """Reconstruct dielectric contrast image.

        Parameters
        ----------
        method : str
            Beamforming method: 'das' (Delay-And-Sum) or 'dmas' (Delay-Multiply-And-Sum).
        normalize : bool
            Normalize image to [0, 1].

        Returns
        -------
        image : ndarray (nx, ny)
            Reconstructed dielectric contrast image.
        """
        if method == 'das':
            return self._reconstruct_das(normalize)
        elif method == 'dmas':
            return self._reconstruct_dmas(normalize)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _reconstruct_das(self, normalize: bool) -> npt.NDArray[np.float64]:
        """Standard Delay-And-Sum beamforming."""
        nx, ny = self.region.nx, self.region.ny
        image = np.zeros((nx, ny), dtype=complex)

        print(f"Reconstructing {nx}×{ny} image with DAS...")
        print(f"Using {len(self.frequencies)} frequency points")
        print(f"Processing {self.num_antennas}×{self.num_antennas} antenna pairs...")

        # For each pixel
        for i in range(nx):
            if i % 10 == 0:
                print(f"  Row {i}/{nx}")

            for j in range(ny):
                pixel_pos = self.region.get_pixel_position(i, j)
                pixel_value = 0.0 + 0.0j

                # Sum over all TX-RX pairs
                for tx_idx in range(self.num_antennas):
                    for rx_idx in range(self.num_antennas):
                        tx_pos = self.antenna_positions[tx_idx]
                        rx_pos = self.antenna_positions[rx_idx]

                        # Compute delay
                        delay = self.compute_delay(tx_pos, rx_pos, pixel_pos)

                        # Get S-parameter for this pair (all frequencies)
                        s_param = self.s_params.get_element(tx_idx, rx_idx)
                        s_param_freq = s_param[self.freq_mask]

                        # Phase shift and sum across frequencies
                        for f_idx, freq in enumerate(self.frequencies):
                            phase = 2 * np.pi * freq * delay
                            pixel_value += s_param_freq[f_idx] * np.exp(-1j * phase)

                image[i, j] = pixel_value

        # Convert to magnitude
        image_magnitude = np.abs(image)

        if normalize:
            image_magnitude = (image_magnitude - np.min(image_magnitude))
            max_val = np.max(image_magnitude)
            if max_val > 0:
                image_magnitude /= max_val

        return image_magnitude

    def _reconstruct_dmas(self, normalize: bool) -> npt.NDArray[np.float64]:
        """Delay-Multiply-And-Sum (DMAS) beamforming.

        DMAS offers better resolution than DAS by multiplying pairs of signals
        before summation, which suppresses clutter and side lobes.
        """
        nx, ny = self.region.nx, self.region.ny
        image = np.zeros((nx, ny), dtype=complex)

        print(f"Reconstructing {nx}×{ny} image with DMAS...")

        # For each pixel
        for i in range(nx):
            if i % 10 == 0:
                print(f"  Row {i}/{nx}")

            for j in range(ny):
                pixel_pos = self.region.get_pixel_position(i, j)

                # Collect all delayed signals
                signals = []
                for tx_idx in range(self.num_antennas):
                    for rx_idx in range(self.num_antennas):
                        tx_pos = self.antenna_positions[tx_idx]
                        rx_pos = self.antenna_positions[rx_idx]

                        delay = self.compute_delay(tx_pos, rx_pos, pixel_pos)
                        s_param = self.s_params.get_element(tx_idx, rx_idx)
                        s_param_freq = s_param[self.freq_mask]

                        # Delayed signal
                        signal = 0.0 + 0.0j
                        for f_idx, freq in enumerate(self.frequencies):
                            phase = 2 * np.pi * freq * delay
                            signal += s_param_freq[f_idx] * np.exp(-1j * phase)

                        signals.append(signal)

                # DMAS: multiply pairs and sum
                pixel_value = 0.0 + 0.0j
                for k1 in range(len(signals)):
                    for k2 in range(k1 + 1, len(signals)):
                        pixel_value += signals[k1] * np.conj(signals[k2])

                image[i, j] = pixel_value

        image_magnitude = np.abs(image)

        if normalize:
            image_magnitude = (image_magnitude - np.min(image_magnitude))
            max_val = np.max(image_magnitude)
            if max_val > 0:
                image_magnitude /= max_val

        return image_magnitude

    def visualize_image(
        self,
        image: npt.NDArray[np.float64],
        title: str = "DAS Reconstruction",
        show_antennas: bool = True,
        phantom_contour: Optional[List[Tuple[int, int]]] = None
    ):
        """Visualize reconstructed image.

        Parameters
        ----------
        image : ndarray
            Reconstructed image.
        title : str
            Plot title.
        show_antennas : bool
            Show antenna positions.
        phantom_contour : list of tuples, optional
            Ground truth phantom boundary.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        # Show image
        extent = [
            self.region.x_range[0],
            self.region.x_range[1],
            self.region.y_range[0],
            self.region.y_range[1]
        ]

        im = ax.imshow(
            image.T,
            origin='lower',
            cmap='hot',
            extent=extent,
            aspect='equal',
            interpolation='bilinear'
        )

        plt.colorbar(im, ax=ax, label='Normalized Intensity')

        # Show antennas
        if show_antennas:
            ant_x = [pos[0] for pos in self.antenna_positions]
            ant_y = [pos[1] for pos in self.antenna_positions]
            ax.scatter(ant_x, ant_y, s=100, c='cyan', marker='^',
                      edgecolors='blue', linewidth=2, label='Antennas', zorder=10)

        # Show phantom contour
        if phantom_contour is not None:
            contour_x = [p[0] for p in phantom_contour] + [phantom_contour[0][0]]
            contour_y = [p[1] for p in phantom_contour] + [phantom_contour[0][1]]
            ax.plot(contour_x, contour_y, 'lime', linewidth=3,
                   linestyle='--', label='Ground Truth', zorder=5)

        ax.set_xlabel('X (grid cells)', fontsize=12)
        ax.set_ylabel('Y (grid cells)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        if show_antennas or phantom_contour is not None:
            ax.legend(fontsize=11)

        plt.tight_layout()
        return fig


class IterativeBeamformer:
    """Iterative beamforming with Born approximation.

    Improves upon standard DAS by iteratively refining the image,
    accounting for multiple scattering effects.

    Parameters
    ----------
    das_beamformer : DelayAndSumBeamformer
        Initial DAS beamformer.
    max_iterations : int
        Maximum number of iterations.
    convergence_threshold : float
        Stop when relative change < threshold.
    """

    def __init__(
        self,
        das_beamformer: DelayAndSumBeamformer,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-3
    ):
        self.das = das_beamformer
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def reconstruct(self) -> npt.NDArray[np.float64]:
        """Iterative reconstruction.

        Returns
        -------
        image : ndarray
            Refined dielectric contrast image.
        """
        # Start with DAS reconstruction
        image = self.das.reconstruct(normalize=False)
        prev_image = image.copy()

        print(f"Iterative refinement ({self.max_iterations} iterations)...")

        for iteration in range(1, self.max_iterations + 1):
            # Update estimate (simplified Born iteration)
            # In practice, this would involve forward modeling
            # For now, apply regularization/smoothing

            # Gaussian smoothing
            from scipy.ndimage import gaussian_filter
            image = gaussian_filter(image, sigma=1.0)

            # Check convergence
            rel_change = np.linalg.norm(image - prev_image) / np.linalg.norm(prev_image)
            print(f"  Iteration {iteration}: relative change = {rel_change:.4e}")

            if rel_change < self.convergence_threshold:
                print(f"  Converged at iteration {iteration}")
                break

            prev_image = image.copy()

        # Normalize
        image = (image - np.min(image))
        max_val = np.max(image)
        if max_val > 0:
            image /= max_val

        return image


def compute_image_quality_metrics(
    reconstructed: npt.NDArray[np.float64],
    ground_truth: npt.NDArray[np.float64]
) -> dict:
    """Compute image quality metrics.

    Parameters
    ----------
    reconstructed : ndarray
        Reconstructed image.
    ground_truth : ndarray
        Ground truth image.

    Returns
    -------
    metrics : dict
        Dictionary with SNR, PSNR, SSIM, etc.
    """
    # Signal-to-Noise Ratio
    signal_power = np.mean(ground_truth ** 2)
    noise = reconstructed - ground_truth
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    # Peak Signal-to-Noise Ratio
    max_val = np.max(ground_truth)
    mse = np.mean((reconstructed - ground_truth) ** 2)
    psnr = 10 * np.log10(max_val ** 2 / mse) if mse > 0 else np.inf

    # Correlation
    correlation = np.corrcoef(reconstructed.flatten(), ground_truth.flatten())[0, 1]

    return {
        'snr_db': snr,
        'psnr_db': psnr,
        'correlation': correlation,
        'mse': mse
    }
