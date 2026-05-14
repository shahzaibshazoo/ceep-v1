"""
Antenna Array Configurations for Microwave Imaging
===================================================

Implements antenna array geometries and feed structures for multistatic
microwave imaging systems:

- Circular arrays (common for head imaging)
- Planar arrays (for breast imaging)
- Conformal arrays (body-fitted geometries)
- Realistic antenna models (dipole, monopole, Vivaldi)

Author: NeuroWave Development Team
Date: 2026-05-13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import numpy.typing as npt

from ceep.sources.waveforms import GaussianSource, ModulatedGaussianSource


@dataclass
class AntennaElement:
    """Single antenna element specification.

    Parameters
    ----------
    position : tuple
        (x, y) for 2D or (x, y, z) for 3D grid coordinates.
    orientation : tuple
        Direction vector (unit vector).
    antenna_type : str
        'dipole', 'monopole', 'vivaldi', or 'patch'.
    feed_point : tuple, optional
        Feed point coordinates (default: same as position).
    """
    position: Tuple[int, ...]
    orientation: Tuple[float, ...]
    antenna_type: str = 'dipole'
    feed_point: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.feed_point is None:
            self.feed_point = self.position

        # Normalize orientation vector
        norm = np.sqrt(sum(o**2 for o in self.orientation))
        self.orientation = tuple(o / norm for o in self.orientation)


class CircularArray:
    """Circular antenna array for head imaging.

    Places N antennas uniformly around a circle. This is the most common
    configuration for stroke detection and brain imaging.

    Parameters
    ----------
    num_antennas : int
        Number of antenna elements.
    radius_mm : float
        Radius of the circular array (millimeters).
    center : tuple
        Center coordinates (x, y) or (x, y, z).
    dx : float
        Grid spacing (meters).
    antenna_type : str
        Type of antenna element.
    polarization : str
        'vertical' (Ez) or 'horizontal' (Ex/Ey).

    Examples
    --------
    >>> # 16-element circular array for head imaging
    >>> array = CircularArray(
    ...     num_antennas=16,
    ...     radius_mm=120,  # 12 cm from head center
    ...     center=(100, 100),
    ...     dx=1e-3,
    ...     antenna_type='monopole'
    ... )
    >>> positions = array.get_antenna_positions()
    >>> sources = array.create_sources(frequency=1e9, bandwidth=500e6)
    """

    def __init__(
        self,
        num_antennas: int,
        radius_mm: float,
        center: Tuple[int, ...],
        dx: float,
        antenna_type: str = 'monopole',
        polarization: str = 'vertical'
    ):
        self.num_antennas = num_antennas
        self.radius = radius_mm * 1e-3  # Convert to meters
        self.center = center
        self.dx = dx
        self.antenna_type = antenna_type
        self.polarization = polarization

        # Compute antenna positions and orientations
        self.elements = self._compute_elements()

    def _compute_elements(self) -> List[AntennaElement]:
        """Compute antenna element positions and orientations."""
        elements = []

        for i in range(self.num_antennas):
            angle = 2 * np.pi * i / self.num_antennas

            # Position on circle
            if len(self.center) == 2:  # 2D
                x_offset = self.radius * np.cos(angle) / self.dx
                y_offset = self.radius * np.sin(angle) / self.dx
                position = (
                    int(self.center[0] + x_offset),
                    int(self.center[1] + y_offset)
                )
                # Orientation points toward center
                orientation = (-np.cos(angle), -np.sin(angle))
            else:  # 3D
                x_offset = self.radius * np.cos(angle) / self.dx
                y_offset = self.radius * np.sin(angle) / self.dx
                position = (
                    int(self.center[0] + x_offset),
                    int(self.center[1] + y_offset),
                    int(self.center[2])
                )
                orientation = (-np.cos(angle), -np.sin(angle), 0.0)

            elements.append(AntennaElement(
                position=position,
                orientation=orientation,
                antenna_type=self.antenna_type
            ))

        return elements

    def get_antenna_positions(self) -> List[Tuple[int, ...]]:
        """Get list of antenna positions (grid indices).

        Returns
        -------
        positions : list of tuples
            Grid coordinates for each antenna.
        """
        return [elem.position for elem in self.elements]

    def create_sources(
        self,
        frequency: float,
        bandwidth: float,
        source_type: str = 'modulated_gaussian',
        delay_factor: float = 5.0
    ) -> List:
        """Create source objects for all antennas.

        Parameters
        ----------
        frequency : float
            Center frequency (Hz).
        bandwidth : float
            Bandwidth (Hz).
        source_type : str
            'gaussian' or 'modulated_gaussian'.
        delay_factor : float
            Delay factor for Gaussian envelope.

        Returns
        -------
        sources : list
            List of source objects for FDTD simulation.
        """
        sources = []

        # Determine field component based on polarization
        if self.polarization == 'vertical':
            field_component = 'Ez'
        else:
            field_component = 'Ex'  # Will need to handle orientation

        for elem in self.elements:
            if source_type == 'modulated_gaussian':
                source = ModulatedGaussianSource(
                    x=elem.position[0],
                    y=elem.position[1],
                    z=elem.position[2] if len(elem.position) == 3 else None,
                    frequency=frequency,
                    bandwidth=bandwidth,
                    field_component=field_component,
                    delay_factor=delay_factor
                )
            else:  # gaussian
                source = GaussianSource(
                    x=elem.position[0],
                    y=elem.position[1],
                    z=elem.position[2] if len(elem.position) == 3 else None,
                    frequency_max=frequency,
                    field_component=field_component,
                    delay_factor=delay_factor
                )

            sources.append(source)

        return sources

    def get_transmit_receive_pairs(self) -> List[Tuple[int, int]]:
        """Get all transmit-receive antenna pairs for multistatic imaging.

        Returns
        -------
        pairs : list of (tx_idx, rx_idx) tuples
            All unique TX-RX pairs including monostatic (i, i).
        """
        pairs = []
        for tx in range(self.num_antennas):
            for rx in range(self.num_antennas):
                pairs.append((tx, rx))
        return pairs

    def compute_array_factor(
        self,
        frequency: float,
        angles: npt.NDArray[np.float64],
        weights: Optional[npt.NDArray[np.complex128]] = None
    ) -> npt.NDArray[np.complex128]:
        """Compute array factor for beam steering.

        Parameters
        ----------
        frequency : float
            Operating frequency (Hz).
        angles : ndarray
            Azimuth angles in radians (0 to 2π).
        weights : ndarray, optional
            Complex weights for each antenna (default: uniform).

        Returns
        -------
        array_factor : ndarray (complex)
            Array factor as function of angle.
        """
        from ceep.core.constants import C_0

        wavelength = C_0 / frequency
        k = 2 * np.pi / wavelength

        if weights is None:
            weights = np.ones(self.num_antennas, dtype=complex)

        AF = np.zeros_like(angles, dtype=complex)

        for i, elem in enumerate(self.elements):
            # Antenna position in 2D plane
            if len(elem.position) == 2:
                x, y = elem.position
            else:
                x, y, _ = elem.position

            x_m = (x - self.center[0]) * self.dx
            y_m = (y - self.center[1]) * self.dx

            # Phase contribution from this antenna
            for j, phi in enumerate(angles):
                phase = k * (x_m * np.cos(phi) + y_m * np.sin(phi))
                AF[j] += weights[i] * np.exp(1j * phase)

        return AF


class PlanarArray:
    """Planar antenna array for breast imaging.

    Rectangular grid of antennas above the imaging region.

    Parameters
    ----------
    nx_antennas, ny_antennas : int
        Number of antennas in x and y directions.
    spacing_mm : float
        Antenna spacing (millimeters).
    corner : tuple
        Bottom-left corner coordinates (grid indices).
    dx : float
        Grid spacing (meters).
    antenna_type : str
        Type of antenna element.

    Examples
    --------
    >>> # 8×8 planar array for breast imaging
    >>> array = PlanarArray(
    ...     nx_antennas=8,
    ...     ny_antennas=8,
    ...     spacing_mm=20,
    ...     corner=(50, 50),
    ...     dx=1e-3
    ... )
    """

    def __init__(
        self,
        nx_antennas: int,
        ny_antennas: int,
        spacing_mm: float,
        corner: Tuple[int, int],
        dx: float,
        antenna_type: str = 'patch',
        height_above_mm: float = 10.0
    ):
        self.nx_antennas = nx_antennas
        self.ny_antennas = ny_antennas
        self.spacing = spacing_mm * 1e-3  # Convert to meters
        self.corner = corner
        self.dx = dx
        self.antenna_type = antenna_type
        self.height_above = height_above_mm * 1e-3

        self.elements = self._compute_elements()

    def _compute_elements(self) -> List[AntennaElement]:
        """Compute planar array element positions."""
        elements = []

        for i in range(self.nx_antennas):
            for j in range(self.ny_antennas):
                x_offset = i * self.spacing / self.dx
                y_offset = j * self.spacing / self.dx
                z_offset = self.height_above / self.dx

                position = (
                    int(self.corner[0] + x_offset),
                    int(self.corner[1] + y_offset),
                    int(self.corner[2] + z_offset) if len(self.corner) == 3 else 0
                )

                # Orientation points downward (toward imaging region)
                orientation = (0.0, 0.0, -1.0)

                elements.append(AntennaElement(
                    position=position,
                    orientation=orientation,
                    antenna_type=self.antenna_type
                ))

        return elements

    def get_antenna_positions(self) -> List[Tuple[int, ...]]:
        """Get list of antenna positions."""
        return [elem.position for elem in self.elements]


class ConformalArray:
    """Conformal antenna array that follows body contours.

    Useful for wearable monitoring systems and body-fitted imaging.

    Parameters
    ----------
    contour_points : list of tuples
        List of (x, y) or (x, y, z) points defining the surface contour.
    num_antennas : int
        Number of antennas to place along contour.
    dx : float
        Grid spacing (meters).
    antenna_type : str
        Type of antenna element.

    Examples
    --------
    >>> # Create elliptical conformal array
    >>> angles = np.linspace(0, 2*np.pi, 100)
    >>> contour = [(int(100 + 50*np.cos(a)), int(100 + 40*np.sin(a)))
    ...            for a in angles]
    >>> array = ConformalArray(contour, num_antennas=12, dx=1e-3)
    """

    def __init__(
        self,
        contour_points: List[Tuple[int, ...]],
        num_antennas: int,
        dx: float,
        antenna_type: str = 'monopole',
        offset_mm: float = 5.0
    ):
        self.contour_points = contour_points
        self.num_antennas = num_antennas
        self.dx = dx
        self.antenna_type = antenna_type
        self.offset = offset_mm * 1e-3  # Offset from surface

        self.elements = self._compute_elements()

    def _compute_elements(self) -> List[AntennaElement]:
        """Distribute antennas uniformly along contour."""
        elements = []

        # Sample points uniformly along contour
        n_contour = len(self.contour_points)
        indices = np.linspace(0, n_contour - 1, self.num_antennas, dtype=int)

        for idx in indices:
            position = self.contour_points[idx]

            # Compute normal vector (pointing inward toward center)
            # Estimate from neighboring points
            idx_prev = max(0, idx - 5)
            idx_next = min(n_contour - 1, idx + 5)

            p_prev = np.array(self.contour_points[idx_prev][:2])
            p_next = np.array(self.contour_points[idx_next][:2])

            # Tangent vector
            tangent = p_next - p_prev
            tangent = tangent / np.linalg.norm(tangent)

            # Normal vector (perpendicular, pointing inward)
            normal = np.array([-tangent[1], tangent[0]])

            # Compute centroid to determine inward direction
            centroid = np.mean([p[:2] for p in self.contour_points], axis=0)
            to_center = centroid - np.array(position[:2])

            if np.dot(normal, to_center) < 0:
                normal = -normal

            # Apply offset
            offset_cells = self.offset / self.dx
            new_x = int(position[0] + normal[0] * offset_cells)
            new_y = int(position[1] + normal[1] * offset_cells)

            if len(position) == 3:
                new_position = (new_x, new_y, position[2])
                orientation = (normal[0], normal[1], 0.0)
            else:
                new_position = (new_x, new_y)
                orientation = (normal[0], normal[1])

            elements.append(AntennaElement(
                position=new_position,
                orientation=orientation,
                antenna_type=self.antenna_type
            ))

        return elements

    def get_antenna_positions(self) -> List[Tuple[int, ...]]:
        """Get list of antenna positions."""
        return [elem.position for elem in self.elements]


def visualize_array_geometry(
    array,
    grid_shape: Tuple[int, int],
    phantom_contour: Optional[List[Tuple[int, int]]] = None
):
    """Visualize antenna array geometry.

    Parameters
    ----------
    array : CircularArray, PlanarArray, or ConformalArray
        Antenna array object.
    grid_shape : tuple
        (nx, ny) grid dimensions.
    phantom_contour : list of tuples, optional
        List of (x, y) points defining phantom boundary.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot phantom contour if provided
    if phantom_contour is not None:
        contour_x = [p[0] for p in phantom_contour] + [phantom_contour[0][0]]
        contour_y = [p[1] for p in phantom_contour] + [phantom_contour[0][1]]
        ax.plot(contour_x, contour_y, 'b-', linewidth=2, label='Phantom')

    # Plot antennas
    positions = array.get_antenna_positions()
    ant_x = [p[0] for p in positions]
    ant_y = [p[1] for p in positions]

    ax.scatter(ant_x, ant_y, s=200, c='red', marker='^',
              edgecolors='black', linewidth=2, label='Antennas', zorder=5)

    # Add antenna numbers
    for i, (x, y) in enumerate(zip(ant_x, ant_y)):
        ax.text(x, y + 3, str(i+1), ha='center', va='bottom',
               fontsize=10, fontweight='bold')

    # Plot orientation vectors
    if hasattr(array, 'elements'):
        for elem in array.elements:
            x, y = elem.position[:2]
            ox, oy = elem.orientation[:2]
            # Scale orientation vector for visibility
            scale = 10
            ax.arrow(x, y, ox * scale, oy * scale,
                    head_width=2, head_length=2, fc='darkred', ec='darkred',
                    alpha=0.6, zorder=4)

    ax.set_xlim(0, grid_shape[0])
    ax.set_ylim(0, grid_shape[1])
    ax.set_aspect('equal')
    ax.set_xlabel('X (grid cells)', fontsize=12)
    ax.set_ylabel('Y (grid cells)', fontsize=12)
    ax.set_title(f'{array.__class__.__name__} Configuration\n'
                f'{len(positions)} Elements', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_imaging_array(
    array_type: str,
    imaging_target: str = 'head',
    grid_shape: Tuple[int, int] = (200, 200),
    dx: float = 1e-3
):
    """Factory function to create standard imaging arrays.

    Parameters
    ----------
    array_type : str
        'circular', 'planar', or 'conformal'.
    imaging_target : str
        'head', 'breast', or 'body'.
    grid_shape : tuple
        Grid dimensions.
    dx : float
        Grid spacing (meters).

    Returns
    -------
    array : AntennaArray
        Configured antenna array.

    Examples
    --------
    >>> # Standard head imaging array
    >>> array = create_imaging_array('circular', 'head')
    """
    center = (grid_shape[0] // 2, grid_shape[1] // 2)

    if array_type == 'circular':
        if imaging_target == 'head':
            # Standard 16-element circular array for stroke detection
            return CircularArray(
                num_antennas=16,
                radius_mm=120,  # 12 cm from center
                center=center,
                dx=dx,
                antenna_type='monopole',
                polarization='vertical'
            )
        elif imaging_target == 'breast':
            # Smaller radius for breast imaging
            return CircularArray(
                num_antennas=12,
                radius_mm=80,
                center=center,
                dx=dx,
                antenna_type='vivaldi',
                polarization='vertical'
            )

    elif array_type == 'planar':
        if imaging_target == 'breast':
            # 8×8 planar array
            corner = (center[0] - 40, center[1] - 40)
            return PlanarArray(
                nx_antennas=8,
                ny_antennas=8,
                spacing_mm=20,
                corner=corner,
                dx=dx,
                antenna_type='patch'
            )

    raise ValueError(f"Unsupported array_type '{array_type}' or "
                    f"imaging_target '{imaging_target}'")


class UniformLinearArray:
    """Uniform Linear Array (ULA) for MIMO and beamforming.

    Standard 1D array with uniform spacing - fundamental building block
    for many communication and radar systems.

    Parameters
    ----------
    num_elements : int
        Number of antenna elements.
    spacing_mm : float
        Element spacing (millimeters). Common: λ/2 for broadside arrays.
    orientation : str
        'horizontal' (x-axis) or 'vertical' (y-axis).
    start_position : tuple
        Starting point (x, y) or (x, y, z).
    dx : float
        Grid spacing (meters).
    antenna_type : str
        Element type.

    Examples
    --------
    >>> # Half-wavelength ULA for 3 GHz
    >>> wavelength = 3e8 / 3e9  # 0.1 m
    >>> ula = UniformLinearArray(
    ...     num_elements=8,
    ...     spacing_mm=wavelength/2 * 1000,  # λ/2 in mm
    ...     orientation='horizontal',
    ...     start_position=(50, 100),
    ...     dx=1e-3
    ... )
    """

    def __init__(
        self,
        num_elements: int,
        spacing_mm: float,
        orientation: str,
        start_position: Tuple[int, ...],
        dx: float,
        antenna_type: str = 'dipole'
    ):
        self.num_elements = num_elements
        self.spacing = spacing_mm * 1e-3
        self.orientation = orientation
        self.start_position = start_position
        self.dx = dx
        self.antenna_type = antenna_type

        self.elements = self._compute_elements()

    def _compute_elements(self) -> List[AntennaElement]:
        """Compute ULA element positions."""
        elements = []
        spacing_cells = int(self.spacing / self.dx)

        for i in range(self.num_elements):
            if self.orientation == 'horizontal':
                offset = (spacing_cells * i, 0, 0) if len(self.start_position) == 3 else (spacing_cells * i, 0)
                orientation_vec = (1.0, 0.0, 0.0) if len(self.start_position) == 3 else (1.0, 0.0)
            elif self.orientation == 'vertical':
                offset = (0, spacing_cells * i, 0) if len(self.start_position) == 3 else (0, spacing_cells * i)
                orientation_vec = (0.0, 1.0, 0.0) if len(self.start_position) == 3 else (0.0, 1.0)
            else:  # diagonal or custom
                angle = np.deg2rad(45)  # Example: 45° diagonal
                offset_x = int(spacing_cells * i * np.cos(angle))
                offset_y = int(spacing_cells * i * np.sin(angle))
                offset = (offset_x, offset_y, 0) if len(self.start_position) == 3 else (offset_x, offset_y)
                orientation_vec = (np.cos(angle), np.sin(angle), 0.0) if len(self.start_position) == 3 else (np.cos(angle), np.sin(angle))

            position = tuple(self.start_position[j] + offset[j]
                           for j in range(len(self.start_position)))

            elements.append(AntennaElement(
                position=position,
                orientation=orientation_vec,
                antenna_type=self.antenna_type
            ))

        return elements

    def get_antenna_positions(self) -> List[Tuple[int, ...]]:
        """Get list of antenna positions."""
        return [elem.position for elem in self.elements]

    def compute_steering_vector(
        self,
        frequency: float,
        angle_deg: float
    ) -> npt.NDArray[np.complex128]:
        """Compute steering vector for beamforming.

        Parameters
        ----------
        frequency : float
            Operating frequency (Hz).
        angle_deg : float
            Steering angle in degrees (0° = broadside).

        Returns
        -------
        steering_vector : ndarray (complex)
            Phase shifts for each element.
        """
        from ceep.core.constants import C_0

        wavelength = C_0 / frequency
        k = 2 * np.pi / wavelength
        angle_rad = np.deg2rad(angle_deg)

        # Phase shift between adjacent elements
        d = self.spacing
        phase_diff = k * d * np.sin(angle_rad)

        # Steering vector
        steering = np.exp(1j * phase_diff * np.arange(self.num_elements))
        return steering / np.sqrt(self.num_elements)  # Normalized


class UniformRectangularArray:
    """Uniform Rectangular Array (URA) for 2D MIMO systems.

    2D planar array with uniform spacing in both dimensions.
    Essential for massive MIMO and spatial multiplexing.

    Parameters
    ----------
    nx_elements, ny_elements : int
        Number of elements in x and y.
    spacing_x_mm, spacing_y_mm : float
        Element spacing (millimeters).
    corner : tuple
        Bottom-left corner position.
    dx : float
        Grid spacing (meters).
    antenna_type : str
        Element type.

    Examples
    --------
    >>> # 4×4 URA for massive MIMO
    >>> ura = UniformRectangularArray(
    ...     nx_elements=4,
    ...     ny_elements=4,
    ...     spacing_x_mm=50,
    ...     spacing_y_mm=50,
    ...     corner=(50, 50),
    ...     dx=1e-3
    ... )
    """

    def __init__(
        self,
        nx_elements: int,
        ny_elements: int,
        spacing_x_mm: float,
        spacing_y_mm: float,
        corner: Tuple[int, ...],
        dx: float,
        antenna_type: str = 'patch'
    ):
        self.nx_elements = nx_elements
        self.ny_elements = ny_elements
        self.spacing_x = spacing_x_mm * 1e-3
        self.spacing_y = spacing_y_mm * 1e-3
        self.corner = corner
        self.dx = dx
        self.antenna_type = antenna_type

        self.elements = self._compute_elements()

    def _compute_elements(self) -> List[AntennaElement]:
        """Compute URA element positions."""
        elements = []

        spacing_x_cells = int(self.spacing_x / self.dx)
        spacing_y_cells = int(self.spacing_y / self.dx)

        for i in range(self.nx_elements):
            for j in range(self.ny_elements):
                if len(self.corner) == 3:
                    position = (
                        self.corner[0] + i * spacing_x_cells,
                        self.corner[1] + j * spacing_y_cells,
                        self.corner[2]
                    )
                    orientation = (0.0, 0.0, 1.0)  # Z-directed
                else:
                    position = (
                        self.corner[0] + i * spacing_x_cells,
                        self.corner[1] + j * spacing_y_cells
                    )
                    orientation = (0.0, 1.0)  # Y-directed for 2D

                elements.append(AntennaElement(
                    position=position,
                    orientation=orientation,
                    antenna_type=self.antenna_type
                ))

        return elements

    def get_antenna_positions(self) -> List[Tuple[int, ...]]:
        """Get list of antenna positions."""
        return [elem.position for elem in self.elements]

    def compute_2d_steering_vector(
        self,
        frequency: float,
        azimuth_deg: float,
        elevation_deg: float = 0.0
    ) -> npt.NDArray[np.complex128]:
        """Compute 2D steering vector for beamforming.

        Parameters
        ----------
        frequency : float
            Operating frequency (Hz).
        azimuth_deg : float
            Azimuth angle (degrees).
        elevation_deg : float
            Elevation angle (degrees).

        Returns
        -------
        steering_vector : ndarray (complex)
            2D steering vector.
        """
        from ceep.core.constants import C_0

        wavelength = C_0 / frequency
        k = 2 * np.pi / wavelength

        az_rad = np.deg2rad(azimuth_deg)
        el_rad = np.deg2rad(elevation_deg)

        steering = np.zeros(self.nx_elements * self.ny_elements, dtype=complex)

        idx = 0
        for i in range(self.nx_elements):
            for j in range(self.ny_elements):
                phase_x = k * self.spacing_x * i * np.sin(az_rad) * np.cos(el_rad)
                phase_y = k * self.spacing_y * j * np.sin(el_rad)
                steering[idx] = np.exp(1j * (phase_x + phase_y))
                idx += 1

        return steering / np.sqrt(len(steering))


class LShapedArray:
    """L-Shaped Array for 2D DOA estimation.

    Two perpendicular linear arrays forming an L-shape.
    Efficient for 2D direction-of-arrival estimation with fewer elements
    than a full rectangular array.

    Parameters
    ----------
    num_x_elements : int
        Number of elements along x-axis.
    num_y_elements : int
        Number of elements along y-axis.
    spacing_mm : float
        Element spacing (millimeters).
    corner : tuple
        Corner position where the two arms meet.
    dx : float
        Grid spacing (meters).

    Examples
    --------
    >>> # L-array for DOA estimation
    >>> l_array = LShapedArray(
    ...     num_x_elements=8,
    ...     num_y_elements=8,
    ...     spacing_mm=50,
    ...     corner=(100, 100),
    ...     dx=1e-3
    ... )
    """

    def __init__(
        self,
        num_x_elements: int,
        num_y_elements: int,
        spacing_mm: float,
        corner: Tuple[int, int],
        dx: float,
        antenna_type: str = 'dipole'
    ):
        self.num_x_elements = num_x_elements
        self.num_y_elements = num_y_elements
        self.spacing = spacing_mm * 1e-3
        self.corner = corner
        self.dx = dx
        self.antenna_type = antenna_type

        self.elements = self._compute_elements()

    def _compute_elements(self) -> List[AntennaElement]:
        """Compute L-array element positions."""
        elements = []
        spacing_cells = int(self.spacing / self.dx)

        # X-arm (horizontal)
        for i in range(self.num_x_elements):
            position = (self.corner[0] + i * spacing_cells, self.corner[1])
            elements.append(AntennaElement(
                position=position,
                orientation=(0.0, 1.0),  # Y-directed
                antenna_type=self.antenna_type
            ))

        # Y-arm (vertical), skip corner element to avoid duplication
        for j in range(1, self.num_y_elements):
            position = (self.corner[0], self.corner[1] + j * spacing_cells)
            elements.append(AntennaElement(
                position=position,
                orientation=(1.0, 0.0),  # X-directed
                antenna_type=self.antenna_type
            ))

        return elements

    def get_antenna_positions(self) -> List[Tuple[int, ...]]:
        """Get list of antenna positions."""
        return [elem.position for elem in self.elements]


class RandomArray:
    """Random/Sparse Array for compressed sensing and imaging.

    Randomly or quasi-randomly placed antennas within a region.
    Useful for compressed sensing, super-resolution, and reducing
    mutual coupling.

    Parameters
    ----------
    num_elements : int
        Number of antenna elements.
    region : tuple
        (x_min, x_max, y_min, y_max) bounding box.
    dx : float
        Grid spacing (meters).
    min_spacing_mm : float, optional
        Minimum spacing between elements (to avoid mutual coupling).
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> # Random sparse array for compressed sensing
    >>> random_array = RandomArray(
    ...     num_elements=16,
    ...     region=(50, 150, 50, 150),
    ...     dx=1e-3,
    ...     min_spacing_mm=20,
    ...     seed=42
    ... )
    """

    def __init__(
        self,
        num_elements: int,
        region: Tuple[int, int, int, int],
        dx: float,
        min_spacing_mm: float = 10.0,
        seed: Optional[int] = None,
        antenna_type: str = 'monopole'
    ):
        self.num_elements = num_elements
        self.region = region
        self.dx = dx
        self.min_spacing = min_spacing_mm * 1e-3
        self.antenna_type = antenna_type

        if seed is not None:
            np.random.seed(seed)

        self.elements = self._compute_elements()

    def _compute_elements(self) -> List[AntennaElement]:
        """Compute random element positions with minimum spacing constraint."""
        elements = []
        min_spacing_cells = int(self.min_spacing / self.dx)

        x_min, x_max, y_min, y_max = self.region
        attempts = 0
        max_attempts = self.num_elements * 1000

        while len(elements) < self.num_elements and attempts < max_attempts:
            # Random position
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            position = (x, y)

            # Check minimum spacing constraint
            valid = True
            for elem in elements:
                dist = np.sqrt((x - elem.position[0])**2 +
                              (y - elem.position[1])**2)
                if dist < min_spacing_cells:
                    valid = False
                    break

            if valid:
                # Random orientation
                angle = np.random.uniform(0, 2*np.pi)
                orientation = (np.cos(angle), np.sin(angle))

                elements.append(AntennaElement(
                    position=position,
                    orientation=orientation,
                    antenna_type=self.antenna_type
                ))

            attempts += 1

        if len(elements) < self.num_elements:
            print(f"Warning: Only placed {len(elements)}/{self.num_elements} elements")

        return elements

    def get_antenna_positions(self) -> List[Tuple[int, ...]]:
        """Get list of antenna positions."""
        return [elem.position for elem in self.elements]
