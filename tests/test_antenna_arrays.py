"""
Tests for antenna array configurations.
"""

import numpy as np
import pytest
from ceep.antennas import (
    CircularArray, PlanarArray, ConformalArray,
    create_imaging_array
)


def test_circular_array_creation():
    """Test circular array initialization."""
    array = CircularArray(
        num_antennas=16,
        radius_mm=100,
        center=(100, 100),
        dx=1e-3
    )

    assert array.num_antennas == 16
    assert len(array.elements) == 16


def test_circular_array_positions():
    """Test that antennas are positioned correctly on circle."""
    array = CircularArray(
        num_antennas=8,
        radius_mm=100,  # 100 mm = 0.1 m
        center=(100, 100),
        dx=1e-3  # 1 mm grid
    )

    positions = array.get_antenna_positions()
    assert len(positions) == 8

    # First antenna should be at approximately (200, 100)
    # radius = 100mm / 1mm = 100 cells
    first_pos = positions[0]
    assert abs(first_pos[0] - 200) < 2  # Allow small rounding error
    assert abs(first_pos[1] - 100) < 2


def test_circular_array_symmetry():
    """Test that opposite antennas are symmetric."""
    array = CircularArray(
        num_antennas=8,
        radius_mm=100,
        center=(100, 100),
        dx=1e-3
    )

    positions = array.get_antenna_positions()

    # Antenna 0 and 4 should be opposite (for 8 antennas)
    pos0 = np.array(positions[0])
    pos4 = np.array(positions[4])
    center = np.array([100, 100])

    # Both should be same distance from center
    dist0 = np.linalg.norm(pos0 - center)
    dist4 = np.linalg.norm(pos4 - center)
    assert abs(dist0 - dist4) < 1


def test_circular_array_sources():
    """Test source creation for circular array."""
    array = CircularArray(
        num_antennas=4,
        radius_mm=100,
        center=(100, 100),
        dx=1e-3
    )

    sources = array.create_sources(
        frequency=1e9,
        bandwidth=500e6,
        source_type='modulated_gaussian'
    )

    assert len(sources) == 4
    # Check that sources are at antenna positions
    for source, elem in zip(sources, array.elements):
        assert source.x == elem.position[0]
        assert source.y == elem.position[1]


def test_transmit_receive_pairs():
    """Test generation of TX-RX pairs for multistatic imaging."""
    array = CircularArray(
        num_antennas=4,
        radius_mm=100,
        center=(100, 100),
        dx=1e-3
    )

    pairs = array.get_transmit_receive_pairs()

    # Should have N×N pairs for N antennas
    assert len(pairs) == 16  # 4×4

    # Should include monostatic (i, i) pairs
    assert (0, 0) in pairs
    assert (1, 1) in pairs

    # Should include bistatic pairs
    assert (0, 1) in pairs
    assert (1, 0) in pairs


def test_planar_array_creation():
    """Test planar array initialization."""
    array = PlanarArray(
        nx_antennas=4,
        ny_antennas=4,
        spacing_mm=20,
        corner=(50, 50),
        dx=1e-3
    )

    assert array.nx_antennas == 4
    assert array.ny_antennas == 4
    assert len(array.elements) == 16  # 4×4


def test_planar_array_grid():
    """Test that planar array forms a regular grid."""
    array = PlanarArray(
        nx_antennas=3,
        ny_antennas=3,
        spacing_mm=20,  # 20 mm spacing
        corner=(50, 50),
        dx=1e-3
    )

    positions = array.get_antenna_positions()

    # Extract x and y coordinates
    x_coords = sorted(set(p[0] for p in positions))
    y_coords = sorted(set(p[1] for p in positions))

    # Should have 3 unique x and y values
    assert len(x_coords) == 3
    assert len(y_coords) == 3

    # Spacing should be approximately 20 cells (20mm / 1mm)
    assert abs(x_coords[1] - x_coords[0] - 20) < 1
    assert abs(y_coords[1] - y_coords[0] - 20) < 1


def test_conformal_array_creation():
    """Test conformal array following a contour."""
    # Create elliptical contour
    angles = np.linspace(0, 2*np.pi, 100)
    contour = [(int(100 + 50*np.cos(a)), int(100 + 40*np.sin(a)))
               for a in angles]

    array = ConformalArray(
        contour_points=contour,
        num_antennas=8,
        dx=1e-3
    )

    assert len(array.elements) == 8


def test_conformal_array_follows_contour():
    """Test that conformal antennas are near the contour."""
    # Circle contour
    radius = 50
    angles = np.linspace(0, 2*np.pi, 100)
    contour = [(int(100 + radius*np.cos(a)), int(100 + radius*np.sin(a)))
               for a in angles]

    array = ConformalArray(
        contour_points=contour,
        num_antennas=8,
        dx=1e-3,
        offset_mm=0  # No offset
    )

    positions = array.get_antenna_positions()
    center = np.array([100, 100])

    # All antennas should be approximately at radius distance
    for pos in positions:
        dist = np.linalg.norm(np.array(pos[:2]) - center)
        assert abs(dist - radius) < 10  # Within 10 cells


def test_create_imaging_array_head():
    """Test factory function for standard head imaging array."""
    array = create_imaging_array('circular', 'head')

    assert isinstance(array, CircularArray)
    assert array.num_antennas == 16


def test_create_imaging_array_breast():
    """Test factory function for breast imaging array."""
    array = create_imaging_array('planar', 'breast')

    assert isinstance(array, PlanarArray)
    assert array.nx_antennas == 8
    assert array.ny_antennas == 8


def test_array_factor():
    """Test array factor computation for beamforming."""
    array = CircularArray(
        num_antennas=8,
        radius_mm=100,
        center=(100, 100),
        dx=1e-3
    )

    angles = np.linspace(0, 2*np.pi, 360)
    AF = array.compute_array_factor(frequency=1e9, angles=angles)

    assert AF.shape == angles.shape
    assert AF.dtype == complex

    # Array factor should have maximum somewhere
    assert np.max(np.abs(AF)) > 0


def test_antenna_orientations():
    """Test that antenna orientations point toward center."""
    array = CircularArray(
        num_antennas=4,
        radius_mm=100,
        center=(100, 100),
        dx=1e-3
    )

    center = np.array([100, 100])

    for elem in array.elements:
        pos = np.array(elem.position[:2])
        orient = np.array(elem.orientation[:2])

        # Orientation should point toward center
        to_center = center - pos
        to_center = to_center / np.linalg.norm(to_center)

        # Dot product should be close to 1
        dot = np.dot(orient, to_center)
        assert dot > 0.9  # Mostly aligned with inward direction


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
