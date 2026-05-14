"""
Tests for anatomical phantom models.
"""

import numpy as np
import pytest
from ceep.phantoms import SimpleHeadPhantom, DetailedBrainPhantom, SkinLayerPhantom


def test_simple_head_phantom_creation():
    """Test simple head phantom initialization."""
    phantom = SimpleHeadPhantom(nx=100, ny=100, dx=1e-3)
    assert phantom.nx == 100
    assert phantom.ny == 100
    assert phantom.head_radius == 90e-3


def test_simple_head_tissue_map():
    """Test tissue map creation."""
    phantom = SimpleHeadPhantom(nx=100, ny=100, dx=1e-3, head_radius_mm=50)
    tissue_map = phantom.create_tissue_map()

    assert tissue_map.shape == (100, 100)

    # Center should be brain
    center_tissue = tissue_map[50, 50]
    assert center_tissue is not None
    assert 'brain' in center_tissue.name.lower()

    # Outside should be None (free space)
    assert tissue_map[0, 0] is None


def test_simple_head_permittivity():
    """Test permittivity map generation."""
    phantom = SimpleHeadPhantom(nx=200, ny=200, dx=1e-3, head_radius_mm=50)
    eps_real, eps_imag = phantom.get_permittivity_map(10e9)

    assert eps_real.shape == (200, 200)
    assert eps_imag.shape == (200, 200)

    # Brain permittivity should be around 40-50
    center_eps = eps_real[100, 100]
    assert 35 < center_eps < 60

    # Far corners should be 1 (free space)
    assert eps_real[0, 0] == 1.0
    assert eps_real[0, 199] == 1.0


def test_detailed_brain_phantom_3d():
    """Test 3D detailed brain phantom."""
    phantom = DetailedBrainPhantom(nx=80, ny=80, nz=80, dx=1e-3)

    assert phantom.nx == 80
    assert phantom.hemorrhages == []


def test_detailed_brain_with_hemorrhage():
    """Test adding hemorrhage to brain phantom."""
    phantom = DetailedBrainPhantom(nx=80, ny=80, nz=80, dx=1e-3)
    phantom.add_hemorrhage(x=45, y=40, z=40, radius_mm=8)

    assert len(phantom.hemorrhages) == 1
    assert phantom.hemorrhages[0]['radius'] == 8e-3


def test_detailed_brain_tissue_map():
    """Test 3D tissue map creation."""
    phantom = DetailedBrainPhantom(nx=60, ny=60, nz=60, dx=1e-3)
    tissue_map = phantom.create_tissue_map()

    assert tissue_map.shape == (60, 60, 60)

    # Center should be white matter
    center_tissue = tissue_map[30, 30, 30]
    assert center_tissue is not None
    assert 'white' in center_tissue.name.lower()


def test_skin_layer_phantom():
    """Test skin layer phantom for melanoma detection."""
    phantom = SkinLayerPhantom(nx=50, ny=50, dx=0.1e-3)

    assert phantom.nx == 50
    assert phantom.dx == 0.1e-3


def test_skin_phantom_with_tumor():
    """Test adding tumor to skin phantom."""
    phantom = SkinLayerPhantom(nx=50, ny=50, dx=0.1e-3)
    phantom.add_tumor(x=10, y=25, depth_mm=1.0, width_mm=2.0, height_mm=1.5)

    tissue_map = phantom.create_tissue_map()
    assert tissue_map.shape == (50, 50)

    # Check that tumor was inserted
    assert len(phantom.tumor_positions) == 1


def test_phantom_layer_differentiation():
    """Test that different tissue layers have different properties."""
    phantom = SimpleHeadPhantom(nx=200, ny=200, dx=1e-3, head_radius_mm=50)
    eps_real, _ = phantom.get_permittivity_map(10e9)

    # Sample points at different depths
    center = 100
    brain_eps = eps_real[center, center]  # Center (brain)
    skull_eps = eps_real[center, center + 43]  # At skull layer

    # Brain should have higher permittivity than skull (brain ~45-50, skull ~10-20)
    assert brain_eps > skull_eps + 5  # Clear difference


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
