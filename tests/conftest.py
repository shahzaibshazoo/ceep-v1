"""Test configuration — ensures backend reset between tests."""

import pytest
import numpy as np
from ceep.core.backend import set_backend
from ceep.core.config import GridConfig, SimulationConfig
from ceep.core.grid import Grid2D
from ceep.core.grid_3d import Grid3D


@pytest.fixture(autouse=True)
def reset_backend():
    """Ensure each test starts with numpy backend."""
    set_backend('numpy')
    yield
    set_backend('numpy')


# ============================================================
# Test Markers
# ============================================================

def pytest_configure(config):
    """Register custom test markers."""
    config.addinivalue_line(
        "markers", "meep: marks tests requiring MEEP (skip if not installed)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (takes > 1s)"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests for edge cases and boundary conditions"
    )


# ============================================================
# Common Fixture: Grids
# ============================================================

@pytest.fixture
def grid_2d_small():
    """Small 2D grid for fast testing."""
    config = SimulationConfig(
        grid=GridConfig(nx=50, ny=50, dx=1e-3, dy=1e-3),
        total_steps=50,
        courant=0.5
    )
    return Grid2D(config)


@pytest.fixture
def grid_2d_medium():
    """Medium 2D grid."""
    config = SimulationConfig(
        grid=GridConfig(nx=100, ny=100, dx=1e-3, dy=1e-3),
        total_steps=100,
        courant=0.5
    )
    return Grid2D(config)


@pytest.fixture
def grid_3d_small():
    """Small 3D grid for fast testing."""
    config = SimulationConfig(
        grid=GridConfig(nx=40, ny=40, nz=40, dx=1e-3, dy=1e-3, dz=1e-3),
        total_steps=50,
        courant=0.5
    )
    return Grid3D(config)


@pytest.fixture
def grid_3d_medium():
    """Medium 3D grid."""
    config = SimulationConfig(
        grid=GridConfig(nx=60, ny=60, nz=60, dx=1e-3, dy=1e-3, dz=1e-3),
        total_steps=100,
        courant=0.5
    )
    return Grid3D(config)


# ============================================================
# Common Fixture: MEEP Availability
# ============================================================

@pytest.fixture
def meep_installed():
    """Check if MEEP is installed and available."""
    try:
        import meep as mp
        return True
    except ImportError:
        return False


# ============================================================
# Common Fixture: Reference Data
# ============================================================

@pytest.fixture
def gaussian_source_params():
    """Standard Gaussian source parameters for reproducible tests."""
    return {
        'x': 25,
        'y': 25,
        'frequency_max': 5e9,
        'delay_factor': 5.0,
    }


@pytest.fixture
def gaussian_source_params_3d():
    """Standard Gaussian source parameters for 3D tests."""
    return {
        'x': 25,
        'y': 25,
        'z': 25,
        'frequency_max': 10e9,
        'delay_factor': 5.0,
    }


# ============================================================
# Common Fixture: Material Definitions
# ============================================================

@pytest.fixture
def material_definitions():
    """Common material property definitions."""
    return {
        'free_space': {'eps_r': 1.0, 'sigma_e': 0.0},
        'dielectric_low': {'eps_r': 2.0, 'sigma_e': 0.0},
        'dielectric_med': {'eps_r': 4.0, 'sigma_e': 0.0},
        'dielectric_high': {'eps_r': 9.0, 'sigma_e': 0.0},
        'lossy_light': {'eps_r': 1.0, 'sigma_e': 0.01},
        'lossy_med': {'eps_r': 1.0, 'sigma_e': 0.1},
        'lossy_heavy': {'eps_r': 1.0, 'sigma_e': 1.0},
    }


# ============================================================
# Common Fixture: Tolerance Thresholds
# ============================================================

@pytest.fixture
def tolerance_params():
    """Tolerance parameters for validation tests."""
    return {
        'field_threshold': 0.001,  # Minimum field amplitude
        'energy_threshold': 1e-10,  # Minimum energy
        'error_2d': 0.05,  # 5% error for 2D comparisons
        'error_3d': 0.03,  # 3% error for 3D comparisons
        'stability_max_field': 100.0,  # Max field magnitude
    }
