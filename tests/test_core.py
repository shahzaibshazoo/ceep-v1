"""
Tests for neurowave.core module.

Covers:
- Physical constants correctness
- Grid configuration validation
- Simulation config and CFL calculation
- Abstract base class contracts
"""

import math
import pytest

from neurowave.core.constants import (
    C_0,
    EPS_0,
    MU_0,
    ETA_0,
    cfl_timestep_2d,
    cfl_timestep_3d,
    wavelength_to_frequency,
    frequency_to_wavelength,
)
from neurowave.core.config import (
    GridConfig,
    SimulationConfig,
    SimulationMode,
    Backend,
)


class TestConstants:
    """Tests for physical constants and utility functions."""

    def test_speed_of_light(self) -> None:
        """Verify c₀ = 1/√(ε₀μ₀)."""
        c_derived = 1.0 / math.sqrt(EPS_0 * MU_0)
        assert abs(c_derived - C_0) / C_0 < 1e-6

    def test_impedance_of_free_space(self) -> None:
        """Verify η₀ = √(μ₀/ε₀) ≈ 377 Ω."""
        eta_derived = math.sqrt(MU_0 / EPS_0)
        assert abs(eta_derived - ETA_0) / ETA_0 < 1e-6
        assert abs(ETA_0 - 376.73) < 0.1  # Sanity check

    def test_wavelength_frequency_conversion(self) -> None:
        """Test wavelength ↔ frequency conversion roundtrip."""
        freq = 1e9  # 1 GHz
        wavelength = frequency_to_wavelength(freq)
        freq_back = wavelength_to_frequency(wavelength)
        assert abs(freq_back - freq) / freq < 1e-12

    def test_cfl_2d_stability(self) -> None:
        """Verify 2D CFL timestep is within stability limit."""
        dx = dy = 1e-3  # 1 mm
        dt = cfl_timestep_2d(dx, dy, courant=0.5)
        dt_max = 1.0 / (C_0 * math.sqrt(2.0 / (dx * dx)))
        assert dt < dt_max

    def test_cfl_3d_stability(self) -> None:
        """Verify 3D CFL timestep is within stability limit."""
        dx = dy = dz = 1e-3
        dt = cfl_timestep_3d(dx, dy, dz, courant=0.5)
        dt_max = 1.0 / (C_0 * math.sqrt(3.0 / (dx * dx)))
        assert dt < dt_max


class TestGridConfig:
    """Tests for GridConfig dataclass."""

    def test_basic_2d_grid(self) -> None:
        """Test basic 2D grid creation."""
        grid = GridConfig(nx=100, ny=100, dx=1e-3, dy=1e-3)
        assert grid.shape == (100, 100)
        assert not grid.is_3d
        assert grid.total_cells == 10_000

    def test_basic_3d_grid(self) -> None:
        """Test basic 3D grid creation."""
        grid = GridConfig(nx=50, ny=50, nz=50, dx=1e-3, dy=1e-3, dz=1e-3)
        assert grid.shape == (50, 50, 50)
        assert grid.is_3d
        assert grid.total_cells == 125_000

    def test_physical_size(self) -> None:
        """Test physical size calculation."""
        grid = GridConfig(nx=200, ny=100, dx=0.5e-3, dy=0.5e-3)
        size = grid.physical_size
        assert abs(size[0] - 0.1) < 1e-10  # 200 * 0.5mm = 0.1m
        assert abs(size[1] - 0.05) < 1e-10

    def test_invalid_dimensions(self) -> None:
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError):
            GridConfig(nx=-1, ny=100)

    def test_invalid_spacing(self) -> None:
        """Test that negative spacing raises ValueError."""
        with pytest.raises(ValueError):
            GridConfig(nx=100, ny=100, dx=-1e-3)


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_basic_config(self) -> None:
        """Test basic simulation configuration."""
        grid = GridConfig(nx=100, ny=100)
        config = SimulationConfig(grid=grid, total_steps=1000)
        assert config.num_steps == 1000
        assert config.dt > 0

    def test_config_from_time(self) -> None:
        """Test configuration with total_time."""
        grid = GridConfig(nx=100, ny=100, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, total_time=1e-9)
        assert config.num_steps > 0
        assert config.num_steps * config.dt >= 1e-9

    def test_invalid_courant(self) -> None:
        """Test that invalid Courant numbers raise ValueError."""
        grid = GridConfig(nx=100, ny=100)
        with pytest.raises(ValueError):
            SimulationConfig(grid=grid, total_steps=100, courant=1.0)

    def test_missing_time_spec(self) -> None:
        """Test that missing time specification raises ValueError."""
        grid = GridConfig(nx=100, ny=100)
        with pytest.raises(ValueError):
            SimulationConfig(grid=grid)

    def test_summary_output(self) -> None:
        """Test that summary() produces non-empty string."""
        grid = GridConfig(nx=100, ny=100)
        config = SimulationConfig(grid=grid, total_steps=500)
        summary = config.summary()
        assert "NeuroWave" in summary
        assert "100" in summary
