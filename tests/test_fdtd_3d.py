"""
Tests for 3D FDTD solver.
"""

import numpy as np
import pytest
from neurowave.core.config import GridConfig, SimulationConfig
from neurowave.solvers.fdtd_3d import FDTD3D
from neurowave.sources.waveforms import GaussianSource
from neurowave.boundaries.absorbing import CPML
from neurowave.core.constants import C_0


def test_3d_grid_creation():
    """Test 3D grid initialization."""
    grid_config = GridConfig(nx=50, ny=50, nz=50, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=100)
    solver = FDTD3D(config)

    assert solver.grid.nx == 50
    assert solver.grid.ny == 50
    assert solver.grid.nz == 50
    assert solver.grid.ex.shape == (50, 50, 50)
    assert solver.grid.hz.shape == (50, 50, 50)


def test_3d_field_updates():
    """Test that 3D field updates execute without error."""
    grid_config = GridConfig(nx=30, ny=30, nz=30, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=10)
    solver = FDTD3D(config)

    # Run a few steps
    for _ in range(10):
        solver.step()

    # Fields should still be mostly zero (no sources)
    assert np.max(np.abs(solver.grid.ex)) < 1e-10
    assert np.max(np.abs(solver.grid.hz)) < 1e-10


def test_3d_source_injection():
    """Test 3D point source injection."""
    grid_config = GridConfig(nx=50, ny=50, nz=50, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=100)

    # Create a point source
    source = GaussianSource(
        x=25, y=25, z=25,
        frequency_max=10e9,
        field_component='Ez',
        delay_factor=5.0
    )

    solver = FDTD3D(config, sources=[source])

    # Add probe at source location
    solver.probe_points = [(25, 25, 25)]
    solver.probe_data[(25, 25, 25)] = []

    solver.run()

    # Check that field was excited
    data = solver.get_probe_data(25, 25, 25)
    assert len(data) == 100
    assert np.max(np.abs(data)) > 0.01  # Should have significant amplitude


def test_3d_wave_propagation():
    """Test that a 3D wave propagates correctly."""
    dx = 1e-3  # 1 mm
    grid_config = GridConfig(nx=100, ny=50, nz=50, dx=dx, dy=dx, dz=dx)
    config = SimulationConfig(grid=grid_config, total_steps=350)

    source = GaussianSource(
        x=25, y=25, z=25,
        frequency_max=10e9,
        field_component='Ez',
        delay_factor=5.0
    )

    solver = FDTD3D(config, sources=[source])

    # Probes must be far from source to avoid near-field contamination from
    # soft source injection. Place at 15 and 35 cells from source (20 cell gap).
    solver.probe_points = [(40, 25, 25), (60, 25, 25)]
    for p in solver.probe_points:
        solver.probe_data[p] = []

    solver.run()

    data_near = solver.get_probe_data(40, 25, 25)
    data_far = solver.get_probe_data(60, 25, 25)

    # Measure delay via wavefront arrival (more robust than peak detection)
    threshold = np.max(np.abs(data_far)) * 0.01
    first_near = np.argmax(np.abs(data_near) > threshold)
    first_far = np.argmax(np.abs(data_far) > threshold)

    # Time delay (20mm separation)
    distance = 20 * dx
    expected_delay_steps = distance / (C_0 * config.dt)

    delay_steps = first_far - first_near
    error = abs(delay_steps - expected_delay_steps) / expected_delay_steps

    assert error < 0.25, f"Wave delay error: {error*100:.1f}% (measured {delay_steps}, expected {expected_delay_steps:.1f})"

    # Verify amplitude ratio (far probe weaker due to 3D spherical spreading)
    ratio = np.max(np.abs(data_far)) / np.max(np.abs(data_near))
    assert 0.05 < ratio < 1.0, f"Amplitude ratio unexpected: {ratio:.3f}"


def test_3d_cpml_initialization():
    """Test CPML initialization for 3D."""
    grid_config = GridConfig(nx=50, ny=50, nz=50, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=10)

    cpml = CPML(thickness=10)
    solver = FDTD3D(config, boundaries=[cpml])

    # Check that CPML arrays were allocated
    assert hasattr(cpml, '_psi_ezx_xlo_3d')
    assert cpml._psi_ezx_xlo_3d.shape == (10, 50, 50)


def test_3d_field_slice_extraction():
    """Test extracting 2D slices from 3D fields."""
    grid_config = GridConfig(nx=60, ny=50, nz=40, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=10)
    solver = FDTD3D(config)

    # Set a test pattern in Ez
    solver.grid.ez[30, 25, 20] = 1.0

    # Extract slices
    xy_slice = solver.get_slice_2d('Ez', 'xy', 20)
    xz_slice = solver.get_slice_2d('Ez', 'xz', 25)
    yz_slice = solver.get_slice_2d('Ez', 'yz', 30)

    assert xy_slice.shape == (60, 50)
    assert xz_slice.shape == (60, 40)
    assert yz_slice.shape == (50, 40)

    # Check that our test point appears in slices
    assert xy_slice[30, 25] == 1.0
    assert xz_slice[30, 20] == 1.0
    assert yz_slice[25, 20] == 1.0


def test_3d_energy_conservation():
    """Test that energy behaves correctly (grows then stabilizes without CPML)."""
    grid_config = GridConfig(nx=50, ny=50, nz=50, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=400)

    # Faster-peaking Gaussian
    source = GaussianSource(
        x=25, y=25, z=25,
        frequency_max=10e9,
        field_component='Ez',
        delay_factor=2.0
    )

    solver = FDTD3D(config, sources=[source])  # No CPML - energy should be conserved

    energies = []
    for _ in range(400):
        solver.step()
        if solver._step % 10 == 0:
            energies.append(solver.compute_energy())

    # Without CPML: energy should grow (source active) then plateau (source stops, conserved)
    peak_energy = np.max(energies)
    final_energy = energies[-1]
    initial_energy = energies[0]

    assert peak_energy > 1e-20, f"Peak energy too small: {peak_energy}"
    assert peak_energy > initial_energy * 100, "Energy should grow significantly from source"
    # Without boundaries, energy is conserved (doesn't decay)
    assert final_energy > peak_energy * 0.5, "Energy shouldn't decay without absorbing boundaries"


def test_3d_numerical_stability():
    """Test that 3D simulation remains stable."""
    grid_config = GridConfig(nx=40, ny=40, nz=40, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=500)

    source = GaussianSource(
        x=20, y=20, z=20,
        frequency_max=10e9,
        field_component='Ez',
        delay_factor=5.0
    )

    solver = FDTD3D(config, sources=[source])
    solver.run()

    assert np.all(np.isfinite(solver.grid.ex))
    assert np.all(np.isfinite(solver.grid.hy))
    assert np.all(np.isfinite(solver.grid.ez))


def test_3d_cpml_stability():
    """Test that 3D CPML remains numerically stable."""
    grid_config = GridConfig(nx=40, ny=40, nz=40, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=600)

    source = GaussianSource(
        x=20, y=20, z=20,
        frequency_max=10e9,
        field_component='Ez',
        delay_factor=5.0
    )

    cpml = CPML(thickness=8)
    solver = FDTD3D(config, sources=[source], boundaries=[cpml])
    solver.run()

    assert np.all(np.isfinite(solver.grid.ex))
    assert np.all(np.isfinite(solver.grid.ey))
    assert np.all(np.isfinite(solver.grid.ez))
    assert np.all(np.isfinite(solver.grid.hx))
    assert np.all(np.isfinite(solver.grid.hy))
    assert np.all(np.isfinite(solver.grid.hz))

    energy = solver.compute_energy()
    assert energy < 1e-10, f"Energy should be bounded, got {energy}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
