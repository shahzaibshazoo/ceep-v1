#!/usr/bin/env python
"""Quick validation script for FDTD refactoring."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from ceep.core.config import GridConfig, SimulationConfig, SimulationMode
from ceep.solvers.fdtd_2d import FDTD2D
from ceep.solvers.fdtd_3d import FDTD3D
from ceep.sources.waveforms import GaussianSource

def test_fdtd2d_basic():
    """Test basic FDTD2D functionality."""
    print("Testing FDTD2D basic functionality...")

    grid = GridConfig(nx=50, ny=50, dx=1e-3, dy=1e-3)
    config = SimulationConfig(grid=grid, total_steps=10, mode=SimulationMode.TMZ)

    # Create with no sources/boundaries
    solver = FDTD2D(config)

    # Check basic properties
    assert solver.current_step == 0
    assert solver.current_time == 0.0
    assert solver.config == config

    # Run one step
    solver.step()
    assert solver.current_step == 1
    assert solver.current_time > 0.0

    # Run several more steps
    solver.run(5)
    assert solver.current_step == 6

    print("✓ FDTD2D basic functionality works")

def test_fdtd2d_with_sources():
    """Test FDTD2D with sources."""
    print("Testing FDTD2D with sources...")

    grid = GridConfig(nx=50, ny=50, dx=1e-3, dy=1e-3)
    config = SimulationConfig(grid=grid, total_steps=10, mode=SimulationMode.TMZ, dt=1e-11)

    src = GaussianSource(x=25, y=25, component="Ez", delay=1e-11, width=1e-11)
    solver = FDTD2D(config, sources=[src])

    # Run full simulation
    solver.run()
    assert solver.current_step == 10

    # Check that field was updated
    ez_field = solver.get_field("Ez")
    assert ez_field.shape == (50, 50)

    print("✓ FDTD2D with sources works")

def test_fdtd2d_with_recording():
    """Test FDTD2D with probe recording."""
    print("Testing FDTD2D with probe recording...")

    grid = GridConfig(nx=50, ny=50, dx=1e-3, dy=1e-3)
    config = SimulationConfig(grid=grid, total_steps=5, mode=SimulationMode.TMZ, dt=1e-11)

    probe_points = [(25, 25), (30, 30)]
    solver = FDTD2D(config, probe_points=probe_points, record_field="Ez")

    # Run simulation
    solver.run()

    # Check probe data was recorded
    assert len(solver.probe_data[(25, 25)]) == 5
    assert len(solver.probe_data[(30, 30)]) == 5

    print("✓ FDTD2D with recording works")

def test_fdtd3d_basic():
    """Test basic FDTD3D functionality."""
    print("Testing FDTD3D basic functionality...")

    grid = GridConfig(nx=20, ny=20, nz=20, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid, total_steps=10)

    # Create with no sources/boundaries
    solver = FDTD3D(config)

    # Check basic properties
    assert solver.current_step == 0
    assert solver.current_time == 0.0
    assert solver.config == config

    # Run one step
    solver.step()
    assert solver.current_step == 1

    # Run more steps
    solver.run(5)
    assert solver.current_step == 6

    print("✓ FDTD3D basic functionality works")

def test_fdtd3d_with_probes():
    """Test FDTD3D with probe recording."""
    print("Testing FDTD3D with probe recording...")

    grid = GridConfig(nx=20, ny=20, nz=20, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid, total_steps=5)

    probe_points = [(10, 10, 10), (15, 15, 15)]
    solver = FDTD3D(config, probe_points=probe_points, record_field="Ez")

    # Run simulation
    solver.run()

    # Check probe data was recorded
    assert len(solver.probe_data[(10, 10, 10)]) == 5
    assert len(solver.probe_data[(15, 15, 15)]) == 5

    print("✓ FDTD3D with probes works")

def test_inheritance():
    """Test that classes properly inherit from FdtdBase."""
    print("Testing class inheritance...")

    from ceep.solvers.fdtd_base import FdtdBase

    grid = GridConfig(nx=20, ny=20, dx=1e-3, dy=1e-3)
    config = SimulationConfig(grid=grid, total_steps=5, mode=SimulationMode.TMZ)

    solver2d = FDTD2D(config)
    assert isinstance(solver2d, FdtdBase)

    grid3d = GridConfig(nx=10, ny=10, nz=10, dx=1e-3, dy=1e-3, dz=1e-3)
    config3d = SimulationConfig(grid=grid3d, total_steps=5)
    solver3d = FDTD3D(config3d)
    assert isinstance(solver3d, FdtdBase)

    print("✓ Inheritance works correctly")

if __name__ == "__main__":
    try:
        test_inheritance()
        test_fdtd2d_basic()
        test_fdtd2d_with_sources()
        test_fdtd2d_with_recording()
        test_fdtd3d_basic()
        test_fdtd3d_with_probes()
        print("\n✅ All validation tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
