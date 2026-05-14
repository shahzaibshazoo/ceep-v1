"""
GPU Validation Tests
====================

Validates that GPU (CuPy) results match CPU (NumPy) within floating-point
tolerance. Tests are skipped if CuPy is not available.

Run with: PYTHONPATH=./src python -m pytest tests/test_gpu.py -v
"""

import numpy as np
import pytest

from ceep.core.backend import is_backend_available, Backend, set_backend
from ceep.core.config import GridConfig, SimulationConfig
from ceep.sources.waveforms import GaussianSource

HAS_CUPY = is_backend_available(Backend.CUPY)
pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


def _run_2d_tmz_sim(steps=100):
    """Run a 2D TMz simulation and return final Ez field + probe data."""
    from ceep.solvers.fdtd_2d import FDTD2D
    from ceep.boundaries.absorbing import CPML

    grid_config = GridConfig(nx=100, ny=100, dx=1e-3, dy=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=steps)

    source = GaussianSource(
        x=50, y=50, frequency_max=10e9,
        field_component='Ez', delay_factor=5.0
    )
    cpml = CPML(thickness=8)

    solver = FDTD2D(
        config=config,
        sources=[source],
        boundaries=[cpml],
        probe_points=[(30, 50), (70, 50)],
        record_field='Ez'
    )
    solver.run()

    ez_final = solver.get_field('Ez')
    probe_30 = solver.probe_data[(30, 50)]
    probe_70 = solver.probe_data[(70, 50)]
    return ez_final, probe_30, probe_70


def _run_3d_sim(steps=50):
    """Run a 3D simulation and return final Ez field + probe data."""
    from ceep.solvers.fdtd_3d import FDTD3D

    grid_config = GridConfig(nx=30, ny=30, nz=30, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=steps)

    source = GaussianSource(
        x=15, y=15, z=15, frequency_max=10e9,
        field_component='Ez', delay_factor=5.0
    )

    solver = FDTD3D(config, sources=[source])
    solver.probe_points = [(10, 15, 15), (20, 15, 15)]
    for p in solver.probe_points:
        solver.probe_data[p] = []
    solver.run()

    return solver.grid.ez, solver.probe_data[(10, 15, 15)], solver.probe_data[(20, 15, 15)]


class TestGPUValidation:
    """Validate GPU results match CPU within floating-point tolerance."""

    def test_2d_tmz_cpu_gpu_match(self):
        """2D TMz simulation produces identical results on CPU and GPU."""
        from ceep.core.backend import to_numpy

        # Run on CPU
        set_backend('numpy')
        ez_cpu, probe_30_cpu, probe_70_cpu = _run_2d_tmz_sim(steps=50)
        ez_cpu = np.array(ez_cpu)

        # Run on GPU
        set_backend('cupy')
        ez_gpu, probe_30_gpu, probe_70_gpu = _run_2d_tmz_sim(steps=50)
        ez_gpu = to_numpy(ez_gpu)

        # Compare
        np.testing.assert_allclose(ez_gpu, ez_cpu, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(probe_30_gpu, probe_30_cpu, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(probe_70_gpu, probe_70_cpu, rtol=1e-12, atol=1e-15)

    def test_3d_cpu_gpu_match(self):
        """3D simulation produces identical results on CPU and GPU."""
        from ceep.core.backend import to_numpy

        # Run on CPU
        set_backend('numpy')
        ez_cpu, probe_a_cpu, probe_b_cpu = _run_3d_sim(steps=30)
        ez_cpu = np.array(ez_cpu)

        # Run on GPU
        set_backend('cupy')
        ez_gpu, probe_a_gpu, probe_b_gpu = _run_3d_sim(steps=30)
        ez_gpu = to_numpy(ez_gpu)

        np.testing.assert_allclose(ez_gpu, ez_cpu, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(probe_a_gpu, probe_a_cpu, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(probe_b_gpu, probe_b_cpu, rtol=1e-12, atol=1e-15)

    def test_cpml_cpu_gpu_match(self):
        """CPML absorption produces identical results on both backends."""
        from ceep.core.backend import to_numpy
        from ceep.solvers.fdtd_3d import FDTD3D
        from ceep.boundaries.absorbing import CPML

        def run_cpml_sim():
            grid_config = GridConfig(nx=30, ny=30, nz=30, dx=1e-3, dy=1e-3, dz=1e-3)
            config = SimulationConfig(grid=grid_config, total_steps=50)
            source = GaussianSource(
                x=15, y=15, z=15, frequency_max=10e9,
                field_component='Ez', delay_factor=5.0
            )
            cpml = CPML(thickness=6)
            solver = FDTD3D(config, sources=[source], boundaries=[cpml])
            solver.run()
            return solver.grid.ez, solver.compute_energy()

        set_backend('numpy')
        ez_cpu, energy_cpu = run_cpml_sim()
        ez_cpu = np.array(ez_cpu)

        set_backend('cupy')
        ez_gpu, energy_gpu = run_cpml_sim()
        ez_gpu = to_numpy(ez_gpu)

        np.testing.assert_allclose(ez_gpu, ez_cpu, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(energy_gpu, energy_cpu, rtol=1e-12)

    def test_gpu_performance_sanity(self):
        """GPU should not be significantly slower than CPU for medium grids."""
        import time
        from ceep.solvers.fdtd_2d import FDTD2D

        grid_config = GridConfig(nx=200, ny=200, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid_config, total_steps=100)
        source = GaussianSource(
            x=100, y=100, frequency_max=10e9,
            field_component='Ez', delay_factor=5.0
        )

        # CPU timing
        set_backend('numpy')
        solver_cpu = FDTD2D(config=config, sources=[source])
        t0 = time.time()
        solver_cpu.run()
        cpu_time = time.time() - t0

        # GPU timing (includes JIT compilation on first run)
        set_backend('cupy')
        import cupy as cp
        solver_gpu = FDTD2D(config=config, sources=[source])
        # Warmup
        solver_gpu.step()
        cp.cuda.Device().synchronize()
        solver_gpu = FDTD2D(config=config, sources=[source])
        t0 = time.time()
        solver_gpu.run()
        cp.cuda.Device().synchronize()
        gpu_time = time.time() - t0

        # GPU should be at most 5x slower (accounting for small grid overhead)
        # On real workloads (1000x1000+), GPU will be much faster
        assert gpu_time < cpu_time * 5, (
            f"GPU too slow: {gpu_time:.3f}s vs CPU {cpu_time:.3f}s"
        )


class TestBackendSwitch:
    """Test that backend switching works correctly."""

    def test_backend_switch_preserves_results(self):
        """Switching backend mid-session produces correct results."""
        from ceep.solvers.fdtd_2d import FDTD2D

        grid_config = GridConfig(nx=50, ny=50, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid_config, total_steps=20)
        source = GaussianSource(
            x=25, y=25, frequency_max=10e9,
            field_component='Ez', delay_factor=5.0
        )

        # Run on CPU first
        set_backend('numpy')
        solver = FDTD2D(config=config, sources=[source])
        solver.run()
        cpu_ez = np.array(solver.get_field('Ez'))

        # Switch to GPU and run same sim
        set_backend('cupy')
        from ceep.core.backend import to_numpy
        solver = FDTD2D(config=config, sources=[source])
        solver.run()
        gpu_ez = to_numpy(solver.get_field('Ez'))

        np.testing.assert_allclose(gpu_ez, cpu_ez, rtol=1e-12, atol=1e-15)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
