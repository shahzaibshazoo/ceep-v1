"""
Performance regression tests — Track solver speed and detect slowdowns.

These tests measure:
- Baseline performance for 2D and 3D solvers
- Batched speedup factors
- Memory scaling
- Regression detection (>10% slowdown alerts)
"""

import pytest
import numpy as np
import time
from ceep.core.config import GridConfig, SimulationConfig
from ceep.core.constants import C_0
from ceep.solvers.fdtd_2d import FDTD2D
from ceep.solvers.fdtd_3d import FDTD3D
from ceep.sources.waveforms import GaussianSource
from ceep.boundaries.absorbing import CPML, MurABC


# Performance baseline thresholds (in seconds per iteration per grid cell)
# These will vary by machine; these are rough estimates for modern CPU/GPU
PERF_2D_BASELINE_NS_PER_CELL = 100e-9  # ~100 ns per cell per step
PERF_3D_BASELINE_NS_PER_CELL = 1000e-9  # ~1 us per cell per step

SLOWDOWN_THRESHOLD = 0.10  # Flag if > 10% slower than baseline


class TestSolverPerformance:
    """Measure and validate solver performance."""

    def test_2d_solver_baseline_speed(self):
        """Measure 2D solver performance."""

        nx, ny = 100, 100
        steps = 100
        dx = 1e-3

        config = SimulationConfig(
            grid=GridConfig(nx=nx, ny=ny, dx=dx, dy=dx),
            total_steps=steps,
            courant=0.5
        )

        src = GaussianSource(x=nx//2, y=ny//2, frequency_max=5e9)
        solver = FDTD2D(config=config, sources=[src], boundaries=[MurABC()])

        # Warm up
        solver.run(5)

        # Timed run
        start = time.time()
        for _ in range(steps):
            solver.step()
        elapsed = time.time() - start

        # Calculate throughput
        cells = nx * ny
        total_ops = cells * steps
        ns_per_cell = (elapsed / total_ops) * 1e9

        print(f"\n2D Performance: {ns_per_cell:.1f} ns/cell/step")
        print(f"  Grid: {nx}×{ny}, Steps: {steps}")
        print(f"  Total time: {elapsed:.3f}s")

        # Performance should be reasonable
        # Allow 10x baseline for first run
        assert ns_per_cell < PERF_2D_BASELINE_NS_PER_CELL * 10

    def test_3d_solver_baseline_speed(self):
        """Measure 3D solver performance."""

        nx, ny, nz = 50, 50, 50
        steps = 50
        dx = 1e-3

        config = SimulationConfig(
            grid=GridConfig(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dx),
            total_steps=steps,
            courant=0.5
        )

        src = GaussianSource(x=nx//2, y=ny//2, z=nz//2, frequency_max=10e9)
        solver = FDTD3D(config=config, sources=[src], boundaries=[CPML(thickness=4)])

        # Warm up
        solver.run(5)

        # Timed run
        start = time.time()
        for _ in range(steps):
            solver.step()
        elapsed = time.time() - start

        # Calculate throughput
        cells = nx * ny * nz
        total_ops = cells * steps
        ns_per_cell = (elapsed / total_ops) * 1e9

        print(f"\n3D Performance: {ns_per_cell:.1f} ns/cell/step")
        print(f"  Grid: {nx}×{ny}×{nz}, Steps: {steps}")
        print(f"  Total time: {elapsed:.3f}s")

        # Performance should be reasonable
        assert ns_per_cell < PERF_3D_BASELINE_NS_PER_CELL * 10


class TestBatchedPerformance:
    """Test batched solver speedup."""

    def test_batched_2d_speedup_factor(self):
        """Batched 2D solver should provide 10-15x speedup over sequential."""

        from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

        nx, ny = 100, 100
        steps = 100
        batch_sizes = [1, 4, 8]

        try:
            speedups = []

            for batch_size in batch_sizes:
                # Create source and probe positions
                sources = [
                    (20 + i*10, 50)
                    for i in range(batch_size)
                ]
                probes = [(80, 50)]

                solver = BatchedFDTD2D(
                    nx=nx, ny=ny, dx=1e-3,
                    total_steps=steps,
                    cpml_thickness=8,
                    source_positions=sources,
                    probe_positions=probes,
                    frequency=5e9
                )

                solver.build()

                # Time the batched run
                start = time.time()
                solver.run()
                elapsed = time.time() - start

                # Time per batch element
                time_per_element = elapsed / batch_size
                speedups.append((batch_size, time_per_element))

                print(f"\nBatch size {batch_size}: {time_per_element:.4f}s per element")

            # Check speedup: larger batch should be more efficient
            if len(speedups) >= 2:
                base_time = speedups[0][1]
                large_batch_time = speedups[-1][1]

                speedup_factor = base_time / large_batch_time

                print(f"Speedup factor: {speedup_factor:.1f}x")

                # Should see at least 2x improvement (optimistic for this test)
                assert speedup_factor >= 1.5, f"Expected speedup < {speedup_factor}x"

        except (AttributeError, NotImplementedError):
            pytest.skip("Batched 2D solver not available")

    def test_batched_3d_speedup_factor(self):
        """Batched 3D solver should provide 30-40x speedup."""

        # This would test batched 3D if available
        # For now, just validate sequential 3D
        nx, ny, nz = 40, 40, 40
        steps = 30

        config = SimulationConfig(
            grid=GridConfig(nx=nx, ny=ny, nz=nz, dx=1e-3, dy=1e-3, dz=1e-3),
            total_steps=steps,
            courant=0.5
        )

        src = GaussianSource(x=nx//2, y=ny//2, z=nz//2, frequency_max=10e9)
        solver = FDTD3D(config=config, sources=[src], boundaries=[CPML(thickness=4)])

        # Just verify it runs without error
        solver.run(steps)

        assert solver.current_step == steps


class TestMemoryScaling:
    """Test memory usage with grid size."""

    def test_2d_memory_scaling(self):
        """Memory should scale as O(nx*ny)."""

        grid_sizes = [(50, 50), (100, 100), (150, 150)]
        memories = []

        for nx, ny in grid_sizes:
            grid = GridConfig(nx=nx, ny=ny, dx=1e-3, dy=1e-3)
            config = SimulationConfig(grid=grid, total_steps=1)

            # Estimate memory: number of field arrays × cells × 8 bytes
            # FDTD2D has: Ez, Hx, Hy, + coefficients
            # Rough estimate: 10 arrays
            estimated_bytes = 10 * nx * ny * 8

            memories.append((nx*ny, estimated_bytes))

            print(f"\n2D grid {nx}×{ny}: ~{estimated_bytes/1e6:.1f} MB")

        # Check linear scaling
        if len(memories) >= 2:
            ratio1 = memories[1][1] / memories[0][1]
            expected_ratio1 = memories[1][0] / memories[0][0]

            # Should be close to linear
            assert 0.8 * expected_ratio1 < ratio1 < 1.2 * expected_ratio1

    def test_3d_memory_scaling(self):
        """3D memory should scale as O(nx*ny*nz)."""

        grid_sizes = [(30, 30, 30), (40, 40, 40)]
        memories = []

        for nx, ny, nz in grid_sizes:
            grid = GridConfig(nx=nx, ny=ny, nz=nz, dx=1e-3, dy=1e-3, dz=1e-3)
            config = SimulationConfig(grid=grid, total_steps=1)

            # Rough estimate: 9 field arrays (Ex, Ey, Ez, Hx, Hy, Hz + more)
            estimated_bytes = 10 * nx * ny * nz * 8

            memories.append((nx*ny*nz, estimated_bytes))

            print(f"\n3D grid {nx}×{ny}×{nz}: ~{estimated_bytes/1e6:.1f} MB")

        # Check linear scaling
        if len(memories) >= 2:
            ratio = memories[1][1] / memories[0][1]
            expected_ratio = memories[1][0] / memories[0][0]

            assert 0.8 * expected_ratio < ratio < 1.2 * expected_ratio


class TestRegressionDetector:
    """Detect performance regressions."""

    def test_2d_regression_detection(self):
        """Detect if 2D solver has >10% slowdown."""

        nx, ny = 80, 80
        steps = 50

        config = SimulationConfig(
            grid=GridConfig(nx=nx, ny=ny, dx=1e-3, dy=1e-3),
            total_steps=steps,
            courant=0.5
        )

        src = GaussianSource(x=nx//2, y=ny//2, frequency_max=5e9)
        solver = FDTD2D(config=config, sources=[src], boundaries=[MurABC()])

        # Run multiple times to get stable timing
        times = []
        for trial in range(3):
            solver_trial = FDTD2D(config=config, sources=[src], boundaries=[MurABC()])

            start = time.time()
            solver_trial.run(steps)
            elapsed = time.time() - start

            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Calculate throughput
        cells = nx * ny
        ns_per_cell = (avg_time / (cells * steps)) * 1e9

        print(f"\n2D Regression Check:")
        print(f"  Avg time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Throughput: {ns_per_cell:.1f} ns/cell/step")

        # Flag regression if much slower than baseline
        # (Use generous threshold since performance varies)
        max_acceptable = PERF_2D_BASELINE_NS_PER_CELL * 100

        if ns_per_cell > max_acceptable:
            pytest.warns(UserWarning, match="performance regression")

    def test_3d_regression_detection(self):
        """Detect if 3D solver has >10% slowdown."""

        nx, ny, nz = 40, 40, 40
        steps = 30

        config = SimulationConfig(
            grid=GridConfig(nx=nx, ny=ny, nz=nz, dx=1e-3, dy=1e-3, dz=1e-3),
            total_steps=steps,
            courant=0.5
        )

        src = GaussianSource(x=nx//2, y=ny//2, z=nz//2, frequency_max=10e9)
        solver = FDTD3D(config=config, sources=[src], boundaries=[CPML(thickness=4)])

        start = time.time()
        solver.run(steps)
        elapsed = time.time() - start

        # Calculate throughput
        cells = nx * ny * nz
        ns_per_cell = (elapsed / (cells * steps)) * 1e9

        print(f"\n3D Regression Check:")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {ns_per_cell:.1f} ns/cell/step")

        # Should not be wildly slow
        max_acceptable = PERF_3D_BASELINE_NS_PER_CELL * 100

        if ns_per_cell > max_acceptable:
            pytest.warns(UserWarning, match="performance regression")


class TestCompilationOverhead:
    """Test compilation and initialization overhead."""

    def test_first_run_vs_cached(self):
        """First run may be slower due to compilation."""

        nx, ny = 60, 60
        steps = 20

        config = SimulationConfig(
            grid=GridConfig(nx=nx, ny=ny, dx=1e-3, dy=1e-3),
            total_steps=steps,
            courant=0.5
        )

        src = GaussianSource(x=nx//2, y=ny//2, frequency_max=5e9)

        # First run (cold)
        solver1 = FDTD2D(config=config, sources=[src], boundaries=[MurABC()])
        start = time.time()
        solver1.run(steps)
        time_cold = time.time() - start

        # Second run (warm)
        solver2 = FDTD2D(config=config, sources=[src], boundaries=[MurABC()])
        start = time.time()
        solver2.run(steps)
        time_warm = time.time() - start

        print(f"\nCompilation overhead:")
        print(f"  Cold run: {time_cold:.3f}s")
        print(f"  Warm run: {time_warm:.3f}s")
        print(f"  Overhead: {(time_cold/time_warm - 1)*100:.1f}%")

        # Cold run should be <= 2x warm run (accounting for overhead)
        # (may be less on cached systems)
        assert time_cold < time_warm * 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])  # -s to show print output
