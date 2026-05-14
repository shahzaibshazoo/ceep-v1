#!/usr/bin/env python
"""
GPU vs CPU Performance Benchmark
=================================

Compares FDTD solver performance between NumPy (CPU) and CuPy (GPU) backends
across multiple grid sizes.

Usage:
    PYTHONPATH=./src python benchmarks/gpu_vs_cpu.py

Requirements:
    - CuPy with a compatible NVIDIA GPU
    - pip install cupy-cuda12x  (or cupy-cuda11x for older GPUs)

Output:
    Table showing execution time, cell-steps/s, and speedup for each grid size.
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'src')

from neurowave.core.backend import is_backend_available, Backend, set_backend
from neurowave.core.config import GridConfig, SimulationConfig
from neurowave.sources.waveforms import GaussianSource


def benchmark_2d(nx, ny, steps, backend_name):
    """Run a 2D TMz benchmark and return timing info."""
    from neurowave.solvers.fdtd_2d import FDTD2D
    from neurowave.boundaries.absorbing import CPML

    set_backend(backend_name)

    grid_config = GridConfig(nx=nx, ny=ny, dx=1e-3, dy=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=steps)
    source = GaussianSource(
        x=nx // 2, y=ny // 2, frequency_max=10e9,
        field_component='Ez', delay_factor=5.0
    )
    cpml = CPML(thickness=10)

    solver = FDTD2D(config=config, sources=[source], boundaries=[cpml])

    # Warmup (for GPU JIT)
    if backend_name == 'cupy':
        import cupy as cp
        for _ in range(5):
            solver.step()
        cp.cuda.Device().synchronize()
        # Reset
        solver = FDTD2D(config=config, sources=[source], boundaries=[cpml])

    # Timed run
    if backend_name == 'cupy':
        import cupy as cp
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        solver.run()
        cp.cuda.Device().synchronize()
    else:
        t0 = time.perf_counter()
        solver.run()

    elapsed = time.perf_counter() - t0
    cell_steps = nx * ny * steps
    rate = cell_steps / elapsed / 1e6  # Mcell-steps/s

    return elapsed, rate


def benchmark_3d(nx, ny, nz, steps, backend_name):
    """Run a 3D benchmark and return timing info."""
    from neurowave.solvers.fdtd_3d import FDTD3D

    set_backend(backend_name)

    grid_config = GridConfig(nx=nx, ny=ny, nz=nz, dx=1e-3, dy=1e-3, dz=1e-3)
    config = SimulationConfig(grid=grid_config, total_steps=steps)
    source = GaussianSource(
        x=nx // 2, y=ny // 2, z=nz // 2,
        frequency_max=10e9, field_component='Ez', delay_factor=5.0
    )

    solver = FDTD3D(config, sources=[source])

    # Warmup
    if backend_name == 'cupy':
        import cupy as cp
        for _ in range(3):
            solver.step()
        cp.cuda.Device().synchronize()
        solver = FDTD3D(config, sources=[source])

    if backend_name == 'cupy':
        import cupy as cp
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        solver.run()
        cp.cuda.Device().synchronize()
    else:
        t0 = time.perf_counter()
        solver.run()

    elapsed = time.perf_counter() - t0
    cell_steps = nx * ny * nz * steps
    rate = cell_steps / elapsed / 1e6

    return elapsed, rate


def main():
    has_gpu = is_backend_available(Backend.CUPY)

    print("=" * 78)
    print("  NeuroWave FDTD Benchmark: CPU (NumPy) vs GPU (CuPy)")
    print("=" * 78)

    if has_gpu:
        import cupy as cp
        dev = cp.cuda.Device()
        print(f"  GPU: {dev.attributes}")
        mem_total = dev.mem_info[1] / 1e9
        print(f"  GPU Memory: {mem_total:.1f} GB")
    else:
        print("  GPU: Not available (CuPy not installed)")
        print("  Running CPU-only benchmarks...")

    print()

    # 2D Benchmarks
    print("-" * 78)
    print("  2D TMz FDTD (with CPML)")
    print("-" * 78)
    print(f"  {'Grid Size':<16} {'Steps':<8} {'CPU (s)':<10} {'GPU (s)':<10} "
          f"{'CPU Rate':<14} {'GPU Rate':<14} {'Speedup':<8}")
    print(f"  {'─' * 14:<16} {'─' * 6:<8} {'─' * 8:<10} {'─' * 8:<10} "
          f"{'─' * 12:<14} {'─' * 12:<14} {'─' * 6:<8}")

    grid_sizes_2d = [
        (100, 100, 200),
        (200, 200, 200),
        (500, 500, 100),
        (1000, 1000, 50),
    ]

    if has_gpu:
        grid_sizes_2d.append((2000, 2000, 20))

    for nx, ny, steps in grid_sizes_2d:
        cpu_time, cpu_rate = benchmark_2d(nx, ny, steps, 'numpy')

        if has_gpu:
            gpu_time, gpu_rate = benchmark_2d(nx, ny, steps, 'cupy')
            speedup = cpu_time / gpu_time
            print(f"  {nx}x{ny:<11} {steps:<8} {cpu_time:<10.3f} {gpu_time:<10.3f} "
                  f"{cpu_rate:<14.1f} {gpu_rate:<14.1f} {speedup:<8.1f}x")
        else:
            print(f"  {nx}x{ny:<11} {steps:<8} {cpu_time:<10.3f} {'N/A':<10} "
                  f"{cpu_rate:<14.1f} {'N/A':<14} {'N/A':<8}")

    # 3D Benchmarks
    print()
    print("-" * 78)
    print("  3D FDTD (free space)")
    print("-" * 78)
    print(f"  {'Grid Size':<16} {'Steps':<8} {'CPU (s)':<10} {'GPU (s)':<10} "
          f"{'CPU Rate':<14} {'GPU Rate':<14} {'Speedup':<8}")
    print(f"  {'─' * 14:<16} {'─' * 6:<8} {'─' * 8:<10} {'─' * 8:<10} "
          f"{'─' * 12:<14} {'─' * 12:<14} {'─' * 6:<8}")

    grid_sizes_3d = [
        (30, 30, 30, 100),
        (50, 50, 50, 50),
        (80, 80, 80, 30),
        (100, 100, 100, 20),
    ]

    if has_gpu:
        grid_sizes_3d.append((150, 150, 150, 10))

    for nx, ny, nz, steps in grid_sizes_3d:
        cpu_time, cpu_rate = benchmark_3d(nx, ny, nz, steps, 'numpy')

        if has_gpu:
            gpu_time, gpu_rate = benchmark_3d(nx, ny, nz, steps, 'cupy')
            speedup = cpu_time / gpu_time
            print(f"  {nx}x{ny}x{nz:<8} {steps:<8} {cpu_time:<10.3f} {gpu_time:<10.3f} "
                  f"{cpu_rate:<14.1f} {gpu_rate:<14.1f} {speedup:<8.1f}x")
        else:
            print(f"  {nx}x{ny}x{nz:<8} {steps:<8} {cpu_time:<10.3f} {'N/A':<10} "
                  f"{cpu_rate:<14.1f} {'N/A':<14} {'N/A':<8}")

    print()
    print("=" * 78)
    print("  Rates in Mcell-steps/s (higher is better)")
    if has_gpu:
        print("  GPU speedup increases with grid size (larger grids = better GPU utilization)")
    print("=" * 78)

    # Reset to numpy
    set_backend('numpy')


if __name__ == '__main__':
    main()
