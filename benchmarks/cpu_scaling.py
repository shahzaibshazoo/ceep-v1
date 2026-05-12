#!/usr/bin/env python3
"""
CPU Performance Benchmark — Grid Size Scaling
==============================================

Measures FDTD solver performance across different grid sizes to
establish baseline CPU performance metrics.

Metrics: steps/sec, cell-steps/sec, memory usage
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurowave.core.config import GridConfig, SimulationConfig, SimulationMode
from neurowave.sources.waveforms import GaussianSource
from neurowave.boundaries.absorbing import CPML
from neurowave.solvers.fdtd_2d import FDTD2D


def benchmark_grid_size(nx: int, ny: int, steps: int = 100) -> dict:
    """Run benchmark for a given grid size."""
    grid = GridConfig(nx=nx, ny=ny, dx=1e-3, dy=1e-3)
    config = SimulationConfig(grid=grid, total_steps=steps, courant=0.5)
    source = GaussianSource(x=nx // 2, y=ny // 2, frequency_max=5e9)
    boundary = CPML(thickness=8)

    solver = FDTD2D(config=config, sources=[source], boundaries=[boundary])

    start = time.perf_counter()
    solver.run()
    elapsed = time.perf_counter() - start

    cells = nx * ny
    return {
        "nx": nx,
        "ny": ny,
        "cells": cells,
        "steps": steps,
        "time_s": elapsed,
        "steps_per_sec": steps / elapsed,
        "cell_steps_per_sec": cells * steps / elapsed,
        "memory_mb": solver.grid.memory_usage_mb,
    }


def main() -> None:
    print("=" * 70)
    print("NeuroWave — CPU Performance Benchmark")
    print("=" * 70)
    print()

    grid_sizes = [
        (50, 50),
        (100, 100),
        (200, 200),
        (300, 300),
        (400, 400),
        (500, 500),
    ]
    steps = 100

    print(f"{'Grid':>10} {'Cells':>10} {'Time(s)':>10} {'Steps/s':>10} "
          f"{'MCells·Steps/s':>16} {'Mem(MB)':>10}")
    print("-" * 70)

    results = []
    for nx, ny in grid_sizes:
        r = benchmark_grid_size(nx, ny, steps)
        results.append(r)
        print(f"{nx}×{ny:>4} {r['cells']:>10,} {r['time_s']:>10.3f} "
              f"{r['steps_per_sec']:>10.0f} "
              f"{r['cell_steps_per_sec']/1e6:>16.2f} "
              f"{r['memory_mb']:>10.1f}")

    print()
    print("=" * 70)

    # Summary
    best = max(results, key=lambda x: x["cell_steps_per_sec"])
    print(f"Peak throughput: {best['cell_steps_per_sec']/1e6:.2f} M cell·steps/s "
          f"at {best['nx']}×{best['ny']}")
    print()
    print("Note: This is CPU (NumPy) baseline. GPU acceleration is planned.")
    print("=" * 70)


if __name__ == "__main__":
    main()
