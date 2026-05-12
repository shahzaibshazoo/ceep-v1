#!/usr/bin/env python3
"""
PML Reflection Coefficient Benchmark
====================================

Measures the reflection coefficient of CPML by comparing a test grid
against a large reference grid where boundary reflections haven't
arrived yet.

Physics:
    R(t) = E_test(t) - E_ref(t)
    Reflection (dB) = 20 * log10( max|R(t)| / max|E_ref(t)| )
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from neurowave.core.config import GridConfig, SimulationConfig, SimulationMode
from neurowave.sources.waveforms import GaussianSource
from neurowave.boundaries.absorbing import CPML
from neurowave.solvers.fdtd_2d import FDTD2D


def run_simulation(nx: int, thickness: int, order: int, sigma_factor: float, steps: int) -> np.ndarray:
    """Run simulation and return Ez field at probe point (100, 50)."""
    grid = GridConfig(nx=nx, ny=100, dx=1e-3, dy=1e-3)
    config = SimulationConfig(
        grid=grid, mode=SimulationMode.TMZ, courant=0.5, total_steps=steps
    )
    
    # Source at x=50, y=50
    source = GaussianSource(x=50, y=50, frequency_max=5e9, amplitude=1.0)
    
    # CPML boundary
    cpml = CPML(thickness=thickness, order=order, sigma_factor=sigma_factor)
    
    # Probe at x=100, y=50
    solver = FDTD2D(
        config=config, 
        sources=[source], 
        boundaries=[cpml],
        probe_points=[(100, 50)]
    )
    solver.run()
    
    return np.array(solver.probe_data[(100, 50)])


def main() -> None:
    print("=" * 60)
    print("NeuroWave — CPML Reflection Benchmark")
    print("=" * 60)
    
    steps = 400
    
    # 1. Reference simulation (large grid, reflections don't reach probe)
    print("Running reference simulation (nx=800)...")
    start = time.perf_counter()
    ref_sig = run_simulation(nx=800, thickness=10, order=3, sigma_factor=1.5, steps=steps)
    ref_max = np.max(np.abs(ref_sig))
    print(f"Reference done in {time.perf_counter() - start:.2f}s. Max amplitude: {ref_max:.4e}\n")
    
    # 2. Test different CPML parameters
    print(f"{'Thickness':>10} | {'Order':>6} | {'Sigma':>6} | {'Refl (dB)':>10}")
    print("-" * 45)
    
    best_db = 0.0
    best_params = None
    
    for t in [5, 10, 15, 20]:
        for order in [3, 4]:
            for sf in [1.0, 1.5, 2.0]:
                # Grid size: 100 (probe) + 5 (gap) + t (PML)
                test_nx = 105 + t
                test_sig = run_simulation(
                    nx=test_nx, thickness=t, order=order, sigma_factor=sf, steps=steps
                )
                
                # Compute reflection
                diff = test_sig - ref_sig
                max_diff = np.max(np.abs(diff))
                
                if max_diff == 0:
                    refl_db = -200.0
                else:
                    refl_db = 20 * np.log10(max_diff / ref_max)
                
                print(f"{t:>10} | {order:>6} | {sf:>6.1f} | {refl_db:>10.2f}")
                
                if refl_db < best_db:
                    best_db = refl_db
                    best_params = (t, order, sf)
                    
    print("-" * 45)
    print(f"Best performance: {best_db:.2f} dB with thickness={best_params[0]}, "
          f"order={best_params[1]}, sigma_factor={best_params[2]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
