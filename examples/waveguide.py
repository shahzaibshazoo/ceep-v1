#!/usr/bin/env python3
"""
Parallel-Plate Waveguide Example
================================

Simulates EM wave propagation in a PEC-bounded parallel-plate waveguide.

Physics:
    Two PEC walls create a waveguide. Only modes with cutoff
    frequency below the source frequency will propagate.

    For a waveguide of width d, the cutoff for TMz mode m is:
        f_c(m) = m·c / (2·d)

    TM₁ cutoff for d=40mm: f_c = 3e8/(2×0.04) = 3.75 GHz

Usage:
    python examples/waveguide.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neurowave.core.config import GridConfig, SimulationConfig, SimulationMode
from neurowave.sources.waveforms import SinusoidalSource
from neurowave.boundaries.absorbing import PEC, CPML
from neurowave.solvers.fdtd_2d import FDTD2D
from neurowave.visualization.field_plot import plot_field_2d, plot_field_snapshots


def main() -> None:
    print("=" * 60)
    print("NeuroWave — Parallel-Plate Waveguide")
    print("=" * 60)

    # Grid: long and narrow (waveguide shape)
    grid = GridConfig(nx=400, ny=80, dx=0.5e-3, dy=0.5e-3)
    config = SimulationConfig(
        grid=grid, mode=SimulationMode.TMZ,
        courant=0.5, total_steps=800,
    )
    print(config.summary())

    # CW source at 5 GHz (above TM1 cutoff of 3.75 GHz)
    source = SinusoidalSource(x=20, y=40, frequency=5e9, amplitude=1.0)

    # CPML at x-ends, PEC at y-walls (waveguide walls)
    cpml = CPML(thickness=10, order=3)

    solver = FDTD2D(
        config=config, sources=[source], boundaries=[cpml],
        record_field="Ez", record_interval=100,
    )

    # PEC walls at y=0 and y=ny-1 (applied manually each step via material)
    # Actually, let's set high conductivity walls
    solver.grid.set_material_region(0, 400, 0, 5, sigma_e=1e6)
    solver.grid.set_material_region(0, 400, 75, 80, sigma_e=1e6)

    print(f"\nWaveguide width: {70 * 0.5} mm")
    print(f"TM1 cutoff: {3e8 / (2 * 70 * 0.5e-3) / 1e9:.2f} GHz")
    print(f"Source frequency: 5.0 GHz (above cutoff → propagating)")

    start = time.perf_counter()
    solver.run()
    elapsed = time.perf_counter() - start
    print(f"\nCompleted in {elapsed:.2f}s")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    plot_field_2d(
        solver.get_field("Ez"),
        title="Waveguide — Ez at 5 GHz",
        dx=config.grid.dx, dy=config.grid.dy,
        save_path=os.path.join(output_dir, "waveguide_final.png"),
    )

    if solver.field_snapshots:
        n = len(solver.field_snapshots)
        indices = list(range(n))
        plot_field_snapshots(
            [solver.field_snapshots[i] for i in indices],
            step_indices=[i * 100 for i in indices],
            dx=config.grid.dx, dy=config.grid.dy,
            save_path=os.path.join(output_dir, "waveguide_snapshots.png"),
        )

    print(f"Outputs saved to {output_dir}/")
    print("Done! ✅")


if __name__ == "__main__":
    main()
