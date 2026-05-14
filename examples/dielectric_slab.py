#!/usr/bin/env python3
"""
Dielectric Slab Example — Gaussian Pulse Hitting a Dielectric Interface
=======================================================================

Demonstrates:
- Material property assignment (dielectric slab with eps_r=4)
- Wave reflection and transmission at dielectric interface
- Slower wave propagation inside dielectric
- CPML boundaries absorbing transmitted and reflected waves

Physics:
    At a dielectric interface, the reflection coefficient is:
        Γ = (η₂ - η₁) / (η₂ + η₁)
    
    For eps_r=4: η₂ = η₀/√4 = η₀/2
        Γ = (η₀/2 - η₀) / (η₀/2 + η₀) = -1/3 ≈ -0.333

Usage:
    python examples/dielectric_slab.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ceep.core.config import GridConfig, SimulationConfig, SimulationMode
from ceep.sources.waveforms import GaussianSource
from ceep.boundaries.absorbing import CPML
from ceep.solvers.fdtd_2d import FDTD2D
from ceep.visualization.field_plot import plot_field_2d, plot_field_snapshots


def main() -> None:
    print("=" * 60)
    print("NeuroWave — Dielectric Slab Example")
    print("=" * 60)

    grid = GridConfig(nx=300, ny=200, dx=0.5e-3, dy=0.5e-3)
    config = SimulationConfig(
        grid=grid, mode=SimulationMode.TMZ,
        courant=0.5, total_steps=500,
    )
    print(config.summary())

    # Source: Gaussian pulse on the left side
    source = GaussianSource(x=50, y=100, frequency_max=5e9, amplitude=1.0)

    # CPML boundaries
    boundary = CPML(thickness=12, order=3)

    # Create solver
    solver = FDTD2D(
        config=config, sources=[source], boundaries=[boundary],
        record_field="Ez", record_interval=50,
        probe_points=[(50, 100), (150, 100), (250, 100)],
    )

    # Add dielectric slab: eps_r=4.0 from x=130 to x=170
    solver.grid.set_material_region(130, 170, 0, 200, eps_r=4.0)
    print(f"\nDielectric slab: x=[130,170], eps_r=4.0")
    print(f"Expected reflection coeff: Γ = -1/3 ≈ -0.333")
    print(f"Wave speed in slab: c/2 = {3e8/2:.2e} m/s")

    # Run
    start = time.perf_counter()
    solver.run()
    elapsed = time.perf_counter() - start
    print(f"\nCompleted in {elapsed:.2f}s ({config.num_steps/elapsed:.0f} steps/s)")

    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Final field
    plot_field_2d(
        solver.get_field("Ez"),
        title="Ez — Dielectric Slab (ε_r=4)",
        dx=config.grid.dx, dy=config.grid.dy,
        save_path=os.path.join(output_dir, "dielectric_slab_final.png"),
    )

    # Snapshots
    if solver.field_snapshots:
        n = len(solver.field_snapshots)
        indices = np.linspace(0, n-1, min(8, n), dtype=int)
        plot_field_snapshots(
            [solver.field_snapshots[i] for i in indices],
            step_indices=[i * 50 for i in indices],
            dx=config.grid.dx, dy=config.grid.dy,
            save_path=os.path.join(output_dir, "dielectric_slab_snapshots.png"),
        )

    print(f"Outputs saved to {output_dir}/")
    print("Done! ✅")


if __name__ == "__main__":
    main()
