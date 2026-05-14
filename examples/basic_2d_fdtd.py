#!/usr/bin/env python3
"""
Basic 2D FDTD Simulation — Free Space Propagation with Gaussian Pulse
=====================================================================

Demonstrates NeuroWave's core FDTD capabilities:
- 200×200 TMz grid (free space)
- Gaussian pulse source at center
- CPML absorbing boundaries
- Field snapshots saved as images

This is the simplest possible FDTD simulation — a point source
radiating in free space with PML absorption at boundaries.

Usage:
    python examples/basic_2d_fdtd.py
"""

import sys
import os
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ceep.core.config import GridConfig, SimulationConfig, SimulationMode, Backend
from ceep.sources.waveforms import GaussianSource
from ceep.boundaries.absorbing import CPML, MurABC
from ceep.solvers.fdtd_2d import FDTD2D
from ceep.visualization.field_plot import (
    plot_field_2d,
    plot_field_snapshots,
    plot_source_waveform,
)


def main() -> None:
    """Run basic 2D FDTD simulation."""
    print("=" * 60)
    print("NeuroWave — Basic 2D FDTD Simulation")
    print("=" * 60)
    print()

    # --- Configuration ---
    # 200×200 grid, 1mm spacing, free space
    grid = GridConfig(nx=200, ny=200, dx=1e-3, dy=1e-3)

    config = SimulationConfig(
        grid=grid,
        mode=SimulationMode.TMZ,
        courant=0.5,
        total_steps=300,
        backend=Backend.NUMPY,
    )

    print(config.summary())
    print()

    # --- Source ---
    # Gaussian pulse at grid center, max frequency 5 GHz
    source = GaussianSource(
        x=100, y=100,
        frequency_max=5e9,
        amplitude=1.0,
    )
    print(f"Source: Gaussian pulse at ({source.x}, {source.y})")
    print(f"  τ = {source.tau*1e12:.1f} ps, t₀ = {source.t0*1e12:.1f} ps")
    print(f"  f_max = 5 GHz, λ_min = {3e8/5e9*1e3:.1f} mm")
    print(f"  Points per wavelength: {3e8/5e9/config.grid.dx:.0f}")
    print()

    # --- Boundary ---
    # CPML with 10 cells
    boundary = CPML(thickness=10, order=3)
    print(f"Boundary: CPML (thickness={boundary.thickness}, order={boundary.order})")
    print()

    # --- Solver ---
    solver = FDTD2D(
        config=config,
        sources=[source],
        boundaries=[boundary],
        record_field="Ez",
        record_interval=10,  # Save snapshot every 10 steps
    )

    print(f"Grid memory: {solver.grid.memory_usage_mb:.1f} MB")
    print(f"Running {config.num_steps} timesteps...")
    print()

    # --- Run simulation ---
    start = time.perf_counter()
    solver.run()
    elapsed = time.perf_counter() - start

    print(f"Simulation completed in {elapsed:.2f} s")
    print(f"  Steps/sec: {config.num_steps / elapsed:.0f}")
    print(f"  Cell-steps/sec: {config.num_steps * grid.total_cells / elapsed:.2e}")
    print(f"  Snapshots recorded: {len(solver.field_snapshots)}")
    print()

    # --- Visualization ---
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Plot final field
    fig = plot_field_2d(
        solver.get_field("Ez"),
        title=f"Ez Field — Step {solver.current_step}",
        dx=config.grid.dx,
        dy=config.grid.dy,
        save_path=os.path.join(output_dir, "ez_final.png"),
    )
    plt_close(fig)
    print(f"Saved: {output_dir}/ez_final.png")

    # Plot source waveform
    waveform = source.waveform(config.num_steps, config.dt)
    fig = plot_source_waveform(
        waveform, config.dt,
        title="Gaussian Pulse",
        save_path=os.path.join(output_dir, "source_waveform.png"),
    )
    plt_close(fig)
    print(f"Saved: {output_dir}/source_waveform.png")

    # Plot snapshots grid
    if solver.field_snapshots:
        # Pick 8 evenly spaced snapshots
        n_snaps = len(solver.field_snapshots)
        indices = np.linspace(0, n_snaps - 1, min(8, n_snaps), dtype=int)
        selected = [solver.field_snapshots[i] for i in indices]
        step_nums = [i * solver.record_interval for i in indices]

        fig = plot_field_snapshots(
            selected,
            step_indices=step_nums,
            ncols=4,
            dx=config.grid.dx,
            dy=config.grid.dy,
            save_path=os.path.join(output_dir, "ez_snapshots.png"),
        )
        plt_close(fig)
        print(f"Saved: {output_dir}/ez_snapshots.png")

    print()
    print("Simulation complete! ✅")


def plt_close(fig) -> None:
    """Close a matplotlib figure."""
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
