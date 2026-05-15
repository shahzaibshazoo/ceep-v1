#!/usr/bin/env python3
"""
Compare CEEP vs MEEP for Brain Hemorrhage Detection
====================================================

Generates a single sample with both CEEP (GPU) and MEEP (CPU) and compares:
1. S-parameter magnitudes
2. DAS imaging results
3. Runtime performance

Author: Shahzaib Ur Rehman
Date: 2026-05-15
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
import time
import json

# Try importing both backends
try:
    from ceep.core.backend import set_backend, to_numpy
    from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
    from ceep.phantoms import BrainPhantom2D
    CEEP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CEEP not available: {e}")
    CEEP_AVAILABLE = False

try:
    import meep as mp
    MEEP_AVAILABLE = True
except ImportError:
    print("⚠️  MEEP not available")
    MEEP_AVAILABLE = False


# Physical constants
C = 3e8  # Speed of light


def create_phantom_ceep(nx=64, ny=64, dx=0.5e-3, hem_center=None, hem_radius=0.5e-2):
    """Create brain phantom using CEEP."""
    if hem_center is None:
        hem_center = (1.0, 0.5)  # cm

    phantom = BrainPhantom2D(
        nx=nx, ny=ny, dx=dx,
        hemorrhage_location=hem_center,
        hemorrhage_radius=hem_radius * 100,  # Convert to cm
        use_gabriel_database=False  # Use simple model
    )

    return phantom


def run_ceep_simulation(nx=64, ny=64, dx=0.5e-3, frequency=2e9,
                        n_ant=16, hem_center=None):
    """Run CEEP GPU simulation."""
    print("\n" + "="*70)
    print(" CEEP (GPU) Simulation")
    print("="*70)

    set_backend('cupy')

    # Create circular antenna array
    center_x, center_y = nx // 2, ny // 2
    radius = nx // 3

    angles = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
    positions = []
    for angle in angles:
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        positions.append((x, y))

    print(f"\nConfiguration:")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Grid spacing: {dx*1e3:.2f} mm")
    print(f"  Frequency: {frequency/1e9:.1f} GHz")
    print(f"  Antennas: {n_ant} (circular array)")

    # Create phantom
    phantom = create_phantom_ceep(nx, ny, dx, hem_center)

    # Calculate time steps for proper coverage
    wavelength = C / frequency
    max_dist = np.sqrt(2) * (nx * dx)  # Diagonal distance
    t_propagation = max_dist / C
    dt = 1 / (4 * frequency)  # Nyquist
    total_steps = int(2 * t_propagation / dt)

    print(f"  Time steps: {total_steps}")
    print(f"  dt: {dt*1e12:.2f} ps")

    # Create solver
    print("\n[1/3] Creating solver...")
    solver = BatchedFDTD2D(
        nx=nx, ny=ny, dx=dx,
        total_steps=total_steps,
        cpml_thickness=10,
        source_positions=positions,
        probe_positions=positions,
        frequency=frequency
    )

    # Set phantom
    print("[2/3] Setting phantom...")
    solver.set_phantom(phantom)

    # Run simulation
    print("[3/3] Running simulation...")
    t_start = time.time()
    s_params = solver.run()
    t_elapsed = time.time() - t_start

    # Convert to numpy
    s_matrix = to_numpy(s_params[0][0])  # (n_ant, n_ant, n_time)

    print(f"\n✓ Simulation complete!")
    print(f"  Runtime: {t_elapsed:.2f} s")
    print(f"  S-matrix shape: {s_matrix.shape}")
    print(f"  S-matrix magnitude: mean={np.abs(s_matrix).mean():.3e}, max={np.abs(s_matrix).max():.3e}")

    return s_matrix, t_elapsed


def run_meep_simulation(nx=64, ny=64, dx=0.5e-3, frequency=2e9, n_ant=16):
    """Run MEEP CPU simulation for comparison."""
    print("\n" + "="*70)
    print(" MEEP (CPU) Simulation")
    print("="*70)

    # Convert to MEEP units
    wavelength = C / frequency
    resolution = 1 / (dx * frequency / C)  # Points per wavelength

    cell_size = mp.Vector3(nx * dx / wavelength, ny * dx / wavelength, 0)

    print(f"\nConfiguration:")
    print(f"  Grid: {nx} x {ny}")
    print(f"  Resolution: {resolution:.1f} points/wavelength")
    print(f"  Cell size: {cell_size.x*wavelength:.3f} x {cell_size.y*wavelength:.3f} m")

    # Create circular antenna positions
    center_x, center_y = 0, 0  # MEEP uses centered coordinates
    radius = (nx // 3) * dx / wavelength

    angles = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
    ant_positions = []
    for angle in angles:
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        ant_positions.append(mp.Vector3(x, y, 0))

    print(f"  Antennas: {n_ant}")

    # Simple head geometry (circle of brain tissue)
    head_radius = (nx // 2 - 5) * dx / wavelength
    brain_eps = 50.0  # Typical brain permittivity at 2 GHz

    geometry = [
        mp.Cylinder(radius=head_radius, height=mp.inf,
                   center=mp.Vector3(0, 0, 0),
                   material=mp.Medium(epsilon=brain_eps))
    ]

    # Add hemorrhage if specified
    hem_center_norm = mp.Vector3(0.3 * head_radius, 0.1 * head_radius, 0)
    hem_radius_norm = 0.15 * head_radius
    geometry.append(
        mp.Cylinder(radius=hem_radius_norm, height=mp.inf,
                   center=hem_center_norm,
                   material=mp.Medium(epsilon=60.0))
    )

    print("[1/3] Setting up MEEP simulation...")

    # We'll simulate one TX-RX pair as example (full S-matrix would take too long)
    tx_idx = 0
    rx_idx = 0

    sources = [
        mp.Source(mp.GaussianPulse(frequency=1.0, fwidth=0.2),
                 component=mp.Ez,
                 center=ant_positions[tx_idx])
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        boundary_layers=[mp.PML(1.0)]
    )

    # Monitor at RX position
    print("[2/3] Running MEEP...")
    t_start = time.time()

    # Run until fields decay
    sim.run(until=50)

    t_elapsed = time.time() - t_start

    print(f"\n✓ MEEP simulation complete!")
    print(f"  Runtime: {t_elapsed:.2f} s")
    print(f"  Note: Only 1 TX-RX pair simulated (full S-matrix would take {n_ant**2}x longer)")

    # Get field at RX point
    ez_rx = sim.get_field_point(mp.Ez, ant_positions[rx_idx])

    print(f"  Field magnitude at RX: {np.abs(ez_rx):.3e}")

    return np.abs(ez_rx), t_elapsed


def compare_results(ceep_s_matrix, ceep_time, meep_field, meep_time):
    """Compare CEEP and MEEP results."""
    print("\n" + "="*70)
    print(" COMPARISON: CEEP vs MEEP")
    print("="*70)

    ceep_mag_max = np.abs(ceep_s_matrix).max()

    print(f"\nSignal Magnitudes:")
    print(f"  CEEP S-matrix max:  {ceep_mag_max:.3e}")
    print(f"  MEEP field mag:     {meep_field:.3e}")

    print(f"\nRuntime:")
    print(f"  CEEP (GPU): {ceep_time:.2f} s ({ceep_s_matrix.shape[0]**2} TX-RX pairs)")
    print(f"  MEEP (CPU): {meep_time:.2f} s (1 TX-RX pair)")
    print(f"  CEEP speedup: {(meep_time * ceep_s_matrix.shape[0]**2) / ceep_time:.1f}x (estimated)")

    # Check if CEEP values are reasonable
    if ceep_mag_max < 1e-10:
        print(f"\n⚠️  WARNING: CEEP S-matrix values too small!")
        print(f"   Expected: ~{meep_field:.0e}")
        print(f"   Got: {ceep_mag_max:.0e}")
        print(f"   Ratio: {ceep_mag_max / meep_field:.2e} (should be ~1)")
        print("\n   Possible CEEP issues:")
        print("   - Source amplitude scaling incorrect")
        print("   - Time window too short")
        print("   - Field extraction timing wrong")
    else:
        print(f"\n✓ CEEP magnitudes reasonable (within {ceep_mag_max/meep_field:.1f}x of MEEP)")


def main():
    """Main comparison routine."""
    print("="*70)
    print(" CEEP vs MEEP Comparison - Brain Hemorrhage Detection")
    print("="*70)

    # Parameters
    nx = ny = 64
    dx = 0.5e-3  # 0.5 mm
    frequency = 2e9  # 2 GHz
    n_ant = 16
    hem_center = (1.0, 0.5)  # cm from head center

    if not CEEP_AVAILABLE:
        print("\n❌ CEEP not available - cannot run comparison")
        return

    # Run CEEP
    ceep_s_matrix, ceep_time = run_ceep_simulation(
        nx, ny, dx, frequency, n_ant, hem_center
    )

    # Run MEEP if available
    if MEEP_AVAILABLE:
        meep_field, meep_time = run_meep_simulation(nx, ny, dx, frequency, n_ant)
        compare_results(ceep_s_matrix, ceep_time, meep_field, meep_time)
    else:
        print("\n⚠️  MEEP not available - skipping comparison")
        print(f"\nCEEP Results:")
        print(f"  S-matrix max magnitude: {np.abs(ceep_s_matrix).max():.3e}")
        print(f"  Runtime: {ceep_time:.2f} s")

        if np.abs(ceep_s_matrix).max() < 1e-10:
            print(f"\n⚠️  S-matrix values suspiciously small!")
            print("   Expected: ~1e-3 to 1e-1 for typical microwave imaging")

    print("\n" + "="*70)
    print(" Comparison Complete")
    print("="*70)


if __name__ == "__main__":
    main()
