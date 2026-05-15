#!/usr/bin/env python3
"""
CEEP vs MEEP Radar Comparison
==============================

Side-by-side comparison of radar beamforming results to validate
that left-right ambiguity is a fundamental limitation, not specific
to CEEP implementation.

This script:
1. Runs CEEP radar simulation (GPU-accelerated)
2. Runs MEEP radar simulation (CPU)
3. Compares beamforming spectra
4. Validates both show ±θ ambiguity

Author: Shahzaib Ur Rehman
Date: 2026-05-14
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

print("="*70)
print(" CEEP vs MEEP: Radar Beamforming Validation")
print("="*70)
print("\nThis will run identical radar experiments in both frameworks")
print("to confirm left-right ambiguity is a physics limitation.\n")

# ============================================================
# Shared Configuration
# ============================================================

FREQUENCY = 10e9
WAVELENGTH = 3e8 / FREQUENCY
NUM_ELEMENTS = 16
TARGET_RANGE = 1.5
TARGET_ANGLE = 30.0
TARGET_RADIUS = 0.05

print(f"Configuration:")
print(f"  Frequency:     {FREQUENCY/1e9} GHz")
print(f"  Array:         {NUM_ELEMENTS}-element ULA")
print(f"  Target:        {TARGET_ANGLE}° at {TARGET_RANGE} m")
print(f"  Spacing:       λ/2 = {WAVELENGTH*500:.1f} mm")
print()

# ============================================================
# PART 1: CEEP Simulation
# ============================================================

print("="*70)
print(" [1/2] RUNNING CEEP (GPU-Accelerated)")
print("="*70)
print()

try:
    from ceep.core.backend import set_backend, to_numpy
    from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

    set_backend('cupy')

    # Grid setup
    DX = WAVELENGTH / 15
    NX = NY = 2000

    # Time parameters
    C = 3e8
    DT = 0.99 * DX / (C * np.sqrt(2))
    ROUND_TRIP = 2 * TARGET_RANGE / C
    TOTAL_STEPS = int(ROUND_TRIP / DT) + 500

    # ULA positions
    center_x = NX // 2
    center_y = NY // 4
    spacing = int((WAVELENGTH / 2) / DX)
    positions = [(center_x + int((i - 7.5) * spacing), center_y)
                 for i in range(NUM_ELEMENTS)]

    # Target position
    angle_rad = np.deg2rad(TARGET_ANGLE)
    ula_center_x = center_x * DX
    ula_center_y = center_y * DX
    target_x = ula_center_x + TARGET_RANGE * np.sin(angle_rad)
    target_y = ula_center_y + TARGET_RANGE * np.cos(angle_rad)
    ix_target = int(target_x / DX)
    iy_target = int(target_y / DX)

    # Geometry
    eps_grid = np.ones((NX, NY))
    y_grid, x_grid = np.ogrid[:NX, :NY]
    radius_grid = int(TARGET_RADIUS / DX)
    mask = ((x_grid - ix_target)**2 + (y_grid - iy_target)**2 <= radius_grid**2)
    eps_grid[mask] = 1000.0

    print(f"Grid: {NX}×{NY}, {DX*1000:.2f} mm resolution")
    print(f"Timesteps: {TOTAL_STEPS}")
    print(f"Running batched FDTD...\n")

    # Run CEEP
    solver = BatchedFDTD2D(
        nx=NX, ny=NY, dx=DX,
        total_steps=TOTAL_STEPS,
        cpml_thickness=15,
        source_positions=positions,
        probe_positions=positions,
        frequency=FREQUENCY
    )
    solver._eps_r[:] = eps_grid

    t_ceep_start = time.time()
    s_matrix_ceep = solver.run()
    t_ceep = time.time() - t_ceep_start

    # Extract S-matrix
    s_matrix_ceep_array = np.zeros((NUM_ELEMENTS, NUM_ELEMENTS, TOTAL_STEPS))
    for tx in range(NUM_ELEMENTS):
        for rx in range(NUM_ELEMENTS):
            s_matrix_ceep_array[tx, rx, :] = to_numpy(s_matrix_ceep[tx][rx])

    # Beamforming (use reflection window)
    reflection_sample = int(ROUND_TRIP / solver.dt)
    window_start = max(0, reflection_sample - 100)
    window_end = min(TOTAL_STEPS, reflection_sample + 100)

    monostatic_ceep = s_matrix_ceep_array[
        np.arange(NUM_ELEMENTS), np.arange(NUM_ELEMENTS), :
    ]
    windowed_ceep = monostatic_ceep[:, window_start:window_end]

    angles = np.linspace(-90, 90, 360)
    power_ceep = []
    k = 2 * np.pi / WAVELENGTH
    pos_array = np.array(positions) * DX

    for angle in np.deg2rad(angles):
        phases = k * pos_array[:, 0] * np.sin(angle)
        a = np.exp(1j * phases) / np.sqrt(NUM_ELEMENTS)
        weighted = a[:, np.newaxis] * windowed_ceep
        p = np.abs(weighted.sum())
        power_ceep.append(p)

    power_ceep = np.array(power_ceep)
    power_ceep_db = 10 * np.log10(power_ceep / power_ceep.max() + 1e-10)

    peak_ceep_idx = np.argmax(power_ceep)
    estimated_ceep = angles[peak_ceep_idx]
    error_ceep = abs(estimated_ceep - TARGET_ANGLE)

    # Check ambiguity
    left_peak_ceep = power_ceep_db[angles < 0].max()
    right_peak_ceep = power_ceep_db[angles > 0].max()
    ambiguity_ceep = abs(left_peak_ceep - right_peak_ceep)

    print(f"✓ CEEP complete in {t_ceep:.1f}s")
    print(f"  Estimated angle: {estimated_ceep:.1f}°")
    print(f"  Error:           {error_ceep:.1f}°")
    print(f"  Left peak:       {left_peak_ceep:.1f} dB")
    print(f"  Right peak:      {right_peak_ceep:.1f} dB")
    print(f"  Ambiguity gap:   {ambiguity_ceep:.1f} dB")

    if ambiguity_ceep < 3:
        print(f"  Status:          ⚠️  AMBIGUOUS (±θ)")
    else:
        print(f"  Status:          ✓ Unambiguous")
    print()

    ceep_success = True

except Exception as e:
    print(f"❌ CEEP simulation failed: {e}\n")
    ceep_success = False
    power_ceep_db = None

# ============================================================
# PART 2: MEEP Simulation
# ============================================================

print("="*70)
print(" [2/2] RUNNING MEEP (CPU Reference)")
print("="*70)
print()

try:
    import meep as mp

    # MEEP setup (normalized units)
    wavelength_meep = 1.0
    freq_meep = 1.0
    fcen = freq_meep
    df = 0.1 * fcen

    length_scale = WAVELENGTH
    domain_size = 3.0
    sx = sy = domain_size / length_scale
    resolution_meep = 15

    # ULA positions in MEEP units
    center_x_meep = sx / 2
    center_y_meep = sy / 4
    spacing_meep = (WAVELENGTH / 2) / length_scale

    positions_meep = []
    for i in range(NUM_ELEMENTS):
        x = center_x_meep + (i - 7.5) * spacing_meep
        y = center_y_meep
        positions_meep.append((x, y))

    # Target in MEEP units
    angle_rad = np.deg2rad(TARGET_ANGLE)
    target_x_meep = center_x_meep + (TARGET_RANGE * np.sin(angle_rad)) / length_scale
    target_y_meep = center_y_meep + (TARGET_RANGE * np.cos(angle_rad)) / length_scale
    target_radius_meep = TARGET_RADIUS / length_scale

    geometry_meep = [
        mp.Sphere(
            radius=target_radius_meep,
            center=mp.Vector3(target_x_meep, target_y_meep),
            material=mp.metal
        )
    ]

    round_trip_meep = (2 * TARGET_RANGE / C) * (C / length_scale)
    runtime_meep = round_trip_meep + 10

    print(f"Domain: {sx:.1f}×{sy:.1f} (MEEP units)")
    print(f"Resolution: {resolution_meep} pts/λ")
    print(f"Runtime: {runtime_meep:.1f}")
    print(f"Simulating {NUM_ELEMENTS} TX antennas...\n")

    # Run MEEP simulations
    s_matrix_meep = np.zeros((NUM_ELEMENTS, NUM_ELEMENTS))

    t_meep_start = time.time()

    for tx in range(NUM_ELEMENTS):
        print(f"  TX {tx+1}/{NUM_ELEMENTS}...", end='', flush=True)

        tx_pos = positions_meep[tx]
        sources = [
            mp.Source(
                mp.GaussianPulse(fcen, fwidth=df),
                component=mp.Ez,
                center=mp.Vector3(tx_pos[0], tx_pos[1]),
                size=mp.Vector3(0, 0)
            )
        ]

        cell = mp.Vector3(sx, sy)
        pml_layers = [mp.PML(1.0)]

        sim = mp.Simulation(
            cell_size=cell,
            geometry=geometry_meep,
            sources=sources,
            boundary_layers=pml_layers,
            resolution=resolution_meep
        )

        # Record at all RX
        rx_monitors = []
        for rx_pos in positions_meep:
            mon = sim.add_dft_fields(
                [mp.Ez], fcen, fcen, 1,
                center=mp.Vector3(rx_pos[0], rx_pos[1]),
                size=mp.Vector3(0, 0)
            )
            rx_monitors.append(mon)

        sim.run(until=runtime_meep)

        # Extract signals
        for rx in range(NUM_ELEMENTS):
            ez_data = sim.get_dft_array(rx_monitors[rx], mp.Ez, 0)
            s_matrix_meep[tx, rx] = np.abs(ez_data).sum()

        print(" ✓")

    t_meep = time.time() - t_meep_start

    # MEEP beamforming
    monostatic_meep = np.diag(s_matrix_meep)
    power_meep = []

    for angle in np.deg2rad(angles):
        phases = k * pos_array[:, 0] * np.sin(angle)
        a = np.exp(1j * phases) / np.sqrt(NUM_ELEMENTS)
        p = np.abs(np.dot(a.conj(), monostatic_meep))
        power_meep.append(p)

    power_meep = np.array(power_meep)
    power_meep_db = 10 * np.log10(power_meep / power_meep.max() + 1e-10)

    peak_meep_idx = np.argmax(power_meep)
    estimated_meep = angles[peak_meep_idx]
    error_meep = abs(estimated_meep - TARGET_ANGLE)

    left_peak_meep = power_meep_db[angles < 0].max()
    right_peak_meep = power_meep_db[angles > 0].max()
    ambiguity_meep = abs(left_peak_meep - right_peak_meep)

    print(f"\n✓ MEEP complete in {t_meep:.1f}s")
    print(f"  Estimated angle: {estimated_meep:.1f}°")
    print(f"  Error:           {error_meep:.1f}°")
    print(f"  Left peak:       {left_peak_meep:.1f} dB")
    print(f"  Right peak:      {right_peak_meep:.1f} dB")
    print(f"  Ambiguity gap:   {ambiguity_meep:.1f} dB")

    if ambiguity_meep < 3:
        print(f"  Status:          ⚠️  AMBIGUOUS (±θ)")
    else:
        print(f"  Status:          ✓ Unambiguous")
    print()

    meep_success = True

except Exception as e:
    print(f"❌ MEEP simulation failed: {e}\n")
    meep_success = False
    power_meep_db = None

# ============================================================
# Comparison and Visualization
# ============================================================

print("="*70)
print(" COMPARISON RESULTS")
print("="*70)
print()

if ceep_success and meep_success:
    print(f"{'Metric':<25} {'CEEP':<20} {'MEEP':<20}")
    print("-" * 70)
    print(f"{'Simulation time':<25} {t_ceep:<20.1f} {t_meep:<20.1f}")
    print(f"{'Speedup':<25} {'—':<20} {f'{t_meep/t_ceep:.1f}× slower':<20}")
    print(f"{'Estimated angle':<25} {estimated_ceep:<20.1f} {estimated_meep:<20.1f}")
    print(f"{'Error':<25} {error_ceep:<20.1f} {error_meep:<20.1f}")
    print(f"{'Left peak (dB)':<25} {left_peak_ceep:<20.1f} {left_peak_meep:<20.1f}")
    print(f"{'Right peak (dB)':<25} {right_peak_ceep:<20.1f} {right_peak_meep:<20.1f}")
    print(f"{'Ambiguity gap (dB)':<25} {ambiguity_ceep:<20.1f} {ambiguity_meep:<20.1f}")

    both_ambiguous = (ambiguity_ceep < 3) and (ambiguity_meep < 3)

    if both_ambiguous:
        print(f"{'Status':<25} {'AMBIGUOUS':<20} {'AMBIGUOUS':<20}")
        print()
        print("✓ VALIDATION SUCCESS!")
        print("  Both CEEP and MEEP show left-right ambiguity.")
        print("  This confirms it's a fundamental physics limitation,")
        print("  not a bug in CEEP implementation.")
    else:
        print(f"{'Status':<25} {'AMBIGUOUS' if ambiguity_ceep < 3 else 'OK':<20} {'AMBIGUOUS' if ambiguity_meep < 3 else 'OK':<20}")
        print()
        print("⚠️  Results differ between frameworks")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CEEP spectrum
    axes[0].plot(angles, power_ceep_db, 'b-', linewidth=2, label='CEEP')
    axes[0].axvline(TARGET_ANGLE, color='red', linestyle='--', linewidth=2,
                    label=f'Ground truth ({TARGET_ANGLE}°)')
    axes[0].axvline(estimated_ceep, color='green', linestyle=':', linewidth=2,
                    label=f'Peak ({estimated_ceep:.1f}°)')
    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('Power (dB)')
    axes[0].set_title(f'CEEP Beamforming (GPU, {t_ceep:.1f}s)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-90, 90)
    axes[0].set_ylim(-40, 5)

    # MEEP spectrum
    axes[1].plot(angles, power_meep_db, 'r-', linewidth=2, label='MEEP')
    axes[1].axvline(TARGET_ANGLE, color='red', linestyle='--', linewidth=2,
                    label=f'Ground truth ({TARGET_ANGLE}°)')
    axes[1].axvline(estimated_meep, color='green', linestyle=':', linewidth=2,
                    label=f'Peak ({estimated_meep:.1f}°)')
    axes[1].set_xlabel('Angle (degrees)')
    axes[1].set_ylabel('Power (dB)')
    axes[1].set_title(f'MEEP Beamforming (CPU, {t_meep:.1f}s)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-90, 90)
    axes[1].set_ylim(-40, 5)

    plt.tight_layout()
    plt.savefig('ceep_vs_meep_comparison.png', dpi=150, bbox_inches='tight')
    print()
    print("✓ Comparison plot saved to 'ceep_vs_meep_comparison.png'")

elif ceep_success:
    print("✓ CEEP completed successfully")
    print("❌ MEEP failed - cannot compare")
elif meep_success:
    print("❌ CEEP failed")
    print("✓ MEEP completed successfully")
else:
    print("❌ Both simulations failed")

print("="*70)
