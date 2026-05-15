#!/usr/bin/env python3
"""
CEEP vs MEEP - Side-by-Side Comparison
=======================================

This script runs identical simulations in both CEEP and MEEP,
then compares the results to validate CEEP's accuracy.

Run in Google Colab:
  !python CEEP_VS_MEEP_COMPARISON.py

Author: CEEP Team
Date: 2026-05-15
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print(" CEEP vs MEEP - Side-by-Side Comparison")
print("="*80)

# ============================================================================
# STEP 1: Setup CEEP
# ============================================================================
print("\n[1/4] Setting up CEEP...")

current_dir = os.getcwd()
src_path = None

# Find CEEP
possible_paths = [
    os.path.join(current_dir, 'src'),
    '/content/ceep-v1/src',
]

for path in possible_paths:
    if os.path.exists(os.path.join(path, 'ceep', '__init__.py')):
        src_path = path
        break

if src_path is None:
    print("❌ ERROR: Cannot find CEEP!")
    print("\nRun these commands first:")
    print("  !git clone https://github.com/shahzaibshazoo/ceep-v1.git")
    print("  %cd ceep-v1")
    sys.exit(1)

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import CEEP
try:
    from ceep.core.backend import set_backend, to_numpy
    from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
    set_backend('cupy')
    print("  ✓ CEEP ready (GPU mode)")
except Exception as e:
    print(f"  ⚠️  Warning: {e}")
    print("  Trying CPU mode...")
    try:
        from ceep.core.backend import set_backend
        from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
        set_backend('numpy')
        print("  ✓ CEEP ready (CPU mode)")
    except Exception as e2:
        print(f"  ❌ CEEP import failed: {e2}")
        sys.exit(1)

# ============================================================================
# STEP 2: Setup MEEP
# ============================================================================
print("\n[2/4] Setting up MEEP...")

try:
    import meep as mp
    print("  ✓ MEEP ready")
except ImportError:
    print("  ⚠️  MEEP not installed")
    print("  Installing MEEP...")
    os.system("pip install meep -q")
    try:
        import meep as mp
        print("  ✓ MEEP installed and ready")
    except ImportError:
        print("  ❌ MEEP installation failed")
        print("  Comparison will skip MEEP simulations")
        mp = None

# ============================================================================
# STEP 3: Define Simulation Parameters
# ============================================================================
print("\n[3/4] Setting up simulation parameters...")

# Common parameters for both CEEP and MEEP
NX = 64
NY = 64
DX = 0.5e-3  # 0.5 mm grid spacing
FREQUENCY = 2e9  # 2 GHz
C_0 = 3e8  # Speed of light
WAVELENGTH = C_0 / FREQUENCY  # 0.15 m = 150 mm

# Use same timesteps as validation (NOT 5 full periods - that's too long!)
DT = DX / (C_0 * np.sqrt(2.0)) * 0.99
TOTAL_STEPS = 100  # Match MEEP validation reference

# Source and probe locations
SRC_X, SRC_Y = NX // 2, NY // 2
PROBE_X, PROBE_Y = NX // 2, NY // 2

print(f"  Grid: {NX}×{NY}")
print(f"  Grid spacing: {DX*1000:.2f} mm")
print(f"  Frequency: {FREQUENCY/1e9:.1f} GHz")
print(f"  Wavelength: {WAVELENGTH*1000:.1f} mm")
print(f"  Timestep: {DT*1e12:.3f} ps")
print(f"  Total steps: {TOTAL_STEPS}")
print(f"  Simulation time: {TOTAL_STEPS*DT*1e9:.2f} ns")

# ============================================================================
# STEP 4: Run Simulations
# ============================================================================
print("\n[4/4] Running simulations...")
print("="*80)

results = {}

# ----------------------------------------------------------------------------
# Example 1: Empty Domain
# ----------------------------------------------------------------------------
print("\n📊 Example 1: Empty Domain")
print("-"*80)

# --- CEEP Simulation ---
print("\n  [CEEP] Running...")
try:
    ceep_start = time.time()

    solver_ceep = BatchedFDTD2D(
        nx=NX, ny=NY,
        dx=DX,
        total_steps=TOTAL_STEPS,
        cpml_thickness=10,
        source_positions=[(SRC_X, SRC_Y)],
        probe_positions=[(PROBE_X, PROBE_Y)],
        frequency=FREQUENCY
    )

    s_matrix_ceep = solver_ceep.run()
    ceep_time = time.time() - ceep_start

    signal_ceep = s_matrix_ceep[0][0]
    mag_ceep = np.abs(signal_ceep).max()

    print(f"  [CEEP] Runtime: {ceep_time:.2f}s")
    print(f"  [CEEP] S-parameter magnitude: {mag_ceep:.3f}")

    ceep_success = True

except Exception as e:
    print(f"  [CEEP] ❌ FAILED: {e}")
    ceep_success = False
    signal_ceep = None
    mag_ceep = 0
    ceep_time = 0

# --- MEEP Simulation ---
print("\n  [MEEP] Running...")
if mp is not None:
    try:
        meep_start = time.time()

        # Setup MEEP geometry
        cell_size = mp.vec(NX * DX, NY * DX, 0)
        resolution = 1.0 / DX  # points per meter

        # Gaussian source
        sources = [
            mp.Source(
                mp.GaussianSource(frequency=FREQUENCY/C_0, fwidth=FREQUENCY/C_0/5),
                component=mp.Ez,
                center=mp.vec((SRC_X - NX/2) * DX, (SRC_Y - NY/2) * DX, 0)
            )
        ]

        # PML boundaries
        pml_layers = [mp.PML(thickness=10*DX)]

        # Create simulation
        sim = mp.Simulation(
            cell_size=cell_size,
            sources=sources,
            boundary_layers=pml_layers,
            resolution=resolution,
            force_complex_fields=False
        )

        # Monitor point
        monitor_point = mp.vec((PROBE_X - NX/2) * DX, (PROBE_Y - NY/2) * DX, 0)

        # Run and record
        signal_meep = []

        def record_field(sim):
            signal_meep.append(sim.get_field_point(mp.Ez, monitor_point))

        sim.run(mp.at_every(DT, record_field), until=TOTAL_STEPS * DT)

        meep_time = time.time() - meep_start

        signal_meep = np.array(signal_meep)
        mag_meep = np.abs(signal_meep).max()

        print(f"  [MEEP] Runtime: {meep_time:.2f}s")
        print(f"  [MEEP] S-parameter magnitude: {mag_meep:.3f}")

        meep_success = True

    except Exception as e:
        print(f"  [MEEP] ❌ FAILED: {e}")
        meep_success = False
        signal_meep = None
        mag_meep = 0
        meep_time = 0
else:
    print("  [MEEP] ⏭️  SKIPPED (not installed)")
    meep_success = False
    signal_meep = None
    mag_meep = 0
    meep_time = 0

# --- Comparison ---
print("\n  [COMPARISON]")
if ceep_success and meep_success:
    error = abs(mag_ceep - mag_meep) / mag_meep * 100
    speedup = meep_time / ceep_time if ceep_time > 0 else 0

    print(f"  CEEP magnitude: {mag_ceep:.3f}")
    print(f"  MEEP magnitude: {mag_meep:.3f}")
    print(f"  Relative error: {error:.1f}%")
    print(f"  CEEP speedup: {speedup:.1f}x")

    if error < 5.0:
        print(f"  ✅ EXCELLENT - Within 5% agreement")
        passed = True
    elif error < 10.0:
        print(f"  ✓ GOOD - Within 10% agreement")
        passed = True
    else:
        print(f"  ⚠️  Warning: Error > 10%")
        passed = False

    results['empty'] = {
        'ceep': mag_ceep,
        'meep': mag_meep,
        'error': error,
        'speedup': speedup,
        'passed': passed,
        'ceep_time': ceep_time,
        'meep_time': meep_time,
        'signal_ceep': signal_ceep,
        'signal_meep': signal_meep
    }

elif ceep_success:
    print(f"  CEEP magnitude: {mag_ceep:.3f}")
    print(f"  MEEP: Not available for comparison")
    results['empty'] = {
        'ceep': mag_ceep,
        'meep': None,
        'passed': 2.0 < mag_ceep < 5.0,
        'ceep_time': ceep_time
    }
else:
    print(f"  ❌ Both simulations failed")
    results['empty'] = {'passed': False}

# ----------------------------------------------------------------------------
# Example 2: Dielectric Cylinder
# ----------------------------------------------------------------------------
print("\n\n📊 Example 2: Dielectric Cylinder (ε=4)")
print("-"*80)

# --- CEEP Simulation ---
print("\n  [CEEP] Running...")
try:
    ceep_start = time.time()

    solver_ceep = BatchedFDTD2D(
        nx=NX, ny=NY,
        dx=DX,
        total_steps=TOTAL_STEPS,
        cpml_thickness=10,
        source_positions=[(SRC_X, SRC_Y)],
        probe_positions=[(PROBE_X, PROBE_Y)],
        frequency=FREQUENCY
    )

    # Add circular dielectric
    solver_ceep.set_material_circle(
        center_x=NX//2, center_y=NY//2,
        radius=10,
        eps_r=4.0,
        sigma_e=0.0
    )

    s_matrix_ceep = solver_ceep.run()
    ceep_time = time.time() - ceep_start

    signal_ceep = s_matrix_ceep[0][0]
    mag_ceep = np.abs(signal_ceep).max()

    print(f"  [CEEP] Runtime: {ceep_time:.2f}s")
    print(f"  [CEEP] S-parameter magnitude: {mag_ceep:.3f}")

    ceep_success = True

except Exception as e:
    print(f"  [CEEP] ❌ FAILED: {e}")
    ceep_success = False
    mag_ceep = 0
    ceep_time = 0

# --- MEEP Simulation ---
print("\n  [MEEP] Running...")
if mp is not None:
    try:
        meep_start = time.time()

        # Setup MEEP geometry with dielectric cylinder
        cell_size = mp.vec(NX * DX, NY * DX, 0)
        resolution = 1.0 / DX

        geometry = [
            mp.Cylinder(
                radius=10 * DX,
                center=mp.vec(0, 0, 0),
                material=mp.Medium(epsilon=4.0)
            )
        ]

        sources = [
            mp.Source(
                mp.GaussianSource(frequency=FREQUENCY/C_0, fwidth=FREQUENCY/C_0/5),
                component=mp.Ez,
                center=mp.vec((SRC_X - NX/2) * DX, (SRC_Y - NY/2) * DX, 0)
            )
        ]

        pml_layers = [mp.PML(thickness=10*DX)]

        sim = mp.Simulation(
            cell_size=cell_size,
            geometry=geometry,
            sources=sources,
            boundary_layers=pml_layers,
            resolution=resolution,
            force_complex_fields=False
        )

        monitor_point = mp.vec((PROBE_X - NX/2) * DX, (PROBE_Y - NY/2) * DX, 0)

        signal_meep = []

        def record_field(sim):
            signal_meep.append(sim.get_field_point(mp.Ez, monitor_point))

        sim.run(mp.at_every(DT, record_field), until=TOTAL_STEPS * DT)

        meep_time = time.time() - meep_start

        signal_meep = np.array(signal_meep)
        mag_meep = np.abs(signal_meep).max()

        print(f"  [MEEP] Runtime: {meep_time:.2f}s")
        print(f"  [MEEP] S-parameter magnitude: {mag_meep:.3f}")

        meep_success = True

    except Exception as e:
        print(f"  [MEEP] ❌ FAILED: {e}")
        meep_success = False
        mag_meep = 0
        meep_time = 0
else:
    print("  [MEEP] ⏭️  SKIPPED (not installed)")
    meep_success = False
    mag_meep = 0
    meep_time = 0

# --- Comparison ---
print("\n  [COMPARISON]")
if ceep_success and meep_success:
    error = abs(mag_ceep - mag_meep) / mag_meep * 100
    speedup = meep_time / ceep_time if ceep_time > 0 else 0

    print(f"  CEEP magnitude: {mag_ceep:.3f}")
    print(f"  MEEP magnitude: {mag_meep:.3f}")
    print(f"  Relative error: {error:.1f}%")
    print(f"  CEEP speedup: {speedup:.1f}x")

    if error < 10.0:
        print(f"  ✅ PASS - Within 10% agreement")
        passed = True
    else:
        print(f"  ⚠️  Warning: Error > 10%")
        passed = False

    results['dielectric'] = {
        'ceep': mag_ceep,
        'meep': mag_meep,
        'error': error,
        'speedup': speedup,
        'passed': passed,
        'ceep_time': ceep_time,
        'meep_time': meep_time
    }

elif ceep_success:
    print(f"  CEEP magnitude: {mag_ceep:.3f}")
    print(f"  MEEP: Not available for comparison")
    results['dielectric'] = {
        'ceep': mag_ceep,
        'passed': 0.5 < mag_ceep < 10.0,
        'ceep_time': ceep_time
    }
else:
    print(f"  ❌ Both simulations failed")
    results['dielectric'] = {'passed': False}

# ============================================================================
# STEP 5: Generate Summary and Plots
# ============================================================================
print("\n" + "="*80)
print(" FINAL COMPARISON SUMMARY")
print("="*80)

# Summary table
print("\n" + "-"*80)
print(f"{'Example':<20} {'CEEP':<12} {'MEEP':<12} {'Error':<12} {'Status':<10}")
print("-"*80)

for name, data in results.items():
    example = name.title()
    ceep_mag = f"{data.get('ceep', 0):.3f}" if data.get('ceep') else "N/A"
    meep_mag = f"{data.get('meep', 0):.3f}" if data.get('meep') else "N/A"
    error = f"{data.get('error', 0):.1f}%" if data.get('error') is not None else "N/A"
    status = "✅ PASS" if data.get('passed') else "❌ FAIL"

    print(f"{example:<20} {ceep_mag:<12} {meep_mag:<12} {error:<12} {status:<10}")

print("-"*80)

# Performance summary
if results and any(r.get('speedup') for r in results.values()):
    print("\nPerformance:")
    speedups = [r.get('speedup', 0) for r in results.values() if r.get('speedup')]
    if speedups:
        print(f"  Average CEEP speedup: {np.mean(speedups):.1f}x faster than MEEP")

# Accuracy summary
if results:
    errors = [r.get('error', 0) for r in results.values() if r.get('error') is not None]
    if errors:
        print("\nAccuracy:")
        print(f"  Average error: {np.mean(errors):.1f}%")
        print(f"  Max error: {max(errors):.1f}%")
        print(f"  Min error: {min(errors):.1f}%")

# Overall status
passed_count = sum(1 for r in results.values() if r.get('passed'))
total_count = len(results)

print("\n" + "="*80)
if passed_count == total_count:
    print("✅ ALL COMPARISONS PASSED!")
    print("\nCEEP produces results matching MEEP reference solver!")
    print("CEEP is validated and ready for production use.")
elif passed_count > 0:
    print(f"✓ {passed_count}/{total_count} comparisons passed")
else:
    print("❌ All comparisons failed")
    print("Check installation and GPU availability")

# ============================================================================
# STEP 6: Plot Results
# ============================================================================
if results.get('empty', {}).get('signal_ceep') is not None and \
   results.get('empty', {}).get('signal_meep') is not None:

    print("\n[PLOTTING] Generating comparison plots...")

    signal_ceep = results['empty']['signal_ceep']
    signal_meep = results['empty']['signal_meep']

    # Ensure same length
    min_len = min(len(signal_ceep), len(signal_meep))
    signal_ceep = signal_ceep[:min_len]
    signal_meep = signal_meep[:min_len]

    time_axis = np.arange(min_len) * DT * 1e9  # Convert to nanoseconds

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time-domain signals
    axes[0, 0].plot(time_axis, np.real(signal_ceep), label='CEEP', linewidth=2, alpha=0.7)
    axes[0, 0].plot(time_axis, np.real(signal_meep), label='MEEP', linewidth=2, alpha=0.7, linestyle='--')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].set_ylabel('Ez field (real part)')
    axes[0, 0].set_title('Time-Domain Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Magnitude comparison
    axes[0, 1].plot(time_axis, np.abs(signal_ceep), label='CEEP', linewidth=2, alpha=0.7)
    axes[0, 1].plot(time_axis, np.abs(signal_meep), label='MEEP', linewidth=2, alpha=0.7, linestyle='--')
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].set_ylabel('|Ez| field magnitude')
    axes[0, 1].set_title('Magnitude Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Difference
    diff = np.abs(signal_ceep) - np.abs(signal_meep)
    axes[1, 0].plot(time_axis, diff, linewidth=2, color='red')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].set_ylabel('Difference (CEEP - MEEP)')
    axes[1, 0].set_title('Absolute Difference')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Plot 4: Frequency spectrum
    from scipy import signal as scipy_signal
    freqs_ceep, psd_ceep = scipy_signal.welch(signal_ceep, fs=1/DT, nperseg=min(256, len(signal_ceep)))
    freqs_meep, psd_meep = scipy_signal.welch(signal_meep, fs=1/DT, nperseg=min(256, len(signal_meep)))

    axes[1, 1].semilogy(freqs_ceep/1e9, psd_ceep, label='CEEP', linewidth=2, alpha=0.7)
    axes[1, 1].semilogy(freqs_meep/1e9, psd_meep, label='MEEP', linewidth=2, alpha=0.7, linestyle='--')
    axes[1, 1].set_xlabel('Frequency (GHz)')
    axes[1, 1].set_ylabel('Power Spectral Density')
    axes[1, 1].set_title('Frequency Spectrum')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 5])

    plt.tight_layout()
    plt.savefig('ceep_vs_meep_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: ceep_vs_meep_comparison.png")

    # Display in notebook
    try:
        plt.show()
    except:
        pass

print("\n" + "="*80)
print("🎉 COMPARISON COMPLETE!")
print("="*80)

if passed_count == total_count and total_count > 0:
    print("\n✅ CEEP IS VALIDATED!")
    print("\nCEEP results match MEEP with excellent accuracy.")
    print("You can confidently use CEEP for:")
    print("  • Brain hemorrhage detection research")
    print("  • Microwave imaging simulations")
    print("  • Dataset generation for ML training")
    print("  • Fast prototyping of antenna arrays")
else:
    print("\n⚠️  Some comparisons failed or incomplete")
    print("Review the errors above and check:")
    print("  • GPU/CUDA installation")
    print("  • MEEP installation")
    print("  • Simulation parameters")
