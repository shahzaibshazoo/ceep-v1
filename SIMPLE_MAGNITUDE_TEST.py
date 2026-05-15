#!/usr/bin/env python3
"""
Simple Magnitude Test - Check if CEEP produces correct S-parameters
====================================================================

This tests ONLY the magnitude, without requiring MEEP.

Expected result: S-parameter magnitude ≈ 3.368

Run in Colab:
  !python SIMPLE_MAGNITUDE_TEST.py
"""

import os
import sys
import numpy as np

print("="*80)
print(" CEEP - Simple Magnitude Test")
print("="*80)

# Setup path
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
if not os.path.exists(src_path):
    src_path = '/content/ceep-v1/src'

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import CEEP
print("\n[1/2] Importing CEEP...")
try:
    from ceep.core.backend import set_backend
    from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
    set_backend('cupy')
    print("  ✓ CEEP imported (GPU mode)")
except Exception as e:
    print(f"  ⚠️  GPU failed: {e}")
    try:
        from ceep.core.backend import set_backend
        from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
        set_backend('numpy')
        print("  ✓ CEEP imported (CPU mode)")
    except Exception as e2:
        print(f"  ❌ Import failed: {e2}")
        sys.exit(1)

# Run test
print("\n[2/2] Running test simulation...")
print("-"*80)

try:
    # Exact parameters from MEEP validation
    solver = BatchedFDTD2D(
        nx=64, ny=64,
        dx=0.5e-3,
        total_steps=100,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )

    print(f"  Grid: 64×64")
    print(f"  Frequency: 2 GHz")
    print(f"  Timesteps: 100")
    print(f"  Running simulation...")

    s_matrix = solver.run()
    signal = s_matrix[0][0]
    magnitude = np.abs(signal).max()

    print(f"\n  ✓ Simulation complete!")
    print(f"\n  Results:")
    print(f"    S-parameter magnitude: {magnitude:.3f}")
    print(f"    Expected (MEEP):       3.368")

    error = abs(magnitude - 3.368) / 3.368 * 100

    print(f"    Relative error:        {error:.1f}%")

    print(f"\n  Validation:")
    if error < 1.0:
        print(f"    🎯 EXCELLENT - Within 1% of MEEP!")
        print(f"    ✅ CEEP is working correctly!")
        status = "PASS"
    elif error < 5.0:
        print(f"    ✅ VERY GOOD - Within 5% of MEEP!")
        print(f"    ✓ CEEP is validated!")
        status = "PASS"
    elif error < 10.0:
        print(f"    ✓ GOOD - Within 10% of MEEP")
        print(f"    Acceptable for most applications")
        status = "PASS"
    else:
        print(f"    ❌ ERROR TOO LARGE")
        print(f"    Expected magnitude ~3.4, got {magnitude:.3f}")
        status = "FAIL"

    # Additional diagnostics
    print(f"\n  Signal Statistics:")
    print(f"    Peak magnitude:  {np.abs(signal).max():.3f}")
    print(f"    Mean magnitude:  {np.abs(signal).mean():.3f}")
    print(f"    RMS value:       {np.sqrt(np.mean(np.abs(signal)**2)):.3f}")
    print(f"    Signal length:   {len(signal)} samples")

    # Check for common issues
    print(f"\n  Diagnostics:")
    if np.any(np.isnan(signal)):
        print(f"    ⚠️  WARNING: Signal contains NaN values")
    elif np.any(np.isinf(signal)):
        print(f"    ⚠️  WARNING: Signal contains Inf values")
    elif magnitude < 0.1:
        print(f"    ⚠️  WARNING: Magnitude too small (< 0.1)")
    elif magnitude > 100:
        print(f"    ⚠️  WARNING: Magnitude too large (> 100)")
    else:
        print(f"    ✓ Signal appears healthy")

except Exception as e:
    print(f"\n  ❌ SIMULATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    status = "FAIL"

print("\n" + "="*80)
if status == "PASS":
    print(" ✅ TEST PASSED - CEEP IS WORKING CORRECTLY!")
else:
    print(" ❌ TEST FAILED - CHECK INSTALLATION")
print("="*80)

# Detailed diagnostic if failed
if status == "FAIL":
    print("\nDiagnostic Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  CEEP source path: {src_path}")

    # Check git version
    import subprocess
    try:
        result = subprocess.run(['git', 'log', '-1', '--oneline'],
                              capture_output=True, text=True, cwd=os.getcwd())
        print(f"  Git commit: {result.stdout.strip()}")
    except:
        print(f"  Git commit: Unable to determine")

    print("\nTroubleshooting:")
    print("  1. Pull latest changes: !git pull origin master")
    print("  2. Check GPU: !nvidia-smi")
    print("  3. Reinstall CuPy: !pip install cupy-cuda12x -U")
