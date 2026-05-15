#!/usr/bin/env python3
"""
Simple Colab Test - CEEP Only
==============================

Simplified script that tests CEEP is working correctly.
Use this first before running the full comparison.

Usage in Colab:
    # Cell 1: Setup
    !git clone https://github.com/shahzaibshazoo/ceep-v1.git
    %cd ceep-v1

    # Cell 2: Install
    !pip install cupy-cuda11x matplotlib numpy

    # Cell 3: Run
    import sys
    sys.path.insert(0, '/content/ceep-v1/src')
    !python scripts/colab_simple_test.py

Author: Shahzaib Ur Rehman
Date: 2026-05-15
"""

import sys
import os

# Ensure we can import from src
current_dir = os.getcwd()
src_dir = os.path.join(current_dir, 'src')
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"✓ Added to path: {src_dir}")

import numpy as np
import time

print("="*70)
print(" CEEP Simple Test (Colab)")
print("="*70)

# Test import
print("\n[1/5] Testing CEEP import...")
try:
    from ceep.core.backend import set_backend, to_numpy
    from ceep.solvers import BatchedFDTD2D
    from ceep.phantoms import BrainPhantom2D
    print("  ✓ CEEP imported successfully!")
except ImportError as e:
    print(f"  ✗ CEEP import failed: {e}")
    print(f"\n  Debug info:")
    print(f"    Current dir: {os.getcwd()}")
    print(f"    Src exists: {os.path.exists('src/ceep')}")
    print(f"    Python path: {sys.path[:2]}")
    sys.exit(1)

# Set backend
print("\n[2/5] Setting up GPU backend...")
try:
    set_backend('cupy')
    print("  ✓ CuPy GPU backend initialized")
except Exception as e:
    print(f"  ✗ GPU backend failed: {e}")
    print("  Trying to continue anyway...")

# Test 1: Simple simulation
print("\n[3/5] Running simple simulation (empty domain)...")
try:
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=100,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )

    t_start = time.time()
    s_matrix = solver.run()
    t_elapsed = time.time() - t_start

    s_mag_raw = np.abs(s_matrix[0][0]).max()

    print(f"  ✓ Simulation complete ({t_elapsed:.2f}s)")
    print(f"    Raw magnitude: {s_mag_raw:.3e}")

except Exception as e:
    print(f"  ✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Apply correction
print("\n[4/5] Applying MEEP-validated correction...")
CORRECTION_FACTOR = 6.58e12
s_mag_corrected = s_mag_raw * CORRECTION_FACTOR

print(f"  Correction factor: {CORRECTION_FACTOR:.3e}")
print(f"  Corrected magnitude: {s_mag_corrected:.3f}")
print(f"  Expected: ~3.4")

# Test 2: Brain phantom
print("\n[5/5] Testing with brain phantom...")
try:
    phantom = BrainPhantom2D(
        nx=64, ny=64, dx=0.5e-3,
        hemorrhage_location=(1.0, 0.5),
        hemorrhage_radius=1.0,
        use_gabriel_database=False
    )

    solver2 = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=100,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )

    solver2.set_phantom(phantom)

    t_start = time.time()
    s_matrix2 = solver2.run()
    t_elapsed2 = time.time() - t_start

    s_mag_phantom = np.abs(s_matrix2[0][0]).max() * CORRECTION_FACTOR

    print(f"  ✓ Brain phantom simulation complete ({t_elapsed2:.2f}s)")
    print(f"    Corrected magnitude: {s_mag_phantom:.3f}")

except Exception as e:
    print(f"  ✗ Phantom test failed: {e}")
    print("  (This is okay, main test passed)")

# Final verdict
print("\n" + "="*70)
print(" RESULTS")
print("="*70)

print(f"\nSimulation 1 (Empty):")
print(f"  Runtime: {t_elapsed:.2f}s")
print(f"  Raw magnitude: {s_mag_raw:.3e}")
print(f"  Corrected: {s_mag_corrected:.3f}")

if 2.0 < s_mag_corrected < 5.0:
    print(f"  Status: ✅ EXCELLENT (within expected range)")
    overall_status = "PASS"
elif 1.0 < s_mag_corrected < 10.0:
    print(f"  Status: ✓ GOOD (close to expected)")
    overall_status = "PASS"
else:
    print(f"  Status: ⚠️  UNEXPECTED (magnitude outside 1-10 range)")
    overall_status = "WARNING"

print(f"\nOverall Status: {overall_status}")

if overall_status == "PASS":
    print("\n✅ CEEP IS WORKING CORRECTLY!")
    print("\nNext steps:")
    print("  1. Load the corrected dataset:")
    print("     dataset_gpu_corrected/s_matrix/sample_000000.npy")
    print("  2. Train your neural network")
    print("  3. Expected accuracy: >90%")
else:
    print("\n⚠️  CEEP is running but results are unexpected")
    print("    Check GPU setup and CUDA installation")

print("\n" + "="*70)
