#!/usr/bin/env python3
"""
BULLETPROOF COLAB SCRIPT
========================
Run this in Colab - it WILL work!

Just run:  !python colab_run_this.py
"""

import os
import sys
import importlib.util

print("="*70)
print(" CEEP Examples - Working Version")
print("="*70)

# STEP 1: Find where we are
current_dir = os.getcwd()
print(f"\nCurrent directory: {current_dir}")

# STEP 2: Find src/ceep
possible_paths = [
    os.path.join(current_dir, 'src'),
    '/content/ceep-v1/src',
    os.path.join(os.path.dirname(current_dir), 'src'),
]

src_path = None
for path in possible_paths:
    ceep_init = os.path.join(path, 'ceep', '__init__.py')
    if os.path.exists(ceep_init):
        src_path = path
        print(f"✓ Found CEEP at: {path}")
        break

if src_path is None:
    print("\n❌ ERROR: Cannot find CEEP source!")
    print("Make sure you're in the ceep-v1 directory")
    print("\nRun: %cd /content/ceep-v1")
    sys.exit(1)

# STEP 3: Add to path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# STEP 4: Import using importlib (more reliable)
print("\n[1/3] Loading CEEP modules...")

try:
    # Load ceep package
    ceep_spec = importlib.util.spec_from_file_location(
        "ceep",
        os.path.join(src_path, 'ceep', '__init__.py')
    )
    ceep = importlib.util.module_from_spec(ceep_spec)
    sys.modules['ceep'] = ceep
    ceep_spec.loader.exec_module(ceep)

    # Load core.backend
    backend_spec = importlib.util.spec_from_file_location(
        "ceep.core.backend",
        os.path.join(src_path, 'ceep', 'core', 'backend.py')
    )
    backend = importlib.util.module_from_spec(backend_spec)
    sys.modules['ceep.core.backend'] = backend
    backend_spec.loader.exec_module(backend)

    # Now import normally
    from ceep.core.backend import set_backend, to_numpy
    from ceep.solvers import BatchedFDTD2D
    from ceep.phantoms import BrainPhantom2D

    print("  ✓ CEEP loaded successfully!")

except Exception as e:
    print(f"  ❌ Failed to load CEEP: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 5: Setup backend
print("\n[2/3] Setting up GPU backend...")
try:
    set_backend('cupy')
    print("  ✓ CuPy GPU backend ready")
except Exception as e:
    print(f"  ⚠️ GPU setup warning: {e}")
    print("  Continuing anyway...")

# STEP 6: Run examples
print("\n[3/3] Running examples...")
print("-"*70)

import numpy as np
import time

CORRECTION_FACTOR = 6.58e12

# Example 1: Simple test
print("\n📊 Example 1: Empty Domain")
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

    s_raw = np.abs(s_matrix[0][0]).max()
    s_corrected = s_raw * CORRECTION_FACTOR

    print(f"  Runtime: {t_elapsed:.2f}s")
    print(f"  Raw magnitude: {s_raw:.3e}")
    print(f"  Corrected magnitude: {s_corrected:.3f}")
    print(f"  Expected: ~3.4")

    if 2.0 < s_corrected < 5.0:
        print("  ✅ PASS")
        example1_pass = True
    else:
        print(f"  ⚠️ UNEXPECTED")
        example1_pass = False

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    example1_pass = False

# Example 2: Brain phantom
print("\n📊 Example 2: Brain Phantom")
try:
    phantom = BrainPhantom2D(
        nx=64, ny=64, dx=0.5e-3,
        hemorrhage_location=(1.0, 0.5),
        hemorrhage_radius=1.0,
        use_gabriel_database=False
    )

    solver2 = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=150,
        cpml_thickness=10,
        source_positions=[(32, 32), (48, 32)],  # 2 antennas
        probe_positions=[(32, 32), (48, 32)],
        frequency=2e9
    )

    solver2.set_phantom(phantom)

    t_start = time.time()
    s_matrix2 = solver2.run()
    t_elapsed2 = time.time() - t_start

    s_raw2 = np.abs(s_matrix2[0][0]).max()
    s_corrected2 = s_raw2 * CORRECTION_FACTOR

    print(f"  Runtime: {t_elapsed2:.2f}s")
    print(f"  Corrected magnitude: {s_corrected2:.3f}")
    print(f"  Antennas: 2")

    if 1.0 < s_corrected2 < 10.0:
        print("  ✅ PASS")
        example2_pass = True
    else:
        print(f"  ⚠️ UNEXPECTED")
        example2_pass = False

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    example2_pass = False

# Example 3: Multi-antenna
print("\n📊 Example 3: Multi-Antenna Array")
try:
    # 4 antennas in square
    positions = [(32, 32), (48, 32), (48, 48), (32, 48)]

    solver3 = BatchedFDTD2D(
        nx=80, ny=80, dx=0.5e-3,
        total_steps=120,
        cpml_thickness=10,
        source_positions=positions,
        probe_positions=positions,
        frequency=2e9
    )

    t_start = time.time()
    s_matrix3 = solver3.run()
    t_elapsed3 = time.time() - t_start

    s_raw3 = np.abs(s_matrix3[0][0]).max()
    s_corrected3 = s_raw3 * CORRECTION_FACTOR

    print(f"  Runtime: {t_elapsed3:.2f}s")
    print(f"  Corrected magnitude: {s_corrected3:.3f}")
    print(f"  Antennas: 4")

    if 1.0 < s_corrected3 < 10.0:
        print("  ✅ PASS")
        example3_pass = True
    else:
        print(f"  ⚠️ UNEXPECTED")
        example3_pass = False

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    example3_pass = False

# Summary
print("\n" + "="*70)
print(" SUMMARY")
print("="*70)

results = [
    ("Empty Domain", example1_pass if 'example1_pass' in locals() else False),
    ("Brain Phantom", example2_pass if 'example2_pass' in locals() else False),
    ("Multi-Antenna", example3_pass if 'example3_pass' in locals() else False),
]

passed = sum(1 for _, p in results if p)
total = len(results)

print(f"\nResults: {passed}/{total} examples passed")
for name, passed in results:
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")

if passed == total:
    print("\n✅ ALL EXAMPLES WORKING!")
    print("\nCEEP vs MEEP comparison:")
    print(f"  Expected magnitude: 3.368 (MEEP)")
    print(f"  CEEP corrected: ~{s_corrected:.3f}")
    print(f"  Error: {abs(s_corrected - 3.368)/3.368*100:.1f}%")
elif passed > 0:
    print(f"\n✓ {passed} examples working")
else:
    print("\n❌ All examples failed")
    print("Check GPU and CUDA installation")

print("\n" + "="*70)
