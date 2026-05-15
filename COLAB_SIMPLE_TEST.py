#!/usr/bin/env python3
"""
CEEP Library - Simple Test (No Phantom Import)
===============================================

Simpler version that tests core functionality without BrainPhantom2D.
Use this if you're having import issues with the phantom module.

Run in Google Colab:
  !python COLAB_SIMPLE_TEST.py

Author: CEEP Team
Date: 2026-05-15
"""

import os
import sys
import time
import numpy as np

print("="*80)
print(" CEEP Library - Simple Test Suite")
print("="*80)

# ============================================================================
# STEP 1: Setup Environment
# ============================================================================
print("\n[1/3] Setting up environment...")

current_dir = os.getcwd()
print(f"  Current directory: {current_dir}")

# Find CEEP
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
        break

if src_path is None:
    print("\n❌ ERROR: Cannot find CEEP!")
    print("\nRun these commands first:")
    print("  !git clone https://github.com/shahzaibshazoo/ceep-v1.git")
    print("  %cd ceep-v1")
    sys.exit(1)

if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"  ✓ Found CEEP at: {src_path}")

# Import CEEP (without phantoms for now)
try:
    from ceep.core.backend import set_backend, to_numpy
    from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
    print("  ✓ CEEP core imported successfully")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Setup GPU backend
print("\n[2/3] Configuring GPU backend...")
try:
    set_backend('cupy')
    print("  ✓ CuPy GPU backend ready")
except Exception as e:
    print(f"  ⚠️  Warning: {e}")

# ============================================================================
# STEP 2: Run Tests
# ============================================================================
print("\n[3/3] Running tests...")
print("="*80)

MEEP_REFERENCE = 3.368
TOLERANCE = 5.0

results = []

# ----------------------------------------------------------------------------
# Test 1: Empty Domain (MEEP Validation)
# ----------------------------------------------------------------------------
print("\n📊 Test 1: Empty Domain (MEEP Validation)")
print("-"*80)

try:
    solver = BatchedFDTD2D(
        nx=64, ny=64,
        dx=0.5e-3,
        total_steps=100,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )

    t_start = time.time()
    s_matrix = solver.run()
    t_elapsed = time.time() - t_start

    s_mag = np.abs(s_matrix[0][0]).max()
    error = abs(s_mag - MEEP_REFERENCE) / MEEP_REFERENCE * 100

    print(f"  Config: 64×64 grid, 100 timesteps")
    print(f"  Runtime: {t_elapsed:.2f}s")
    print(f"  S-parameter: {s_mag:.3f}")
    print(f"  MEEP reference: {MEEP_REFERENCE:.3f}")
    print(f"  Error: {error:.1f}%")

    passed = error < TOLERANCE
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Status: {status}")

    results.append({
        'name': 'Empty Domain',
        'runtime': t_elapsed,
        'magnitude': s_mag,
        'error': error,
        'passed': passed
    })

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    results.append({'name': 'Empty Domain', 'passed': False})

# ----------------------------------------------------------------------------
# Test 2: 2-Antenna Array
# ----------------------------------------------------------------------------
print("\n📊 Test 2: 2-Antenna Array")
print("-"*80)

try:
    positions = [(25, 40), (55, 40)]

    solver = BatchedFDTD2D(
        nx=80, ny=80,
        dx=0.5e-3,
        total_steps=120,
        cpml_thickness=10,
        source_positions=positions,
        probe_positions=positions,
        frequency=2e9
    )

    t_start = time.time()
    s_matrix = solver.run()
    t_elapsed = time.time() - t_start

    s_values = [np.abs(s_matrix[tx][rx]).max() for tx in range(2) for rx in range(2)]
    s_max = max(s_values)

    print(f"  Config: 80×80 grid, 2 antennas")
    print(f"  Runtime: {t_elapsed:.2f}s")
    print(f"  Max S-parameter: {s_max:.3f}")

    passed = 2.0 < s_max < 5.0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Status: {status}")

    results.append({
        'name': '2-Antenna',
        'runtime': t_elapsed,
        'magnitude': s_max,
        'passed': passed
    })

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    results.append({'name': '2-Antenna', 'passed': False})

# ----------------------------------------------------------------------------
# Test 3: Brain Tissue (Manual Phantom)
# ----------------------------------------------------------------------------
print("\n📊 Test 3: Brain Tissue (Manual Phantom)")
print("-"*80)

try:
    # Create brain phantom manually
    nx, ny = 64, 64
    cx, cy = nx // 2, ny // 2

    eps_r = np.ones((nx, ny), dtype=np.float64)
    sigma_e = np.zeros((nx, ny), dtype=np.float64)

    # Circular brain region
    x, y = np.ogrid[:nx, :ny]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    brain_mask = r < 20
    eps_r[brain_mask] = 50.0  # Brain tissue at 2 GHz
    sigma_e[brain_mask] = 1.5

    # Add hemorrhage
    hem_x, hem_y = cx + 8, cy + 5
    hem_mask = np.sqrt((x - hem_x)**2 + (y - hem_y)**2) < 5
    eps_r[hem_mask] = 60.0  # Blood
    sigma_e[hem_mask] = 2.0

    positions = [(32, 32)]

    solver = BatchedFDTD2D(
        nx=nx, ny=ny,
        dx=0.5e-3,
        total_steps=150,
        cpml_thickness=10,
        source_positions=positions,
        probe_positions=positions,
        frequency=2e9
    )

    # Set materials directly
    solver._eps_r[:] = eps_r
    solver._sigma_e[:] = sigma_e

    t_start = time.time()
    s_matrix = solver.run()
    t_elapsed = time.time() - t_start

    s_mag = np.abs(s_matrix[0][0]).max()

    print(f"  Config: 64×64 grid, brain phantom")
    print(f"  Runtime: {t_elapsed:.2f}s")
    print(f"  S-parameter: {s_mag:.3f}")

    passed = 0.5 < s_mag < 10.0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Status: {status}")

    results.append({
        'name': 'Brain Tissue',
        'runtime': t_elapsed,
        'magnitude': s_mag,
        'passed': passed
    })

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    results.append({'name': 'Brain Tissue', 'passed': False})

# ----------------------------------------------------------------------------
# Test 4: 8-Antenna Circular Array
# ----------------------------------------------------------------------------
print("\n📊 Test 4: 8-Antenna Circular Array")
print("-"*80)

try:
    nx, ny = 100, 100
    cx, cy = nx // 2, ny // 2
    radius = 30

    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    positions = [
        (int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a)))
        for a in angles
    ]

    solver = BatchedFDTD2D(
        nx=nx, ny=ny,
        dx=0.5e-3,
        total_steps=150,
        cpml_thickness=10,
        source_positions=positions,
        probe_positions=positions,
        frequency=2e9
    )

    t_start = time.time()
    s_matrix = solver.run()
    t_elapsed = time.time() - t_start

    s_values = [np.abs(s_matrix[tx][rx]).max() for tx in range(8) for rx in range(8)]
    s_max = max(s_values)
    s_mean = np.mean(s_values)

    print(f"  Config: 100×100 grid, 8 antennas")
    print(f"  Runtime: {t_elapsed:.2f}s")
    print(f"  Max S-parameter: {s_max:.3f}")
    print(f"  Mean S-parameter: {s_mean:.3f}")

    passed = 2.0 < s_max < 5.0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Status: {status}")

    results.append({
        'name': '8-Antenna',
        'runtime': t_elapsed,
        'magnitude': s_max,
        'passed': passed
    })

except Exception as e:
    print(f"  ❌ FAILED: {e}")
    results.append({'name': '8-Antenna', 'passed': False})

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print(" SUMMARY")
print("="*80)

passed = sum(1 for r in results if r.get('passed', False))
total = len(results)

print(f"\nTest Results: {passed}/{total} tests passed")
print("\n" + "-"*80)
print(f"{'Test':<25} {'Runtime':<12} {'Magnitude':<12} {'Status':<10}")
print("-"*80)

for r in results:
    name = r.get('name', 'Unknown')[:24]
    runtime = f"{r.get('runtime', 0):.2f}s" if 'runtime' in r else "N/A"
    mag = f"{r.get('magnitude', 0):.3f}" if 'magnitude' in r else "N/A"
    status = "✅ PASS" if r.get('passed', False) else "❌ FAIL"
    print(f"{name:<25} {runtime:<12} {mag:<12} {status:<10}")

print("-"*80)

# CEEP vs MEEP
print(f"\nCEEP vs MEEP Validation:")
if results and results[0].get('magnitude'):
    ceep_mag = results[0].get('magnitude')
    error = results[0].get('error', 0)
    print(f"  MEEP reference: {MEEP_REFERENCE:.3f}")
    print(f"  CEEP result: {ceep_mag:.3f}")
    print(f"  Error: {error:.1f}%")

    if error < 1.0:
        print(f"  🎯 EXCELLENT - Within 1% of MEEP!")
    elif error < 5.0:
        print(f"  ✅ VERY GOOD - Within 5% of MEEP!")
    else:
        print(f"  ⚠️  Error: {error:.1f}%")

print("\n" + "="*80)
if passed == total:
    print("✅ ALL TESTS PASSED!")
    print("\nCEEP library is working correctly!")
    print("S-parameters match MEEP reference (no correction factors needed)")
elif passed > 0:
    print(f"✓ {passed}/{total} tests passed")
else:
    print("❌ ALL TESTS FAILED")
    print("Check GPU/CUDA setup")

print("="*80)
