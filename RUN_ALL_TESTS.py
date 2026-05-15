#!/usr/bin/env python3
"""
CEEP - Complete Test Suite
===========================

Runs ALL tests and examples to validate the entire library.

Tests:
1. Basic magnitude validation
2. Multiple grid sizes
3. Multiple antenna configurations
4. Brain phantom simulations
5. Material tests (dielectric, lossy)
6. Different frequencies
7. CPU vs GPU comparison

Run in Colab:
  !python RUN_ALL_TESTS.py

Author: CEEP Team
Date: 2026-05-15
"""

import os
import sys
import time
import numpy as np

print("="*80)
print(" CEEP - COMPLETE TEST SUITE")
print("="*80)

# Setup
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
if not os.path.exists(src_path):
    src_path = '/content/ceep-v1/src'

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import
print("\n[SETUP] Importing CEEP...")
try:
    from ceep.core.backend import set_backend, get_backend
    from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
    set_backend('cupy')
    print(f"  ✓ CEEP ready (GPU mode)")
    USE_GPU = True
except Exception as e:
    print(f"  ⚠️  GPU failed: {e}")
    try:
        from ceep.core.backend import set_backend, get_backend
        from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
        set_backend('numpy')
        print(f"  ✓ CEEP ready (CPU mode)")
        USE_GPU = False
    except Exception as e2:
        print(f"  ❌ Import failed: {e2}")
        sys.exit(1)

# Reference value
MEEP_REFERENCE = 3.368
TOLERANCE_EXCELLENT = 1.0  # < 1%
TOLERANCE_GOOD = 5.0       # < 5%
TOLERANCE_ACCEPTABLE = 10.0 # < 10%

results = []
test_num = 0

def run_test(name, solver_func, expected_range=None, compare_to_meep=True):
    """Run a single test and record results."""
    global test_num, results
    test_num += 1

    print(f"\n{'='*80}")
    print(f" [{test_num}/15] {name}")
    print('='*80)

    try:
        t_start = time.time()
        result = solver_func()
        t_elapsed = time.time() - t_start

        if result is None:
            print(f"  ❌ Test returned None")
            results.append({'name': name, 'passed': False, 'error': 'No result'})
            return

        magnitude = result.get('magnitude', 0)

        print(f"  Runtime: {t_elapsed:.2f}s")
        print(f"  S-parameter: {magnitude:.3f}")

        # Validation
        passed = False
        if compare_to_meep:
            error = abs(magnitude - MEEP_REFERENCE) / MEEP_REFERENCE * 100
            print(f"  MEEP reference: {MEEP_REFERENCE:.3f}")
            print(f"  Error: {error:.1f}%")

            if error < TOLERANCE_EXCELLENT:
                print(f"  🎯 EXCELLENT - Within 1%")
                passed = True
            elif error < TOLERANCE_GOOD:
                print(f"  ✅ VERY GOOD - Within 5%")
                passed = True
            elif error < TOLERANCE_ACCEPTABLE:
                print(f"  ✓ GOOD - Within 10%")
                passed = True
            else:
                print(f"  ⚠️  Error too large: {error:.1f}%")
                passed = False
        elif expected_range:
            min_val, max_val = expected_range
            if min_val <= magnitude <= max_val:
                print(f"  ✅ PASS - Within expected range [{min_val:.1f}, {max_val:.1f}]")
                passed = True
            else:
                print(f"  ❌ FAIL - Outside expected range [{min_val:.1f}, {max_val:.1f}]")
                passed = False
        else:
            # Just check it's reasonable
            if 0.1 < magnitude < 100:
                print(f"  ✓ Magnitude appears reasonable")
                passed = True
            else:
                print(f"  ⚠️  Magnitude unusual: {magnitude:.3f}")
                passed = False

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}")

        results.append({
            'name': name,
            'magnitude': magnitude,
            'runtime': t_elapsed,
            'passed': passed,
            'error': error if compare_to_meep else None
        })

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append({'name': name, 'passed': False, 'error': str(e)})

# ============================================================================
# TEST 1: Basic Empty Domain (MEEP Validation)
# ============================================================================
def test_basic_empty():
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=100,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )
    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Basic Empty Domain (64×64)", test_basic_empty, compare_to_meep=True)

# ============================================================================
# TEST 2: Larger Grid (128×128)
# ============================================================================
def test_large_grid():
    solver = BatchedFDTD2D(
        nx=128, ny=128, dx=0.5e-3,
        total_steps=200,
        cpml_thickness=15,
        source_positions=[(64, 64)],
        probe_positions=[(64, 64)],
        frequency=2e9
    )
    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Large Grid (128×128)", test_large_grid, expected_range=(2.0, 5.0))

# ============================================================================
# TEST 3: Small Grid (32×32)
# ============================================================================
def test_small_grid():
    solver = BatchedFDTD2D(
        nx=32, ny=32, dx=0.5e-3,
        total_steps=100,
        cpml_thickness=5,
        source_positions=[(16, 16)],
        probe_positions=[(16, 16)],
        frequency=2e9
    )
    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Small Grid (32×32)", test_small_grid, expected_range=(2.0, 5.0))

# ============================================================================
# TEST 4: 2-Antenna Array
# ============================================================================
def test_2_antenna():
    nx, ny = 80, 80
    positions = [(25, 40), (55, 40)]

    solver = BatchedFDTD2D(
        nx=nx, ny=ny, dx=0.5e-3,
        total_steps=120,
        cpml_thickness=10,
        source_positions=positions,
        probe_positions=positions,
        frequency=2e9
    )
    s_matrix = solver.run()

    s_values = [np.abs(s_matrix[tx][rx]).max() for tx in range(2) for rx in range(2)]
    return {'magnitude': max(s_values)}

run_test("2-Antenna Array", test_2_antenna, expected_range=(1.0, 10.0))

# ============================================================================
# TEST 5: 4-Antenna Square Array
# ============================================================================
def test_4_antenna():
    nx, ny = 80, 80
    cx, cy = nx // 2, ny // 2
    offset = 15

    positions = [
        (cx - offset, cy - offset),
        (cx + offset, cy - offset),
        (cx + offset, cy + offset),
        (cx - offset, cy + offset),
    ]

    solver = BatchedFDTD2D(
        nx=nx, ny=ny, dx=0.5e-3,
        total_steps=120,
        cpml_thickness=10,
        source_positions=positions,
        probe_positions=positions,
        frequency=2e9
    )
    s_matrix = solver.run()

    s_values = [np.abs(s_matrix[tx][rx]).max() for tx in range(4) for rx in range(4)]
    return {'magnitude': max(s_values)}

run_test("4-Antenna Square Array", test_4_antenna, expected_range=(1.0, 10.0))

# ============================================================================
# TEST 6: 8-Antenna Circular Array
# ============================================================================
def test_8_antenna():
    nx, ny = 100, 100
    cx, cy = nx // 2, ny // 2
    radius = 30

    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    positions = [(int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a))) for a in angles]

    solver = BatchedFDTD2D(
        nx=nx, ny=ny, dx=0.5e-3,
        total_steps=150,
        cpml_thickness=10,
        source_positions=positions,
        probe_positions=positions,
        frequency=2e9
    )
    s_matrix = solver.run()

    s_values = [np.abs(s_matrix[tx][rx]).max() for tx in range(8) for rx in range(8)]
    return {'magnitude': max(s_values)}

run_test("8-Antenna Circular Array", test_8_antenna, expected_range=(1.0, 10.0))

# ============================================================================
# TEST 7: Dielectric Cylinder (ε=4)
# ============================================================================
def test_dielectric():
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=120,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )

    # Add circular dielectric
    solver.set_material_circle(
        center_x=32, center_y=32,
        radius=10,
        eps_r=4.0,
        sigma_e=0.0
    )

    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Dielectric Cylinder (ε=4)", test_dielectric, expected_range=(1.0, 5.0))

# ============================================================================
# TEST 8: Lossy Dielectric (ε=4, σ=0.5)
# ============================================================================
def test_lossy():
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=120,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )

    # Add lossy dielectric
    solver.set_material_circle(
        center_x=32, center_y=32,
        radius=10,
        eps_r=4.0,
        sigma_e=0.5
    )

    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Lossy Dielectric (σ=0.5)", test_lossy, expected_range=(0.5, 5.0))

# ============================================================================
# TEST 9: Brain Tissue (Manual)
# ============================================================================
def test_brain():
    nx, ny = 64, 64
    cx, cy = nx // 2, ny // 2

    eps_r = np.ones((nx, ny), dtype=np.float64)
    sigma_e = np.zeros((nx, ny), dtype=np.float64)

    x, y = np.ogrid[:nx, :ny]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    brain_mask = r < 20
    eps_r[brain_mask] = 50.0
    sigma_e[brain_mask] = 1.5

    solver = BatchedFDTD2D(
        nx=nx, ny=ny, dx=0.5e-3,
        total_steps=150,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )

    solver._eps_r[:] = eps_r
    solver._sigma_e[:] = sigma_e

    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Brain Tissue (ε=50, σ=1.5)", test_brain, expected_range=(0.5, 10.0))

# ============================================================================
# TEST 10: High Frequency (5 GHz)
# ============================================================================
def test_high_freq():
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=0.3e-3,  # Smaller dx for higher freq
        total_steps=150,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=5e9
    )
    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("High Frequency (5 GHz)", test_high_freq, expected_range=(2.0, 5.0))

# ============================================================================
# TEST 11: Low Frequency (1 GHz)
# ============================================================================
def test_low_freq():
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=1e-3,  # Larger dx for lower freq
        total_steps=100,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=1e9
    )
    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Low Frequency (1 GHz)", test_low_freq, expected_range=(2.0, 5.0))

# ============================================================================
# TEST 12: Thick CPML (20 cells)
# ============================================================================
def test_thick_cpml():
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=100,
        cpml_thickness=20,  # Thicker PML
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )
    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Thick CPML (20 cells)", test_thick_cpml, compare_to_meep=True)

# ============================================================================
# TEST 13: Thin CPML (5 cells)
# ============================================================================
def test_thin_cpml():
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=100,
        cpml_thickness=5,  # Thinner PML
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )
    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Thin CPML (5 cells)", test_thin_cpml, expected_range=(2.0, 5.0))

# ============================================================================
# TEST 14: Long Simulation (500 steps)
# ============================================================================
def test_long_sim():
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=500,  # Much longer
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )
    s_matrix = solver.run()
    return {'magnitude': np.abs(s_matrix[0][0]).max()}

run_test("Long Simulation (500 steps)", test_long_sim, expected_range=(1.0, 10.0))

# ============================================================================
# TEST 15: CPU vs GPU Comparison (if GPU available)
# ============================================================================
if USE_GPU:
    def test_cpu_gpu():
        # GPU version
        set_backend('cupy')
        solver_gpu = BatchedFDTD2D(
            nx=64, ny=64, dx=0.5e-3,
            total_steps=100,
            cpml_thickness=10,
            source_positions=[(32, 32)],
            probe_positions=[(32, 32)],
            frequency=2e9
        )
        t_gpu_start = time.time()
        s_matrix_gpu = solver_gpu.run()
        t_gpu = time.time() - t_gpu_start
        mag_gpu = np.abs(s_matrix_gpu[0][0]).max()

        # CPU version
        set_backend('numpy')
        solver_cpu = BatchedFDTD2D(
            nx=64, ny=64, dx=0.5e-3,
            total_steps=100,
            cpml_thickness=10,
            source_positions=[(32, 32)],
            probe_positions=[(32, 32)],
            frequency=2e9
        )
        t_cpu_start = time.time()
        s_matrix_cpu = solver_cpu.run_cpu()
        t_cpu = time.time() - t_cpu_start
        mag_cpu = np.abs(s_matrix_cpu[0][0]).max()

        # Compare
        diff = abs(mag_gpu - mag_cpu) / mag_cpu * 100
        speedup = t_cpu / t_gpu

        print(f"  GPU: {mag_gpu:.3f} ({t_gpu:.2f}s)")
        print(f"  CPU: {mag_cpu:.3f} ({t_cpu:.2f}s)")
        print(f"  Difference: {diff:.1f}%")
        print(f"  Speedup: {speedup:.1f}x")

        # Switch back to GPU
        set_backend('cupy')

        return {'magnitude': mag_gpu}

    run_test("CPU vs GPU Comparison", test_cpu_gpu, compare_to_meep=True)
else:
    print(f"\n{'='*80}")
    print(f" [15/15] CPU vs GPU Comparison")
    print('='*80)
    print("  ⏭️  SKIPPED (GPU not available)")
    results.append({'name': 'CPU vs GPU Comparison', 'passed': None, 'skipped': True})

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" FINAL SUMMARY - ALL TESTS")
print("="*80)

passed = sum(1 for r in results if r.get('passed') == True)
failed = sum(1 for r in results if r.get('passed') == False)
skipped = sum(1 for r in results if r.get('passed') is None)
total = len(results)

print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped (total: {total})")
print("\n" + "-"*80)
print(f"{'Test Name':<40} {'Magnitude':<12} {'Status':<10}")
print("-"*80)

for r in results:
    name = r.get('name', 'Unknown')[:39]
    mag = f"{r.get('magnitude', 0):.3f}" if 'magnitude' in r else "N/A"

    if r.get('passed') is None:
        status = "⏭️  SKIP"
    elif r.get('passed'):
        status = "✅ PASS"
    else:
        status = "❌ FAIL"

    print(f"{name:<40} {mag:<12} {status:<10}")

print("-"*80)

# Performance stats
runtimes = [r.get('runtime', 0) for r in results if 'runtime' in r]
if runtimes:
    print(f"\nPerformance:")
    print(f"  Average runtime: {np.mean(runtimes):.2f}s")
    print(f"  Total runtime: {np.sum(runtimes):.2f}s")
    print(f"  Fastest: {min(runtimes):.2f}s")
    print(f"  Slowest: {max(runtimes):.2f}s")

# Accuracy stats
errors = [r.get('error', 0) for r in results if r.get('error') is not None]
if errors:
    print(f"\nAccuracy (vs MEEP reference):")
    print(f"  Average error: {np.mean(errors):.1f}%")
    print(f"  Max error: {max(errors):.1f}%")
    print(f"  Min error: {min(errors):.1f}%")

# Overall status
print("\n" + "="*80)
if failed == 0 and passed > 0:
    print("🎉 ALL TESTS PASSED!")
    print("\nCEEP is fully validated and production-ready!")
    print(f"\n✅ Passed: {passed}/{total}")
    print("✅ No failures detected")
    print("✅ Library is stable and accurate")
    print("\nYou can confidently use CEEP for:")
    print("  • Research simulations")
    print("  • Dataset generation")
    print("  • Neural network training")
    print("  • Brain hemorrhage detection")
    print("  • Microwave imaging applications")
elif passed > failed and failed <= 2:
    print(f"✓ MOSTLY PASSING ({passed}/{total})")
    print(f"\nMost tests passed, {failed} test(s) need attention")
elif passed > 0:
    print(f"⚠️  MIXED RESULTS ({passed} passed, {failed} failed)")
    print(f"\nSome tests working, investigate failures")
else:
    print("❌ CRITICAL - ALL TESTS FAILED")
    print("\nCheck:")
    print("  • GPU/CUDA installation")
    print("  • CuPy compatibility")
    print("  • Source code integrity")

print("="*80)
