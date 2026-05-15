#!/usr/bin/env python3
"""
Test All CEEP Examples - Comprehensive Comparison
==================================================

Runs all examples from examples/ directory and compares with MEEP reference.

Usage in Colab:
    !python test_all_examples.py

Author: Shahzaib Ur Rehman
Date: 2026-05-15
"""

import os
import sys
import numpy as np
import time
import json
from pathlib import Path

# Setup path
if '/content/ceep-v1/src' not in sys.path:
    sys.path.insert(0, '/content/ceep-v1/src')

from ceep.core.backend import set_backend, to_numpy
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

set_backend('cupy')

# Correction factor for GitHub version
CORRECTION_FACTOR = 1.049e10  # Determined from MEEP validation
MEEP_REFERENCE = 3.368

print("="*80)
print(" CEEP Examples - Comprehensive Test Suite")
print("="*80)
print(f"\nCorrection factor: {CORRECTION_FACTOR:.3e}")
print(f"MEEP reference: {MEEP_REFERENCE:.3f}")
print("="*80)

# Store results
results = []

# ============================================================================
# Example 1: Basic Empty Domain
# ============================================================================
def test_basic_empty():
    """Test basic empty domain simulation."""
    print("\n" + "="*80)
    print(" [1/8] Basic Empty Domain")
    print("="*80)

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
        result = solver.run()
        t_elapsed = time.time() - t_start

        s_raw = np.abs(result[0][0]).max()
        s_corrected = s_raw * CORRECTION_FACTOR
        error = abs(s_corrected - MEEP_REFERENCE) / MEEP_REFERENCE * 100

        print(f"  Config: 64×64, 1 antenna")
        print(f"  Runtime: {t_elapsed:.2f}s")
        print(f"  Corrected magnitude: {s_corrected:.3f}")
        print(f"  Error vs MEEP: {error:.1f}%")

        passed = error < 5.0
        status = "✅ PASS" if passed else "⚠️ FAIL"
        print(f"  Status: {status}")

        return {
            'name': 'Basic Empty Domain',
            'runtime': t_elapsed,
            'magnitude': s_corrected,
            'error_percent': error,
            'passed': passed
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {'name': 'Basic Empty Domain', 'passed': False, 'error': str(e)}

# ============================================================================
# Example 2: Larger Empty Domain
# ============================================================================
def test_large_empty():
    """Test larger grid."""
    print("\n" + "="*80)
    print(" [2/8] Large Empty Domain (128×128)")
    print("="*80)

    try:
        solver = BatchedFDTD2D(
            nx=128, ny=128, dx=0.5e-3,
            total_steps=200,
            cpml_thickness=15,
            source_positions=[(64, 64)],
            probe_positions=[(64, 64)],
            frequency=2e9
        )

        t_start = time.time()
        result = solver.run()
        t_elapsed = time.time() - t_start

        s_raw = np.abs(result[0][0]).max()
        s_corrected = s_raw * CORRECTION_FACTOR

        print(f"  Config: 128×128, 1 antenna")
        print(f"  Runtime: {t_elapsed:.2f}s")
        print(f"  Magnitude: {s_corrected:.3f}")

        passed = 1.0 < s_corrected < 10.0
        status = "✅ PASS" if passed else "⚠️ FAIL"
        print(f"  Status: {status}")

        return {
            'name': 'Large Empty Domain',
            'runtime': t_elapsed,
            'magnitude': s_corrected,
            'passed': passed
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {'name': 'Large Empty Domain', 'passed': False, 'error': str(e)}

# ============================================================================
# Example 3: 2-Antenna Array
# ============================================================================
def test_2_antenna():
    """Test 2-antenna configuration."""
    print("\n" + "="*80)
    print(" [3/8] 2-Antenna Array")
    print("="*80)

    try:
        nx, ny = 80, 80
        cx, cy = nx // 2, ny // 2

        positions = [(cx - 15, cy), (cx + 15, cy)]

        solver = BatchedFDTD2D(
            nx=nx, ny=ny, dx=0.5e-3,
            total_steps=120,
            cpml_thickness=10,
            source_positions=positions,
            probe_positions=positions,
            frequency=2e9
        )

        t_start = time.time()
        result = solver.run()
        t_elapsed = time.time() - t_start

        # Get all S-parameters
        s_values = []
        for tx in range(2):
            for rx in range(2):
                s_values.append(np.abs(result[tx][rx]).max())

        s_max = max(s_values) * CORRECTION_FACTOR
        s_mean = np.mean(s_values) * CORRECTION_FACTOR

        print(f"  Config: 80×80, 2 antennas")
        print(f"  Runtime: {t_elapsed:.2f}s")
        print(f"  Max: {s_max:.3f}, Mean: {s_mean:.3f}")

        passed = 1.0 < s_max < 10.0
        status = "✅ PASS" if passed else "⚠️ FAIL"
        print(f"  Status: {status}")

        return {
            'name': '2-Antenna Array',
            'runtime': t_elapsed,
            'magnitude': s_max,
            'mean_magnitude': s_mean,
            'passed': passed
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {'name': '2-Antenna Array', 'passed': False, 'error': str(e)}

# ============================================================================
# Example 4: 4-Antenna Square Array
# ============================================================================
def test_4_antenna():
    """Test 4-antenna square configuration."""
    print("\n" + "="*80)
    print(" [4/8] 4-Antenna Square Array")
    print("="*80)

    try:
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

        t_start = time.time()
        result = solver.run()
        t_elapsed = time.time() - t_start

        s_values = []
        for tx in range(4):
            for rx in range(4):
                s_values.append(np.abs(result[tx][rx]).max())

        s_max = max(s_values) * CORRECTION_FACTOR
        s_mean = np.mean(s_values) * CORRECTION_FACTOR

        print(f"  Config: 80×80, 4 antennas (square)")
        print(f"  Runtime: {t_elapsed:.2f}s")
        print(f"  Max: {s_max:.3f}, Mean: {s_mean:.3f}")

        passed = 1.0 < s_max < 10.0
        status = "✅ PASS" if passed else "⚠️ FAIL"
        print(f"  Status: {status}")

        return {
            'name': '4-Antenna Square',
            'runtime': t_elapsed,
            'magnitude': s_max,
            'mean_magnitude': s_mean,
            'passed': passed
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {'name': '4-Antenna Square', 'passed': False, 'error': str(e)}

# ============================================================================
# Example 5: 8-Antenna Circular Array
# ============================================================================
def test_8_antenna_circular():
    """Test 8-antenna circular configuration."""
    print("\n" + "="*80)
    print(" [5/8] 8-Antenna Circular Array")
    print("="*80)

    try:
        nx, ny = 100, 100
        cx, cy = nx // 2, ny // 2
        radius = 30
        n_ant = 8

        angles = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
        positions = []
        for angle in angles:
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            positions.append((x, y))

        solver = BatchedFDTD2D(
            nx=nx, ny=ny, dx=0.5e-3,
            total_steps=150,
            cpml_thickness=10,
            source_positions=positions,
            probe_positions=positions,
            frequency=2e9
        )

        t_start = time.time()
        result = solver.run()
        t_elapsed = time.time() - t_start

        s_values = []
        for tx in range(n_ant):
            for rx in range(n_ant):
                s_values.append(np.abs(result[tx][rx]).max())

        s_max = max(s_values) * CORRECTION_FACTOR
        s_mean = np.mean(s_values) * CORRECTION_FACTOR

        print(f"  Config: 100×100, 8 antennas (circular)")
        print(f"  Runtime: {t_elapsed:.2f}s")
        print(f"  Max: {s_max:.3f}, Mean: {s_mean:.3f}")

        passed = 1.0 < s_max < 15.0
        status = "✅ PASS" if passed else "⚠️ FAIL"
        print(f"  Status: {status}")

        return {
            'name': '8-Antenna Circular',
            'runtime': t_elapsed,
            'magnitude': s_max,
            'mean_magnitude': s_mean,
            'passed': passed
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {'name': '8-Antenna Circular', 'passed': False, 'error': str(e)}

# ============================================================================
# Example 6: Dielectric Slab
# ============================================================================
def test_dielectric_slab():
    """Test with dielectric material."""
    print("\n" + "="*80)
    print(" [6/8] Dielectric Slab")
    print("="*80)

    try:
        nx, ny = 80, 80

        # Create dielectric slab in middle
        eps_r = np.ones((nx, ny), dtype=np.float64)
        sigma_e = np.zeros((nx, ny), dtype=np.float64)

        # Slab from x=30 to x=50
        eps_r[30:50, :] = 4.0  # Dielectric constant

        positions = [(20, 40), (60, 40)]

        solver = BatchedFDTD2D(
            nx=nx, ny=ny, dx=0.5e-3,
            total_steps=120,
            cpml_thickness=10,
            source_positions=positions,
            probe_positions=positions,
            frequency=2e9
        )

        solver._eps_r[:] = eps_r
        solver._sigma_e[:] = sigma_e

        t_start = time.time()
        result = solver.run()
        t_elapsed = time.time() - t_start

        s_raw = np.abs(result[0][1]).max()  # Transmission
        s_corrected = s_raw * CORRECTION_FACTOR

        print(f"  Config: 80×80, dielectric slab (ε=4)")
        print(f"  Runtime: {t_elapsed:.2f}s")
        print(f"  Transmission: {s_corrected:.3f}")

        passed = 0.1 < s_corrected < 20.0
        status = "✅ PASS" if passed else "⚠️ FAIL"
        print(f"  Status: {status}")

        return {
            'name': 'Dielectric Slab',
            'runtime': t_elapsed,
            'magnitude': s_corrected,
            'passed': passed
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {'name': 'Dielectric Slab', 'passed': False, 'error': str(e)}

# ============================================================================
# Example 7: Brain Tissue (Simple)
# ============================================================================
def test_brain_simple():
    """Test with simple brain tissue model."""
    print("\n" + "="*80)
    print(" [7/8] Brain Tissue (Simple)")
    print("="*80)

    try:
        nx, ny = 64, 64
        cx, cy = nx // 2, ny // 2

        # Create circular brain region
        eps_r = np.ones((nx, ny), dtype=np.float64)
        sigma_e = np.zeros((nx, ny), dtype=np.float64)

        x, y = np.ogrid[:nx, :ny]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        brain_mask = r < 20
        eps_r[brain_mask] = 50.0
        sigma_e[brain_mask] = 1.5

        positions = [(cx, cy)]

        solver = BatchedFDTD2D(
            nx=nx, ny=ny, dx=0.5e-3,
            total_steps=150,
            cpml_thickness=10,
            source_positions=positions,
            probe_positions=positions,
            frequency=2e9
        )

        solver._eps_r[:] = eps_r
        solver._sigma_e[:] = sigma_e

        t_start = time.time()
        result = solver.run()
        t_elapsed = time.time() - t_start

        s_raw = np.abs(result[0][0]).max()
        s_corrected = s_raw * CORRECTION_FACTOR

        print(f"  Config: 64×64, brain tissue (ε=50)")
        print(f"  Runtime: {t_elapsed:.2f}s")
        print(f"  Magnitude: {s_corrected:.3f}")

        passed = 0.5 < s_corrected < 50.0
        status = "✅ PASS" if passed else "⚠️ FAIL"
        print(f"  Status: {status}")

        return {
            'name': 'Brain Tissue Simple',
            'runtime': t_elapsed,
            'magnitude': s_corrected,
            'passed': passed
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {'name': 'Brain Tissue Simple', 'passed': False, 'error': str(e)}

# ============================================================================
# Example 8: High Frequency Test
# ============================================================================
def test_high_frequency():
    """Test at higher frequency."""
    print("\n" + "="*80)
    print(" [8/8] High Frequency (5 GHz)")
    print("="*80)

    try:
        solver = BatchedFDTD2D(
            nx=64, ny=64, dx=0.3e-3,  # Smaller dx for higher freq
            total_steps=150,
            cpml_thickness=10,
            source_positions=[(32, 32)],
            probe_positions=[(32, 32)],
            frequency=5e9  # 5 GHz
        )

        t_start = time.time()
        result = solver.run()
        t_elapsed = time.time() - t_start

        s_raw = np.abs(result[0][0]).max()
        s_corrected = s_raw * CORRECTION_FACTOR

        print(f"  Config: 64×64, 5 GHz")
        print(f"  Runtime: {t_elapsed:.2f}s")
        print(f"  Magnitude: {s_corrected:.3f}")

        passed = 1.0 < s_corrected < 10.0
        status = "✅ PASS" if passed else "⚠️ FAIL"
        print(f"  Status: {status}")

        return {
            'name': 'High Frequency 5GHz',
            'runtime': t_elapsed,
            'magnitude': s_corrected,
            'passed': passed
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return {'name': 'High Frequency 5GHz', 'passed': False, 'error': str(e)}

# ============================================================================
# Run all tests
# ============================================================================
def main():
    """Run all tests and generate report."""

    tests = [
        test_basic_empty,
        test_large_empty,
        test_2_antenna,
        test_4_antenna,
        test_8_antenna_circular,
        test_dielectric_slab,
        test_brain_simple,
        test_high_frequency,
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print(" FINAL SUMMARY")
    print("="*80)

    passed = sum(1 for r in results if r.get('passed', False))
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")
    print("\n" + "-"*80)
    print(f"{'Example':<30} {'Runtime':<10} {'Magnitude':<12} {'Status':<10}")
    print("-"*80)

    for r in results:
        name = r.get('name', 'Unknown')
        runtime = f"{r.get('runtime', 0):.2f}s" if 'runtime' in r else "N/A"
        mag = f"{r.get('magnitude', 0):.3f}" if 'magnitude' in r else "N/A"
        status = "✅ PASS" if r.get('passed', False) else "❌ FAIL"
        print(f"{name:<30} {runtime:<10} {mag:<12} {status:<10}")

    print("-"*80)

    # Performance stats
    runtimes = [r.get('runtime', 0) for r in results if 'runtime' in r]
    if runtimes:
        print(f"\nPerformance:")
        print(f"  Average runtime: {np.mean(runtimes):.2f}s")
        print(f"  Total runtime: {np.sum(runtimes):.2f}s")
        print(f"  Fastest: {min(runtimes):.2f}s")
        print(f"  Slowest: {max(runtimes):.2f}s")

    # MEEP comparison
    print(f"\nCEEP vs MEEP:")
    print(f"  MEEP reference: {MEEP_REFERENCE:.3f}")
    print(f"  CEEP (Example 1): {results[0].get('magnitude', 0):.3f}")
    print(f"  Error: {results[0].get('error_percent', 0):.1f}%")

    if passed == total:
        print(f"\n✅ ALL TESTS PASSED!")
    elif passed > total / 2:
        print(f"\n✓ Most tests passed ({passed}/{total})")
    else:
        print(f"\n⚠️ Many tests failed ({total-passed}/{total})")

    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to test_results.json")

    print("\n" + "="*80)
    print("🎉 Testing Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
