#!/usr/bin/env python3
"""
Test CPML Fix - Quick Validation
=================================

Tests if the CPML c coefficient fix resolves the boundary reflection issue.

Run on Colab:
  !git pull
  !python TEST_CPML_FIX.py

Should show:
- Empty space: stable magnitude (~3-30 range)
- Brain tissue: similar magnitude to empty
- NO exponential growth

Author: CEEP Team
Date: 2026-05-15
"""

import os
import sys
import numpy as np

print("="*80)
print(" TESTING CPML FIX")
print("="*80)

# Setup
src_path = os.path.join(os.getcwd(), 'src')
if not os.path.exists(src_path):
    src_path = '/content/ceep-v1/src'
sys.path.insert(0, src_path)

from ceep.core.backend import set_backend
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

set_backend('cupy')

# Test parameters
NX = 64
NY = 64
DX = 0.5e-3
FREQUENCY = 2e9
CPML_THICKNESS = 10

print(f"\nConfiguration:")
print(f"  Grid: {NX}×{NY}")
print(f"  Resolution: {DX*1000:.2f} mm")
print(f"  Frequency: {FREQUENCY/1e9:.1f} GHz")
print(f"  CPML thickness: {CPML_THICKNESS}")

# ============================================================================
# TEST 1: Empty space - short (100 steps)
# ============================================================================
print("\n" + "="*80)
print(" TEST 1: Empty Space - 100 steps")
print("="*80)

solver1 = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, total_steps=100,
    cpml_thickness=CPML_THICKNESS,
    source_positions=[(32, 32)],
    probe_positions=[(32, 32)],
    frequency=FREQUENCY
)

s1 = solver1.run()
mag1_short = np.abs(s1[0][0]).max()
print(f"  S-parameter (100 steps): {mag1_short:.3f}")

# ============================================================================
# TEST 2: Empty space - LONG (500 steps) - THIS SHOULD PASS NOW
# ============================================================================
print("\n" + "="*80)
print(" TEST 2: Empty Space - 500 steps (CRITICAL TEST)")
print("="*80)

solver2 = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, total_steps=500,
    cpml_thickness=CPML_THICKNESS,
    source_positions=[(32, 32)],
    probe_positions=[(32, 32)],
    frequency=FREQUENCY
)

s2 = solver2.run()
mag2_long = np.abs(s2[0][0]).max()
print(f"  S-parameter (500 steps): {mag2_long:.3f}")

# Check for growth
growth_ratio = mag2_long / mag1_short
print(f"\n  Growth ratio (500/100): {growth_ratio:.2f}x")

if growth_ratio < 2.0:
    print(f"  ✅ PASS - No exponential growth!")
else:
    print(f"  ❌ FAIL - Still growing exponentially")

# ============================================================================
# TEST 3: Brain tissue - 100 steps
# ============================================================================
print("\n" + "="*80)
print(" TEST 3: Brain Tissue - 100 steps")
print("="*80)

solver3 = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, total_steps=100,
    cpml_thickness=CPML_THICKNESS,
    source_positions=[(32, 32)],
    probe_positions=[(32, 32)],
    frequency=FREQUENCY
)

# Brain tissue everywhere
eps_r3 = np.ones((NX, NY), dtype=np.float64) * 50.0
sigma_e3 = np.ones((NX, NY), dtype=np.float64) * 1.5

solver3._eps_r[:] = eps_r3
solver3._sigma_e[:] = sigma_e3

s3 = solver3.run()
mag3_brain = np.abs(s3[0][0]).max()
print(f"  S-parameter (brain, 100 steps): {mag3_brain:.3f}")

# ============================================================================
# TEST 4: Brain tissue - LONG (500 steps)
# ============================================================================
print("\n" + "="*80)
print(" TEST 4: Brain Tissue - 500 steps (CRITICAL TEST)")
print("="*80)

solver4 = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, total_steps=500,
    cpml_thickness=CPML_THICKNESS,
    source_positions=[(32, 32)],
    probe_positions=[(32, 32)],
    frequency=FREQUENCY
)

solver4._eps_r[:] = eps_r3
solver4._sigma_e[:] = sigma_e3

s4 = solver4.run()
mag4_brain_long = np.abs(s4[0][0]).max()
print(f"  S-parameter (brain, 500 steps): {mag4_brain_long:.3f}")

# Check for growth
growth_ratio_brain = mag4_brain_long / mag3_brain
print(f"\n  Growth ratio (500/100): {growth_ratio_brain:.2f}x")

if growth_ratio_brain < 2.0:
    print(f"  ✅ PASS - No exponential growth!")
else:
    print(f"  ❌ FAIL - Still growing exponentially")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "="*80)
print(" FINAL VERDICT")
print("="*80)

passed = 0
total = 2

if growth_ratio < 2.0:
    passed += 1
    print("✅ Empty space: CPML working")
else:
    print("❌ Empty space: CPML still broken")

if growth_ratio_brain < 2.0:
    passed += 1
    print("✅ Brain tissue: CPML working")
else:
    print("❌ Brain tissue: CPML still broken")

print(f"\nResults: {passed}/2 tests passed")

if passed == 2:
    print("\n🎉 CPML FIX SUCCESSFUL!")
    print("\nNext steps:")
    print("  1. Re-run CEEP reference simulation on Colab")
    print("  2. Re-run compare_ceep_meep.py")
    print("  3. Validate accuracy vs MEEP")
    print("  4. Run full test suite (RUN_ALL_TESTS.py)")
    print("  5. Begin dataset generation")
else:
    print("\n❌ CPML STILL BROKEN")
    print("\nNeed to investigate further:")
    print("  - Check CPML coefficient calculation")
    print("  - Verify psi array updates")
    print("  - Check field derivative computation")
    print("  - Review boundary indexing")

print("="*80)
