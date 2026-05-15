#!/usr/bin/env python3
"""
Debug CPML - Check if CPML code is actually running
"""

import os
import sys
import numpy as np

src_path = os.path.join(os.getcwd(), 'src')
if not os.path.exists(src_path):
    src_path = '/content/ceep-v1/src'
sys.path.insert(0, src_path)

from ceep.core.backend import set_backend
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

set_backend('cupy')

print("Creating solver...")
solver = BatchedFDTD2D(
    nx=64, ny=64, dx=0.5e-3, total_steps=10,
    cpml_thickness=10,
    source_positions=[(32, 32)],
    probe_positions=[(32, 32)],
    frequency=2e9
)

print(f"\nCPML thickness: {solver.cpml_n}")
print(f"Grid size: {solver.nx} x {solver.ny}")

# Build to initialize CPML
solver._build()

print(f"\nCPML b_x values (first 5): {solver.cpml_b_x[:5]}")
print(f"CPML c_x values (first 5): {solver.cpml_c_x[:5]}")

print(f"\nExpected:")
print(f"  b should be between 0 and 1 (e.g., 0.1 to 0.99)")
print(f"  c should be NEGATIVE (b - 1), between -1 and 0")

if solver.cpml_c_x[0] > 0:
    print("\n❌ ERROR: c coefficients are POSITIVE! Should be negative!")
elif solver.cpml_c_x[0] == 0:
    print("\n❌ ERROR: c coefficients are ZERO! CPML not working!")
else:
    print("\n✓ c coefficients are negative (correct)")

# Check psi arrays exist
print(f"\nPsi array shapes:")
print(f"  psi_hxy_lo: {solver.psi_hxy_lo.shape}")
print(f"  psi_ezx_lo: {solver.psi_ezx_lo.shape}")

# Try running one step
print(f"\nRunning 1 timestep...")
try:
    import cupy as cp

    # Save initial field
    ez_before = float(cp.max(cp.abs(solver.ez)))

    # Single step
    solver.run()

    # Check after
    ez_after = float(cp.max(cp.abs(solver.ez)))

    print(f"  Ez before: {ez_before:.6f}")
    print(f"  Ez after:  {ez_after:.6f}")

    # Check if psi got updated
    psi_max = float(cp.max(cp.abs(solver.psi_hxy_lo)))
    print(f"  Max psi value: {psi_max:.6f}")

    if psi_max == 0:
        print("\n❌ WARNING: Psi arrays are still zero! CPML not updating!")
    else:
        print("\n✓ Psi arrays have non-zero values (CPML is updating)")

except Exception as e:
    print(f"\n❌ ERROR during run: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
