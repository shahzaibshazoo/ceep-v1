#!/usr/bin/env python3
"""
Detailed CPML debugging - trace execution
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
import cupy as cp

set_backend('cupy')

print("Creating solver...")
solver = BatchedFDTD2D(
    nx=64, ny=64, dx=0.5e-3, total_steps=5,
    cpml_thickness=10,
    source_positions=[(32, 32)],
    probe_positions=[(32, 32)],
    frequency=2e9
)

print("\nManually building and running...")
solver._build()

print(f"CPML thickness: {solver.cpml_n}")
print(f"Interior region: x:[{solver.cpml_n}:{solver.nx-solver.cpml_n}], y:[{solver.cpml_n}:{solver.ny-solver.cpml_n}]")

# Manually run one step to debug
n = solver.cpml_n
nx, ny = solver.nx, solver.ny

print("\n=== Before timestep 0 ===")
print(f"Ez max: {float(cp.max(cp.abs(solver.ez))):.6f}")
print(f"Psi_hxy_lo max: {float(cp.max(cp.abs(solver.psi_hxy_lo))):.6f}")

# Source injection for step 0
wval = float(solver.waveform[0])
print(f"\nInjecting source: value={wval:.6f}")
solver.ez[0, 32, 32] += wval

print(f"Ez at source: {float(solver.ez[0, 32, 32]):.6f}")
print(f"Ez max: {float(cp.max(cp.abs(solver.ez))):.6f}")

# Step 1: H-field update
print("\n=== H-field update ===")

# Check if interior update will run
print(f"Interior Hx update range: x:[{n}:{nx-n}], y:[{n}:{ny-1}]")
print(f"Interior Hy update range: x:[{n}:{nx-1}], y:[{n}:{ny-n}]")

# Do interior H update
solver.hx[:, n:nx-n, n:ny-1] = (
    solver.da[:, n:nx-n, n:ny-1] * solver.hx[:, n:nx-n, n:ny-1]
    - solver.db[:, n:nx-n, n:ny-1] * solver.inv_dy * (
        solver.ez[:, n:nx-n, n+1:ny] - solver.ez[:, n:nx-n, n:ny-1]
    )
)
solver.hy[:, n:nx-1, n:ny-n] = (
    solver.da[:, n:nx-1, n:ny-n] * solver.hy[:, n:nx-1, n:ny-n]
    + solver.db[:, n:nx-1, n:ny-n] * solver.inv_dx * (
        solver.ez[:, n+1:nx, n:ny-n] - solver.ez[:, n:nx-1, n:ny-n]
    )
)

print(f"After interior H update:")
print(f"  Hx max: {float(cp.max(cp.abs(solver.hx))):.6f}")
print(f"  Hy max: {float(cp.max(cp.abs(solver.hy))):.6f}")

# Now call CPML H update
print("\n=== Calling _apply_h_cpml() ===")

# Check left X boundary (i=0)
i = 0
if i < nx - 1:
    dEz_dx = (solver.ez[:, i+1, :] - solver.ez[:, i, :]) * solver.inv_dx
    print(f"Left boundary (i={i}):")
    print(f"  dEz_dx max: {float(cp.max(cp.abs(dEz_dx))):.6f}")
    print(f"  b[{i}] = {float(solver.cpml_b_x[i]):.6f}")
    print(f"  c[{i}] = {float(solver.cpml_c_x[i]):.6f}")

    # Update psi
    psi_old = solver.psi_hxy_lo[:, i, :].copy()
    solver.psi_hxy_lo[:, i, :] = (
        solver.cpml_b_x[i] * solver.psi_hxy_lo[:, i, :]
        + solver.cpml_c_x[i] * dEz_dx
    )
    psi_new = solver.psi_hxy_lo[:, i, :]

    print(f"  psi_old max: {float(cp.max(cp.abs(psi_old))):.6f}")
    print(f"  psi_new max: {float(cp.max(cp.abs(psi_new))):.6f}")
    print(f"  psi_new should be: b*0 + c*deriv = {float(solver.cpml_c_x[i])} * {float(cp.max(cp.abs(dEz_dx))):.6f}")
    print(f"                    = {float(solver.cpml_c_x[i] * cp.max(cp.abs(dEz_dx))):.6f}")

print("\n=== Analysis ===")
if float(cp.max(cp.abs(dEz_dx))) < 1e-10:
    print("❌ Problem: dEz_dx is ZERO in PML region!")
    print("   This means Ez hasn't propagated to the boundaries yet.")
    print("   Try more timesteps or check if interior update is correct.")
else:
    print("✓ dEz_dx is non-zero")
    if float(cp.max(cp.abs(psi_new))) < 1e-10:
        print("❌ Problem: psi is still ZERO after update!")
        print("   Bug in psi update equation.")
    else:
        print("✓ psi updated to non-zero value!")
