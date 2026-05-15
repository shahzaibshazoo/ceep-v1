#!/usr/bin/env python3
"""
Test if materials are actually affecting CEEP simulations
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

NX, NY = 64, 64
DX = 0.5e-3
positions = [(32, 32)]

print("="*60)
print("TEST 1: Empty space (eps_r=1)")
print("="*60)

solver1 = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, total_steps=100, cpml_thickness=10,
    source_positions=positions, probe_positions=positions, frequency=2e9
)

# Don't set any materials (defaults to eps_r=1)
print(f"eps_r before run: min={solver1._eps_r.min():.1f}, max={solver1._eps_r.max():.1f}")
print(f"sigma_e before run: min={solver1._sigma_e.min():.3f}, max={solver1._sigma_e.max():.3f}")

s1 = solver1.run()
mag1 = np.abs(s1[0][0]).max()
print(f"S-parameter: {mag1:.3f}")

print("\n" + "="*60)
print("TEST 2: Brain tissue (eps_r=50)")
print("="*60)

solver2 = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, total_steps=100, cpml_thickness=10,
    source_positions=positions, probe_positions=positions, frequency=2e9
)

# Set brain tissue everywhere
solver2._eps_r[:] = 50.0
solver2._sigma_e[:] = 1.5

print(f"eps_r before run: min={solver2._eps_r.min():.1f}, max={solver2._eps_r.max():.1f}")
print(f"sigma_e before run: min={solver2._sigma_e.min():.3f}, max={solver2._sigma_e.max():.3f}")

s2 = solver2.run()
mag2 = np.abs(s2[0][0]).max()
print(f"S-parameter: {mag2:.3f}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Empty space: {mag1:.3f}")
print(f"Brain tissue: {mag2:.3f}")
print(f"Difference: {abs(mag1 - mag2):.3f}")

if abs(mag1 - mag2) < 0.1:
    print("\n❌ MATERIALS NOT WORKING - Same result!")
else:
    print(f"\n✅ MATERIALS WORKING - {abs(mag1-mag2)/mag1*100:.1f}% difference")
