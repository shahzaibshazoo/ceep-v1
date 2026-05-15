#!/usr/bin/env python3
"""
CEEP Reference Simulation - Run on Colab
=========================================

Creates a standardized brain imaging scenario and saves results.
Compare with MEEP reference simulation run locally.

Run in Colab:
  !python ceep_reference_simulation.py

Outputs: ceep_results.json
"""

import os
import sys
import json
import numpy as np
import time

print("="*80)
print(" CEEP REFERENCE SIMULATION")
print("="*80)

# Setup
src_path = os.path.join(os.getcwd(), 'src')
if not os.path.exists(src_path):
    src_path = '/content/ceep-v1/src'
sys.path.insert(0, src_path)

from ceep.core.backend import set_backend
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

set_backend('cupy')

# STANDARD PARAMETERS (must match MEEP simulation exactly)
NX = 64
NY = 64
DX = 0.5e-3  # 0.5 mm
FREQUENCY = 2e9  # 2 GHz
TOTAL_STEPS = 100
CPML_THICKNESS = 10

# Single antenna for simplicity
ANTENNA_X = 32
ANTENNA_Y = 32

print(f"\nConfiguration:")
print(f"  Grid: {NX}×{NY}")
print(f"  Resolution: {DX*1000:.2f} mm")
print(f"  Frequency: {FREQUENCY/1e9:.1f} GHz")
print(f"  Timesteps: {TOTAL_STEPS}")
print(f"  Antenna: ({ANTENNA_X}, {ANTENNA_Y})")

# Brain tissue properties (must match MEEP)
BRAIN_EPS_R = 50.0
BRAIN_SIGMA = 1.5  # S/m
BLOOD_EPS_R = 60.0  # Hemorrhage
BLOOD_SIGMA = 2.0  # S/m

print(f"  Brain: ε={BRAIN_EPS_R}, σ={BRAIN_SIGMA} S/m")
print(f"  Blood: ε={BLOOD_EPS_R}, σ={BLOOD_SIGMA} S/m")

# ============================================================================
# SCENARIO 1: Empty space (baseline)
# ============================================================================
print("\n" + "="*80)
print(" SCENARIO 1: Empty Space")
print("="*80)

solver1 = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, total_steps=TOTAL_STEPS,
    cpml_thickness=CPML_THICKNESS,
    source_positions=[(ANTENNA_X, ANTENNA_Y)],
    probe_positions=[(ANTENNA_X, ANTENNA_Y)],
    frequency=FREQUENCY
)

t_start = time.time()
s1 = solver1.run()
t1 = time.time() - t_start

signal1 = s1[0][0]
mag1 = np.abs(signal1).max()

print(f"  Runtime: {t1:.2f}s")
print(f"  S-parameter: {mag1:.3f}")
print(f"  Signal length: {len(signal1)}")

# ============================================================================
# SCENARIO 2: Brain tissue (uniform)
# ============================================================================
print("\n" + "="*80)
print(" SCENARIO 2: Brain Tissue (Uniform)")
print("="*80)

eps_r2 = np.ones((NX, NY), dtype=np.float64)
sigma_e2 = np.zeros((NX, NY), dtype=np.float64)

# Brain tissue everywhere
cx, cy = NX // 2, NY // 2
x, y = np.ogrid[:NX, :NY]
brain_r = np.sqrt((x - cx)**2 + (y - cy)**2)
brain_mask = brain_r < 20  # 20 pixels = 10mm radius

eps_r2[brain_mask] = BRAIN_EPS_R
sigma_e2[brain_mask] = BRAIN_SIGMA

solver2 = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, total_steps=TOTAL_STEPS,
    cpml_thickness=CPML_THICKNESS,
    source_positions=[(ANTENNA_X, ANTENNA_Y)],
    probe_positions=[(ANTENNA_X, ANTENNA_Y)],
    frequency=FREQUENCY
)

solver2._eps_r[:] = eps_r2
solver2._sigma_e[:] = sigma_e2

t_start = time.time()
s2 = solver2.run()
t2 = time.time() - t_start

signal2 = s2[0][0]
mag2 = np.abs(signal2).max()

print(f"  Runtime: {t2:.2f}s")
print(f"  S-parameter: {mag2:.3f}")
print(f"  Signal length: {len(signal2)}")

# ============================================================================
# SCENARIO 3: Brain with small hemorrhage
# ============================================================================
print("\n" + "="*80)
print(" SCENARIO 3: Brain + Small Hemorrhage (5mm)")
print("="*80)

eps_r3 = eps_r2.copy()
sigma_e3 = sigma_e2.copy()

# Add hemorrhage
hem_x, hem_y = cx + 8, cy + 5
hem_r = np.sqrt((x - hem_x)**2 + (y - hem_y)**2)
hem_mask = hem_r < 10  # 10 pixels = 5mm radius

eps_r3[hem_mask] = BLOOD_EPS_R
sigma_e3[hem_mask] = BLOOD_SIGMA

solver3 = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX, total_steps=TOTAL_STEPS,
    cpml_thickness=CPML_THICKNESS,
    source_positions=[(ANTENNA_X, ANTENNA_Y)],
    probe_positions=[(ANTENNA_X, ANTENNA_Y)],
    frequency=FREQUENCY
)

solver3._eps_r[:] = eps_r3
solver3._sigma_e[:] = sigma_e3

t_start = time.time()
s3 = solver3.run()
t3 = time.time() - t_start

signal3 = s3[0][0]
mag3 = np.abs(signal3).max()

print(f"  Runtime: {t3:.2f}s")
print(f"  S-parameter: {mag3:.3f}")
print(f"  Signal length: {len(signal3)}")

# ============================================================================
# Save results
# ============================================================================
results = {
    'parameters': {
        'nx': int(NX),
        'ny': int(NY),
        'dx': float(DX),
        'frequency': float(FREQUENCY),
        'total_steps': int(TOTAL_STEPS),
        'cpml_thickness': int(CPML_THICKNESS),
        'antenna_position': [int(ANTENNA_X), int(ANTENNA_Y)],
        'brain_eps_r': float(BRAIN_EPS_R),
        'brain_sigma': float(BRAIN_SIGMA),
        'blood_eps_r': float(BLOOD_EPS_R),
        'blood_sigma': float(BLOOD_SIGMA)
    },
    'scenarios': {
        'empty': {
            's_parameter': float(mag1),
            'runtime': float(t1),
            'signal_real': signal1.real.tolist(),
            'signal_imag': signal1.imag.tolist()
        },
        'brain': {
            's_parameter': float(mag2),
            'runtime': float(t2),
            'signal_real': signal2.real.tolist(),
            'signal_imag': signal2.imag.tolist()
        },
        'hemorrhage': {
            's_parameter': float(mag3),
            'runtime': float(t3),
            'signal_real': signal3.real.tolist(),
            'signal_imag': signal3.imag.tolist()
        }
    },
    'solver': 'CEEP',
    'backend': 'GPU (CuPy)'
}

with open('ceep_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print(" RESULTS SAVED")
print("="*80)
print(f"Saved to: ceep_results.json")
print(f"\nSummary:")
print(f"  Empty space:  {mag1:.3f}")
print(f"  Brain tissue: {mag2:.3f}")
print(f"  Hemorrhage:   {mag3:.3f}")
print(f"\nTotal runtime: {t1+t2+t3:.2f}s")
print("\n✓ Ready for comparison with MEEP results")
print("="*80)
