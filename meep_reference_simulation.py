#!/usr/bin/env python3
"""
MEEP Reference Simulation - Run Locally
========================================

Creates a standardized brain imaging scenario and saves results.
Compare with CEEP reference simulation run on Colab.

Run locally:
  conda activate pymeep
  python meep_reference_simulation.py

Outputs: meep_results.json
"""

import os
import sys
import json
import numpy as np
import time
import meep as mp

print("="*80)
print(" MEEP REFERENCE SIMULATION")
print("="*80)

# STANDARD PARAMETERS (must match CEEP simulation exactly)
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

# Brain tissue properties (must match CEEP)
BRAIN_EPS_R = 50.0
BRAIN_SIGMA = 1.5  # S/m
BLOOD_EPS_R = 60.0  # Hemorrhage
BLOOD_SIGMA = 2.0  # S/m

print(f"  Brain: ε={BRAIN_EPS_R}, σ={BRAIN_SIGMA} S/m")
print(f"  Blood: ε={BLOOD_EPS_R}, σ={BLOOD_SIGMA} S/m")

# Constants
C_0 = 3e8  # Speed of light

# MEEP coordinate system
cell_size = mp.Vector3(NX * DX, NY * DX, 0)
resolution = 1.0 / DX

# Convert antenna position to MEEP coordinates (centered at origin)
ant_meep_x = (ANTENNA_X - NX/2) * DX
ant_meep_y = (ANTENNA_Y - NY/2) * DX

# PML layers
pml_layers = [mp.PML(thickness=CPML_THICKNESS*DX)]

# Timestep
dt_meep = DX / (C_0 * np.sqrt(2.0)) * 0.99

# ============================================================================
# Helper function to run MEEP simulation
# ============================================================================
def run_meep_scenario(geometry, name):
    """Run MEEP simulation for given geometry."""
    print(f"\n  [MEEP] Running...")

    # Source
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=FREQUENCY/C_0, fwidth=FREQUENCY/C_0/5),
            component=mp.Ez,
            center=mp.Vector3(ant_meep_x, ant_meep_y, 0)
        )
    ]

    # Create simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=resolution,
        force_complex_fields=False
    )

    # Monitor point (same as antenna)
    monitor_point = mp.Vector3(ant_meep_x, ant_meep_y, 0)

    t_start = time.time()

    # Record signal
    signal = []

    def record(sim):
        signal.append(sim.get_field_point(mp.Ez, monitor_point))

    # MEEP uses normalized units where c=1
    # Run for TOTAL_STEPS timesteps
    sim.run(mp.at_every(1, record), until=TOTAL_STEPS)

    t_elapsed = time.time() - t_start

    signal = np.array(signal)
    s_mag = np.abs(signal).max()

    print(f"  [MEEP] Runtime: {t_elapsed:.2f}s")
    print(f"  [MEEP] S-parameter: {s_mag:.3f}")
    print(f"  [MEEP] Signal length: {len(signal)}")

    return {
        's_parameter': float(s_mag),
        'runtime': float(t_elapsed),
        'signal_real': signal.real.tolist(),
        'signal_imag': signal.imag.tolist()
    }

# ============================================================================
# SCENARIO 1: Empty space (baseline)
# ============================================================================
print("\n" + "="*80)
print(" SCENARIO 1: Empty Space")
print("="*80)

geometry_empty = []
result1 = run_meep_scenario(geometry_empty, "Empty Space")

# ============================================================================
# SCENARIO 2: Brain tissue (uniform)
# ============================================================================
print("\n" + "="*80)
print(" SCENARIO 2: Brain Tissue (Uniform)")
print("="*80)

# Brain circle (20 pixels = 10mm radius)
geometry_brain = [
    mp.Cylinder(
        radius=20 * DX,
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(epsilon=BRAIN_EPS_R, D_conductivity=BRAIN_SIGMA)
    )
]

result2 = run_meep_scenario(geometry_brain, "Brain Tissue")

# ============================================================================
# SCENARIO 3: Brain with small hemorrhage
# ============================================================================
print("\n" + "="*80)
print(" SCENARIO 3: Brain + Small Hemorrhage (5mm)")
print("="*80)

# Brain circle + hemorrhage circle
# Hemorrhage offset: (8, 5) pixels from center
hem_x = 8 * DX
hem_y = 5 * DX

geometry_hem = [
    mp.Cylinder(
        radius=20 * DX,
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(epsilon=BRAIN_EPS_R, D_conductivity=BRAIN_SIGMA)
    ),
    mp.Cylinder(
        radius=10 * DX,  # 10 pixels = 5mm radius
        center=mp.Vector3(hem_x, hem_y, 0),
        material=mp.Medium(epsilon=BLOOD_EPS_R, D_conductivity=BLOOD_SIGMA)
    )
]

result3 = run_meep_scenario(geometry_hem, "Brain + Hemorrhage")

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
        'empty': result1,
        'brain': result2,
        'hemorrhage': result3
    },
    'solver': 'MEEP',
    'backend': 'CPU'
}

with open('meep_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print(" RESULTS SAVED")
print("="*80)
print(f"Saved to: meep_results.json")
print(f"\nSummary:")
print(f"  Empty space:  {result1['s_parameter']:.3f}")
print(f"  Brain tissue: {result2['s_parameter']:.3f}")
print(f"  Hemorrhage:   {result3['s_parameter']:.3f}")
print(f"\nTotal runtime: {result1['runtime']+result2['runtime']+result3['runtime']:.2f}s")
print("\n✓ Ready for comparison with CEEP results")
print("="*80)
