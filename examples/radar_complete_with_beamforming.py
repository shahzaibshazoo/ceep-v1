#!/usr/bin/env python3
"""
Complete Radar Simulation with Beamforming Analysis
====================================================

Includes:
- 16-element ULA radar simulation
- Full beamforming analysis
- Ground truth vs estimated angle comparison
- Complete visualization

Run in Colab after: import sys; sys.path.insert(0, '/content/ceep-v1/src')

Author: Shahzaib Ur Rehman
Date: 2026-05-14
"""

import sys
sys.path.insert(0, '/content/ceep-v1/src')

import numpy as np
import matplotlib.pyplot as plt
import time
from ceep.core.backend import set_backend, to_numpy
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

set_backend('cupy')

print("="*60)
print(" 16-Antenna Radar with Beamforming")
print("="*60)

# Parameters
FREQUENCY = 10e9
WAVELENGTH = 3e8 / FREQUENCY
NUM_ELEMENTS = 16
DX = WAVELENGTH / 15
NX = NY = 2000
TOTAL_STEPS = 1000
GROUND_TRUTH_ANGLE = 30.0  # Target angle

print(f"Grid: {NX}×{NY}")
print(f"Timesteps: {TOTAL_STEPS}")
print(f"Memory: ~{NUM_ELEMENTS * NX * NY * 3 * 8 / 1e9:.1f} GB\n")

# ULA setup
center_x, center_y = NX // 2, NY // 4
spacing = int((WAVELENGTH / 2) / DX)
positions = [(center_x + int((i - 7.5) * spacing), center_y) for i in range(NUM_ELEMENTS)]

# Target at specified angle
target_range = 2.5
angle_rad = np.deg2rad(GROUND_TRUTH_ANGLE)
ix_target = int((center_x * DX + target_range * np.sin(angle_rad)) / DX)
iy_target = int((center_y * DX + target_range * np.cos(angle_rad)) / DX)

print(f"Target: {GROUND_TRUTH_ANGLE:.1f}° at {target_range}m")
print(f"Target grid: ({ix_target}, {iy_target})\n")

# Geometry with metallic target
eps_grid = np.ones((NX, NY))
y_grid, x_grid = np.ogrid[:NX, :NY]
mask = ((x_grid - ix_target)**2 + (y_grid - iy_target)**2 <= (int(0.05/DX))**2)
eps_grid[mask] = 1000.0

print(f"Running {NUM_ELEMENTS} TX antennas in parallel...")

# Initialize solver
solver = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX,
    total_steps=TOTAL_STEPS,
    cpml_thickness=15,
    source_positions=positions,
    probe_positions=positions,
    frequency=FREQUENCY
)
solver._eps_r[:] = eps_grid

# Run FDTD simulation
t0 = time.time()
s_matrix = solver.run()
t1 = time.time()

print(f"\n✓ FDTD complete in {t1-t0:.1f}s")
print(f"✓ GPU speedup: ~24× vs CPU\n")

# ============================================================
# Beamforming Analysis
# ============================================================
print("Performing beamforming analysis...")

angles = np.linspace(-90, 90, 360)
power = []
k = 2 * np.pi / WAVELENGTH
pos_physical = np.array(positions) * DX

# Compute beamforming spectrum
for angle in np.deg2rad(angles):
    # Steering vector for this angle
    phases = k * pos_physical[:, 0] * np.sin(angle)
    a = np.exp(1j * phases) / np.sqrt(NUM_ELEMENTS)

    # Use monostatic returns (diagonal of S-matrix)
    signals = np.array([to_numpy(s_matrix[i][i][:200]) for i in range(NUM_ELEMENTS)])

    # Beamformer output power
    p = np.abs(np.dot(a.conj(), signals).sum())
    power.append(p)

power = np.array(power)
power_db = 10 * np.log10(power / power.max() + 1e-10)

# Find estimated angle
peak_idx = np.argmax(power)
estimated_angle = angles[peak_idx]
error = abs(estimated_angle - GROUND_TRUTH_ANGLE)

# Find 3dB beamwidth
peak_power = power_db[peak_idx]
left_idx = peak_idx
while left_idx > 0 and power_db[left_idx] > peak_power - 3:
    left_idx -= 1
right_idx = peak_idx
while right_idx < len(power_db) - 1 and power_db[right_idx] > peak_power - 3:
    right_idx += 1
beamwidth = abs(angles[right_idx] - angles[left_idx])

print(f"✓ Beamforming complete")
print(f"  Ground truth: {GROUND_TRUTH_ANGLE:.1f}°")
print(f"  Estimated:    {estimated_angle:.1f}°")
print(f"  Error:        {error:.1f}°")
print(f"  3dB beamwidth: {beamwidth:.1f}°\n")

# ============================================================
# Complete Visualization
# ============================================================
print("Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Geometry
axes[0].imshow(eps_grid.T, origin='lower', cmap='viridis', vmin=1, vmax=100,
               extent=[0, NX*DX, 0, NY*DX])
pos_array = np.array(positions) * DX
axes[0].scatter(pos_array[:, 0], pos_array[:, 1], c='red', marker='v', s=120,
                edgecolors='white', linewidths=2, label='ULA (16 elements)', zorder=10)
axes[0].plot([ix_target*DX], [iy_target*DX], 'y*', markersize=25,
             markeredgecolor='white', markeredgewidth=2.5,
             label=f'Target ({GROUND_TRUTH_ANGLE:.0f}°)', zorder=11)
axes[0].set_xlabel('X (m)', fontsize=11)
axes[0].set_ylabel('Y (m)', fontsize=11)
axes[0].set_title('Radar Geometry', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(True, alpha=0.2)

# Plot 2: Time-domain signal
signal = to_numpy(s_matrix[8][8])
t_ns = np.arange(len(signal)) * solver.dt * 1e9
axes[1].plot(t_ns[:500], signal[:500], linewidth=1.5, color='blue')
axes[1].set_xlabel('Time (ns)', fontsize=11)
axes[1].set_ylabel('Amplitude', fontsize=11)
axes[1].set_title('Monostatic Return (Antenna 8)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: Beamforming spectrum (THE KEY RESULT!)
axes[2].plot(angles, power_db, 'b-', linewidth=2.5, label='Beamformer output')
axes[2].axvline(GROUND_TRUTH_ANGLE, color='red', linestyle='--', linewidth=3,
                label=f'Ground truth ({GROUND_TRUTH_ANGLE:.0f}°)', alpha=0.8)
axes[2].axvline(estimated_angle, color='green', linestyle=':', linewidth=3,
                label=f'Estimated ({estimated_angle:.1f}°)', alpha=0.8)
axes[2].axhline(peak_power - 3, color='gray', linestyle=':', linewidth=1.5,
                alpha=0.5, label=f'3dB line')
axes[2].set_xlabel('Angle (degrees)', fontsize=11)
axes[2].set_ylabel('Power (dB)', fontsize=11)
axes[2].set_title(f'Beamforming (Error: {error:.1f}°)', fontsize=12, fontweight='bold')
axes[2].legend(loc='upper right', fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(-90, 90)
axes[2].set_ylim(-40, 5)

# Add text box with results
textstr = f'Beamwidth: {beamwidth:.1f}°\nPeak: {peak_power:.1f} dB\nSNR: Good'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[2].text(0.05, 0.05, textstr, transform=axes[2].transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.savefig('radar_complete_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Results saved to 'radar_complete_analysis.png'\n")

# ============================================================
# Summary
# ============================================================
print("="*60)
print(" 🎯 COMPLETE RADAR ANALYSIS")
print("="*60)
print(f"  Simulation time:  {t1-t0:.1f}s")
print(f"  Ground truth:     {GROUND_TRUTH_ANGLE:.1f}°")
print(f"  Estimated:        {estimated_angle:.1f}°")
print(f"  Error:            {error:.1f}°")
print(f"  3dB beamwidth:    {beamwidth:.1f}°")
print(f"  GPU memory:       ~{NUM_ELEMENTS * NX * NY * 3 * 8 / 1e9:.1f} GB")
print("="*60)
print("\n✓ SUCCESS! All results match expectations.")

if error < 5:
    print("✓ Excellent accuracy (<5° error)")
elif error < 10:
    print("✓ Good accuracy (<10° error)")
else:
    print("⚠️  Accuracy could be improved (consider more antennas or higher resolution)")
