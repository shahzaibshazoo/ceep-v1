#!/usr/bin/env python3
"""
Working Radar Simulation - All Bugs Fixed
==========================================

This script fixes ALL identified issues:
1. Target placement X/Y swap bug (critical!)
2. Time-delay beamforming for real-valued signals
3. L-shaped array with proper 2D steering

The fundamental bug was in target placement:
  WRONG: y_idx, x_idx = np.ogrid[:NX, :NY]; mask = ((x_idx - ix)**2 + (y_idx - iy)**2)
         This places target at eps_grid[iy, ix] but solver uses [x, y] indexing!
  RIGHT: x_idx, y_idx = np.ogrid[:NX, :NY]; mask = ((x_idx - ix)**2 + (y_idx - iy)**2)

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

print("="*70)
print(" WORKING RADAR - ALL BUGS FIXED")
print("="*70)
print("\nKey fix: Target placement X/Y swap corrected!\n")

set_backend('cupy')

# ============================================================
# Configuration
# ============================================================

FREQUENCY = 10e9                    # 10 GHz
WAVELENGTH = 3e8 / FREQUENCY        # 3 cm
TARGET_RANGE = 1.5                  # 1.5m
TARGET_ANGLE = 30.0                 # 30 degrees
TARGET_RADIUS = 0.05                # 5cm sphere

GRID_RESOLUTION = 20
DX = WAVELENGTH / GRID_RESOLUTION
DOMAIN_SIZE = 3.0
NX = NY = int(DOMAIN_SIZE / DX)

C = 3e8
DT = 0.99 * DX / (C * np.sqrt(2))
ROUND_TRIP = 2 * TARGET_RANGE / C
TOTAL_STEPS = int(ROUND_TRIP / DT) + 500

# L-shaped array
NUM_HORIZ = 8
NUM_VERT = 4
NUM_ANTENNAS = NUM_HORIZ + NUM_VERT

print(f"Grid: {NX}×{NY}, Resolution: {GRID_RESOLUTION} pts/λ")
print(f"Target: {TARGET_ANGLE}° at {TARGET_RANGE}m")
print(f"Array: L-shaped ({NUM_HORIZ}H + {NUM_VERT}V = {NUM_ANTENNAS} total)")
print(f"Timesteps: {TOTAL_STEPS}")
print()

# ============================================================
# Create Array
# ============================================================

print("[1/5] Creating L-shaped array...")

center_x = NX // 2
center_y = NY // 4
spacing_grid = int((WAVELENGTH / 2) / DX)

positions = []

# Horizontal arm
for i in range(NUM_HORIZ):
    x = center_x + int((i - (NUM_HORIZ-1)/2) * spacing_grid)
    y = center_y
    positions.append((x, y))

# Vertical arm
for i in range(1, NUM_VERT+1):
    x = center_x - int((NUM_HORIZ-1)/2) * spacing_grid
    y = center_y + int(i * spacing_grid)
    positions.append((x, y))

pos_array = np.array(positions)
pos_physical = pos_array * DX

print(f"  ✓ {NUM_ANTENNAS} antennas created")
print()

# ============================================================
# Place Target (FIXED!)
# ============================================================

print("[2/5] Placing target (FIXED X/Y indexing)...")

ula_center_x = center_x * DX
ula_center_y = center_y * DX
angle_rad = np.deg2rad(TARGET_ANGLE)
target_x = ula_center_x + TARGET_RANGE * np.sin(angle_rad)
target_y = ula_center_y + TARGET_RANGE * np.cos(angle_rad)
ix_target = int(target_x / DX)
iy_target = int(target_y / DX)

print(f"  Array center: ({ula_center_x:.3f}, {ula_center_y:.3f}) m")
print(f"  Target: ({target_x:.3f}, {target_y:.3f}) m")
print(f"  Target indices: ({ix_target}, {iy_target})")

eps_grid = np.ones((NX, NY), dtype=np.float64)
radius_grid = int(TARGET_RADIUS / DX)

# CRITICAL FIX: Correct X/Y indexing
# Solver uses eps_r[x, y] where x is axis 0, y is axis 1
# So target should be centered at [ix_target, iy_target]
x_idx, y_idx = np.ogrid[:NX, :NY]  # x_idx varies axis 0, y_idx varies axis 1
target_mask = ((x_idx - ix_target)**2 + (y_idx - iy_target)**2 <= radius_grid**2)
eps_grid[target_mask] = 1000.0

# Verify placement
target_indices = np.argwhere(eps_grid > 100)
if len(target_indices) > 0:
    center_of_mass = target_indices.mean(axis=0)
    print(f"  ✓ Target placed at grid [{int(center_of_mass[0])}, {int(center_of_mass[1])}]")
    print(f"    Physical: ({center_of_mass[0]*DX:.3f}, {center_of_mass[1]*DX:.3f}) m")
else:
    print(f"  ✗ WARNING: No target pixels found!")
print()

# ============================================================
# Run FDTD
# ============================================================

print(f"[3/5] Running FDTD...")

solver = BatchedFDTD2D(
    nx=NX, ny=NY, dx=DX,
    total_steps=TOTAL_STEPS,
    cpml_thickness=15,
    source_positions=positions,
    probe_positions=positions,
    frequency=FREQUENCY
)
solver._eps_r[:] = eps_grid

t_start = time.time()
s_matrix = solver.run()
t_elapsed = time.time() - t_start

throughput = NX * NY * TOTAL_STEPS * NUM_ANTENNAS / (t_elapsed * 1e9)
print(f"  ✓ Complete in {t_elapsed:.1f}s ({throughput:.2f} GCell-steps/s)")
print()

# Extract data
s_matrix_array = np.zeros((NUM_ANTENNAS, NUM_ANTENNAS, TOTAL_STEPS))
for tx in range(NUM_ANTENNAS):
    for rx in range(NUM_ANTENNAS):
        s_matrix_array[tx, rx, :] = to_numpy(s_matrix[tx][rx])

# ============================================================
# Time-Delay Beamforming
# ============================================================

print(f"[4/5] Time-delay beamforming...")

monostatic_signals = s_matrix_array[np.arange(NUM_ANTENNAS), np.arange(NUM_ANTENNAS), :]

reflection_sample = int(ROUND_TRIP / solver.dt)
window_width = 200
sample_start = max(0, reflection_sample - window_width)
sample_end = min(TOTAL_STEPS, reflection_sample + window_width)

windowed_signals = monostatic_signals[:, sample_start:sample_end]
n_samples = windowed_signals.shape[1]

ref_x = pos_physical[:, 0].mean()
ref_y = pos_physical[:, 1].mean()

angles = np.linspace(-90, 90, 361)
power = np.zeros(len(angles))
dt_sim = solver.dt

for idx, theta_deg in enumerate(angles):
    theta = np.deg2rad(theta_deg)

    k_hat_x = np.sin(theta)
    k_hat_y = np.cos(theta)

    delta_d = (pos_physical[:, 0] - ref_x) * k_hat_x + \
              (pos_physical[:, 1] - ref_y) * k_hat_y

    delay_samples = 2.0 * delta_d / (C * dt_sim)

    aligned_sum = np.zeros(n_samples)
    sample_indices = np.arange(n_samples, dtype=np.float64)

    for ant in range(NUM_ANTENNAS):
        read_indices = sample_indices - delay_samples[ant]
        shifted_sig = np.interp(read_indices, sample_indices,
                                windowed_signals[ant, :],
                                left=0.0, right=0.0)
        aligned_sum += shifted_sig

    power[idx] = np.sum(aligned_sum**2)

power_db = 10 * np.log10(power / power.max() + 1e-10)

peak_idx = np.argmax(power)
estimated_angle = angles[peak_idx]
error = abs(estimated_angle - TARGET_ANGLE)

peak_power = power_db[peak_idx]
above_3db = power_db > (peak_power - 3)
if above_3db.sum() > 0:
    indices = np.where(above_3db)[0]
    beamwidth = angles[indices[-1]] - angles[indices[0]]
else:
    beamwidth = 0

print(f"  Ground truth:  {TARGET_ANGLE:.1f}°")
print(f"  Estimated:     {estimated_angle:.1f}°")
print(f"  Error:         {error:.1f}°")
print(f"  3dB beamwidth: {beamwidth:.1f}°")
print()

# ============================================================
# Visualization
# ============================================================

print(f"[5/5] Plotting...")

fig = plt.figure(figsize=(18, 10))
t_ns = np.arange(TOTAL_STEPS) * solver.dt * 1e9

# Plot 1: Geometry
ax1 = plt.subplot(2, 3, 1)
im = ax1.imshow(eps_grid.T, origin='lower', cmap='viridis', vmin=1, vmax=100,
                extent=[0, NX*DX, 0, NY*DX])
ax1.scatter(pos_physical[:, 0], pos_physical[:, 1], c='red', marker='v', s=100,
            edgecolors='white', linewidths=2, label='L-array', zorder=10)
ax1.plot([target_x], [target_y], 'y*', markersize=25,
         markeredgecolor='white', markeredgewidth=2, label=f'Target ({TARGET_ANGLE}°)', zorder=11)
ax1.plot([ref_x, target_x], [ref_y, target_y], 'y--', linewidth=1.5, alpha=0.6)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Geometry (X/Y Fixed!)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(im, ax=ax1, label='εᵣ')

# Plot 2: Time signal
ax2 = plt.subplot(2, 3, 2)
signal = monostatic_signals[0, :]
ax2.plot(t_ns, signal, linewidth=0.8)
ax2.axvspan(t_ns[sample_start], t_ns[sample_end-1], color='red', alpha=0.2)
ax2.axvline(reflection_sample * solver.dt * 1e9, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Amplitude')
ax2.set_title('Time Signal', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Zoomed
ax3 = plt.subplot(2, 3, 3)
zoom_start = max(0, reflection_sample - 300)
zoom_end = min(TOTAL_STEPS, reflection_sample + 300)
ax3.plot(t_ns[zoom_start:zoom_end], signal[zoom_start:zoom_end], linewidth=1)
ax3.axvspan(t_ns[sample_start], t_ns[sample_end-1], color='red', alpha=0.2)
ax3.axvline(reflection_sample * solver.dt * 1e9, color='red', linestyle='--', linewidth=1.5)
ax3.set_xlabel('Time (ns)')
ax3.set_ylabel('Amplitude')
ax3.set_title('Zoomed Reflection', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Beamforming
ax4 = plt.subplot(2, 3, 4)
ax4.plot(angles, power_db, 'b-', linewidth=2.5, label='Time-Delay BF')
ax4.axvline(TARGET_ANGLE, color='red', linestyle='--', linewidth=2.5,
            label=f'Truth ({TARGET_ANGLE}°)', alpha=0.8)
ax4.axvline(estimated_angle, color='green', linestyle=':', linewidth=3,
            label=f'Est. ({estimated_angle:.1f}°)', alpha=0.9)
ax4.axhline(peak_power - 3, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax4.set_xlabel('Angle (degrees)')
ax4.set_ylabel('Power (dB)')
ax4.set_title(f'Beamforming (Error: {error:.1f}°)', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-90, 90)
ax4.set_ylim(-40, 5)

# Plot 5: Array layout
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(pos_physical[:NUM_HORIZ, 0], pos_physical[:NUM_HORIZ, 1],
            c='red', s=150, marker='v', label='Horizontal')
ax5.scatter(pos_physical[NUM_HORIZ:, 0], pos_physical[NUM_HORIZ:, 1],
            c='blue', s=150, marker='^', label='Vertical')
ax5.set_xlabel('X (m)')
ax5.set_ylabel('Y (m)')
ax5.set_title('L-Array Layout', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.axis('equal')

# Plot 6: Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

status = "✓ EXCELLENT" if error < 5 else ("✓ GOOD" if error < 10 else "⚠ FAIR")

summary = f"""
WORKING RADAR - ALL FIXES
{'='*45}

CRITICAL FIX:
  Target placement X/Y swap corrected!
  OLD: y_idx, x_idx = ogrid (WRONG)
  NEW: x_idx, y_idx = ogrid (CORRECT)

CONFIGURATION:
  Frequency:    {FREQUENCY/1e9:.1f} GHz
  Array:        L-shaped ({NUM_ANTENNAS} elem)
  Grid:         {NX}×{NY}
  Timesteps:    {TOTAL_STEPS}

TARGET:
  Ground truth: {TARGET_ANGLE:.1f}°
  Range:        {TARGET_RANGE:.1f} m

PERFORMANCE:
  Time:         {t_elapsed:.1f} s
  Throughput:   {throughput:.2f} GCell-steps/s

RESULTS:
  Estimated:    {estimated_angle:.1f}°
  Error:        {error:.1f}°
  Beamwidth:    {beamwidth:.1f}°

STATUS: {status}
{'='*45}
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('radar_working.png', dpi=150, bbox_inches='tight')
plt.show()

print("="*70)
print(" SUCCESS!")
print("="*70)
print(f"  Ground truth: {TARGET_ANGLE:.1f}°")
print(f"  Estimated:    {estimated_angle:.1f}°")
print(f"  Error:        {error:.1f}°")
print("="*70)

if error < 5:
    print("\n🎉 EXCELLENT! Target placement fix worked!")
elif error < 10:
    print("\n✓ GOOD! Results within acceptable range.")
else:
    print("\n⚠ Still needs investigation")

print("\n✓ Saved: radar_working.png")
print("="*70)
