#!/usr/bin/env python3
"""
Far-Field Radar - Proper Parameters for Plane-Wave Beamforming
===============================================================

The previous simulations failed because:
1. Near-field effects at 1.5m range with 10 GHz
2. Far-field distance = 2 * D^2 / λ where D is array size
3. For 8-element array at λ/2 spacing: D = 3.5λ, far-field = 24.5λ

This version uses parameters ensuring far-field conditions:
- Higher frequency: 77 GHz (automotive radar band)
- Longer range: 5m (ensures far-field)
- Smaller grid (faster simulation)

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
print(" FAR-FIELD RADAR - 77 GHz at 5m Range")
print("="*70)
print()

set_backend('cupy')

# ============================================================
# Parameters (Far-Field Optimized)
# ============================================================

FREQUENCY = 77e9                    # 77 GHz (automotive radar)
WAVELENGTH = 3e8 / FREQUENCY        # 3.9 mm
TARGET_RANGE = 5.0                  # 5m (far-field!)
TARGET_ANGLE = 30.0                 # 30 degrees
TARGET_RADIUS = 0.05                # 5cm sphere

# Array parameters
NUM_HORIZ = 8
NUM_VERT = 4
NUM_ANTENNAS = NUM_HORIZ + NUM_VERT

# Check far-field condition
array_size = (NUM_HORIZ - 1) * (WAVELENGTH / 2)
far_field_distance = 2 * array_size**2 / WAVELENGTH
rayleigh_distance = array_size**2 / WAVELENGTH

print(f"FREQUENCY: {FREQUENCY/1e9:.1f} GHz")
print(f"WAVELENGTH: {WAVELENGTH*1000:.2f} mm")
print(f"TARGET: {TARGET_ANGLE}° at {TARGET_RANGE}m")
print()
print(f"FAR-FIELD CHECK:")
print(f"  Array size:         {array_size*1000:.1f} mm = {array_size/WAVELENGTH:.1f}λ")
print(f"  Rayleigh distance:  {rayleigh_distance:.2f} m")
print(f"  Far-field distance: {far_field_distance:.2f} m")
print(f"  Target range:       {TARGET_RANGE:.2f} m")

if TARGET_RANGE > far_field_distance:
    print(f"  ✓ FAR-FIELD (target > {far_field_distance:.1f}m)")
else:
    print(f"  ⚠ NEAR-FIELD (need > {far_field_distance:.1f}m)")
print()

# Grid parameters
GRID_RESOLUTION = 15                # Coarser for speed (still adequate at 77 GHz)
DX = WAVELENGTH / GRID_RESOLUTION   # 0.26 mm
DOMAIN_SIZE = 7.0                   # 7m domain (accommodate 5m range)
NX = NY = int(DOMAIN_SIZE / DX)

# Reduce grid if too large
MAX_SIZE = 3000
if NX > MAX_SIZE:
    NX = NY = MAX_SIZE
    DX = DOMAIN_SIZE / NX
    GRID_RESOLUTION = WAVELENGTH / DX
    print(f"Grid reduced to {NX}×{NY} (DX={DX*1000:.3f}mm, {GRID_RESOLUTION:.1f} pts/λ)")

C = 3e8
DT = 0.99 * DX / (C * np.sqrt(2))
ROUND_TRIP = 2 * TARGET_RANGE / C
TOTAL_STEPS = int(ROUND_TRIP / DT) + 500

print(f"GRID: {NX}×{NY}")
print(f"DX: {DX*1000:.3f} mm ({GRID_RESOLUTION:.1f} pts/λ)")
print(f"TIMESTEPS: {TOTAL_STEPS}")
print(f"ROUND-TRIP: {ROUND_TRIP*1e9:.1f} ns = sample {int(ROUND_TRIP/DT)}")
print()

mem_estimate = NUM_ANTENNAS * NX * NY * 3 * 8 / 1e9
print(f"MEMORY: {mem_estimate:.2f} GB")
if mem_estimate > 14:
    print("  ⚠ May exceed T4 GPU memory!")
    print("  Consider reducing NX/NY or array elements")
    exit(1)
print()

# ============================================================
# Create Array
# ============================================================

print("[1/5] Creating L-shaped array...")

center_x = NX // 2
center_y = NY // 4
spacing_grid = int((WAVELENGTH / 2) / DX)

positions = []
for i in range(NUM_HORIZ):
    x = center_x + int((i - (NUM_HORIZ-1)/2) * spacing_grid)
    y = center_y
    positions.append((x, y))

for i in range(1, NUM_VERT+1):
    x = center_x - int((NUM_HORIZ-1)/2) * spacing_grid
    y = center_y + int(i * spacing_grid)
    positions.append((x, y))

pos_array = np.array(positions)
pos_physical = pos_array * DX

print(f"  ✓ {NUM_ANTENNAS} antennas")
print(f"  Spacing: {(WAVELENGTH/2)*1000:.2f} mm (λ/2)")
print()

# ============================================================
# Place Target (FIXED INDEXING)
# ============================================================

print("[2/5] Placing target...")

ula_center_x = center_x * DX
ula_center_y = center_y * DX
angle_rad = np.deg2rad(TARGET_ANGLE)
target_x = ula_center_x + TARGET_RANGE * np.sin(angle_rad)
target_y = ula_center_y + TARGET_RANGE * np.cos(angle_rad)
ix_target = int(target_x / DX)
iy_target = int(target_y / DX)

print(f"  Array center: ({ula_center_x:.3f}, {ula_center_y:.3f}) m")
print(f"  Target: ({target_x:.3f}, {target_y:.3f}) m")
print(f"  Indices: ({ix_target}, {iy_target})")

if ix_target >= NX or iy_target >= NY or ix_target < 0 or iy_target < 0:
    print(f"  ✗ ERROR: Target outside grid!")
    print(f"    Need larger domain or smaller range")
    exit(1)

eps_grid = np.ones((NX, NY), dtype=np.float64)
radius_grid = int(TARGET_RADIUS / DX)

# FIXED: Correct X/Y indexing
x_idx, y_idx = np.ogrid[:NX, :NY]
target_mask = ((x_idx - ix_target)**2 + (y_idx - iy_target)**2 <= radius_grid**2)
eps_grid[target_mask] = 1000.0

# Verify
target_pixels = np.sum(eps_grid > 100)
if target_pixels > 0:
    target_indices = np.argwhere(eps_grid > 100)
    com = target_indices.mean(axis=0)
    print(f"  ✓ Target: {target_pixels} pixels at [{int(com[0])}, {int(com[1])}]")
    print(f"    Physical: ({com[0]*DX:.3f}, {com[1]*DX:.3f}) m")
else:
    print(f"  ✗ WARNING: Target too small ({radius_grid} pixels)!")
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
print(f"  ✓ Complete in {t_elapsed:.1f}s")
print(f"    Throughput: {throughput:.2f} GCell-steps/s")
print()

s_matrix_array = np.zeros((NUM_ANTENNAS, NUM_ANTENNAS, TOTAL_STEPS))
for tx in range(NUM_ANTENNAS):
    for rx in range(NUM_ANTENNAS):
        s_matrix_array[tx, rx, :] = to_numpy(s_matrix[tx][rx])

# ============================================================
# Time-Delay Beamforming
# ============================================================

print(f"[4/5] Beamforming...")

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

ax1 = plt.subplot(2, 3, 1)
extent = [0, NX*DX, 0, NY*DX]
im = ax1.imshow(eps_grid.T, origin='lower', cmap='viridis', vmin=1, vmax=100, extent=extent)
ax1.scatter(pos_physical[:, 0], pos_physical[:, 1], c='red', marker='v', s=80,
            edgecolors='white', linewidths=2, label='Array')
ax1.plot([target_x], [target_y], 'y*', markersize=20,
         markeredgecolor='white', markeredgewidth=2, label=f'Target')
ax1.plot([ref_x, target_x], [ref_y, target_y], 'y--', linewidth=1.5, alpha=0.6)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title(f'77 GHz Radar ({TARGET_RANGE}m Far-Field)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(im, ax=ax1, label='εᵣ')

ax2 = plt.subplot(2, 3, 2)
signal = monostatic_signals[0, :]
ax2.plot(t_ns, signal, linewidth=0.8)
ax2.axvspan(t_ns[sample_start], t_ns[sample_end-1], color='red', alpha=0.2)
ax2.axvline(reflection_sample * solver.dt * 1e9, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Amplitude')
ax2.set_title('Time Signal', fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
zoom_start = max(0, reflection_sample - 300)
zoom_end = min(TOTAL_STEPS, reflection_sample + 300)
ax3.plot(t_ns[zoom_start:zoom_end], signal[zoom_start:zoom_end], linewidth=1)
ax3.axvspan(t_ns[sample_start], t_ns[sample_end-1], color='red', alpha=0.2)
ax3.axvline(reflection_sample * solver.dt * 1e9, color='red', linestyle='--', linewidth=1.5)
ax3.set_xlabel('Time (ns)')
ax3.set_ylabel('Amplitude')
ax3.set_title('Zoomed', fontweight='bold')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
ax4.plot(angles, power_db, 'b-', linewidth=2.5)
ax4.axvline(TARGET_ANGLE, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
ax4.axvline(estimated_angle, color='green', linestyle=':', linewidth=3, alpha=0.9)
ax4.axhline(peak_power - 3, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax4.set_xlabel('Angle (degrees)')
ax4.set_ylabel('Power (dB)')
ax4.set_title(f'Beamforming (Error: {error:.1f}°)', fontweight='bold')
ax4.legend([f'BF Spectrum', f'Truth ({TARGET_ANGLE}°)', f'Est. ({estimated_angle:.1f}°)', '3dB'])
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-90, 90)
ax4.set_ylim(-40, 5)

ax5 = plt.subplot(2, 3, 5)
ax5.scatter(pos_physical[:NUM_HORIZ, 0], pos_physical[:NUM_HORIZ, 1],
            c='red', s=120, marker='v', label='Horizontal')
ax5.scatter(pos_physical[NUM_HORIZ:, 0], pos_physical[NUM_HORIZ:, 1],
            c='blue', s=120, marker='^', label='Vertical')
ax5.set_xlabel('X (m)')
ax5.set_ylabel('Y (m)')
ax5.set_title('L-Array', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.axis('equal')

ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

status = "✓ EXCELLENT" if error < 5 else ("✓ GOOD" if error < 10 else "⚠ NEEDS WORK")

summary = f"""
FAR-FIELD RADAR RESULTS
{'='*40}

PARAMETERS:
  Freq:      {FREQUENCY/1e9:.0f} GHz
  Range:     {TARGET_RANGE:.1f} m
  Array:     L-shaped ({NUM_ANTENNAS})

FAR-FIELD:
  Rayleigh:  {rayleigh_distance:.1f} m
  Far-field: {far_field_distance:.1f} m
  Target:    {TARGET_RANGE:.1f} m
  Status:    {'✓ FAR' if TARGET_RANGE > far_field_distance else '⚠ NEAR'}

PERFORMANCE:
  Time:      {t_elapsed:.1f} s
  Throughput: {throughput:.2f} GC/s

RESULTS:
  Truth:     {TARGET_ANGLE:.1f}°
  Estimated: {estimated_angle:.1f}°
  Error:     {error:.1f}°
  Beamwidth: {beamwidth:.1f}°

STATUS: {status}
{'='*40}
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('radar_far_field.png', dpi=150, bbox_inches='tight')
plt.show()

print("="*70)
print(f" RESULTS: {status}")
print("="*70)
print(f"  Ground truth: {TARGET_ANGLE:.1f}°")
print(f"  Estimated:    {estimated_angle:.1f}°")
print(f"  Error:        {error:.1f}°")
print("="*70)
print(f"\n✓ Saved: radar_far_field.png")
print("="*70)
