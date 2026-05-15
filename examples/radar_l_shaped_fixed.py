#!/usr/bin/env python3
"""
L-Shaped Radar Array - FIXED Beamforming
=========================================

FIXES:
1. Proper 2D steering vector (uses BOTH x and y coordinates)
2. Correct phase calculation for L-shaped geometry
3. Azimuth-only scanning (elevation = 0)

The previous version only used X coordinates, which is wrong for
L-shaped arrays where elements have different Y positions too.

Author: Shahzaib Ur Rehman
Date: 2026-05-14
"""

import sys
sys.path.insert(0, '/content/ceep-v1/src')

import numpy as np
import matplotlib.pyplot as plt
import time
from ceep.core.backend import set_backend, to_numpy, print_backend_info
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D

print("="*70)
print(" L-SHAPED RADAR - FIXED BEAMFORMING")
print("="*70)
print("\nFIX: Proper 2D steering vector for L-shaped array\n")

set_backend('cupy')

# ============================================================
# Configuration
# ============================================================

FREQUENCY = 10e9
WAVELENGTH = 3e8 / FREQUENCY
TARGET_RANGE = 1.5
TARGET_ANGLE = 30.0
TARGET_RADIUS = 0.05

# Grid parameters
GRID_RESOLUTION = 20
DX = WAVELENGTH / GRID_RESOLUTION
DOMAIN_SIZE = 3.0
NX = NY = int(DOMAIN_SIZE / DX)

# Time parameters
C = 3e8
DT = 0.99 * DX / (C * np.sqrt(2))
ROUND_TRIP = 2 * TARGET_RANGE / C
REFLECTION_TIME = ROUND_TRIP + 5e-9
TOTAL_STEPS = int(REFLECTION_TIME / DT) + 500

# L-shaped array
NUM_HORIZ = 8
NUM_VERT = 4
NUM_ANTENNAS = NUM_HORIZ + NUM_VERT

print(f"CONFIGURATION:")
print(f"  Frequency:        {FREQUENCY/1e9:.1f} GHz")
print(f"  Wavelength:       {WAVELENGTH*1000:.1f} mm")
print(f"  Grid size:        {NX} × {NY}")
print(f"  Target:           {TARGET_ANGLE}° at {TARGET_RANGE} m")
print(f"  Array:            L-shaped ({NUM_HORIZ}H + {NUM_VERT}V)")
print(f"  Total timesteps:  {TOTAL_STEPS}")
print()

# ============================================================
# Create L-Shaped Array
# ============================================================

print("[1/5] Creating L-shaped antenna array...")

center_x = NX // 2
center_y = NY // 4
spacing_grid = int((WAVELENGTH / 2) / DX)

positions = []

# Horizontal arm (along X-axis)
for i in range(NUM_HORIZ):
    x = center_x + int((i - (NUM_HORIZ-1)/2) * spacing_grid)
    y = center_y
    positions.append((x, y))

# Vertical arm (along Y-axis)
for i in range(1, NUM_VERT+1):
    x = center_x - int((NUM_HORIZ-1)/2) * spacing_grid  # Align with leftmost
    y = center_y + int(i * spacing_grid)
    positions.append((x, y))

pos_array = np.array(positions)
pos_physical = pos_array * DX

print(f"  ✓ Created {len(positions)} antenna positions")
print(f"  ✓ Horizontal: {NUM_HORIZ} elements at Y = {center_y*DX:.2f} m")
print(f"  ✓ Vertical:   {NUM_VERT} elements at X = {(center_x - int((NUM_HORIZ-1)/2)*spacing_grid)*DX:.2f} m")
print()

# ============================================================
# Create Target
# ============================================================

print("[2/5] Placing target...")

ula_center_x = center_x * DX
ula_center_y = center_y * DX
angle_rad = np.deg2rad(TARGET_ANGLE)
target_x = ula_center_x + TARGET_RANGE * np.sin(angle_rad)
target_y = ula_center_y + TARGET_RANGE * np.cos(angle_rad)
ix_target = int(target_x / DX)
iy_target = int(target_y / DX)

eps_grid = np.ones((NX, NY), dtype=np.float64)
radius_grid = int(TARGET_RADIUS / DX)
x_idx, y_idx = np.ogrid[:NX, :NY]  # FIXED: x_idx for axis 0, y_idx for axis 1
target_mask = ((x_idx - ix_target)**2 + (y_idx - iy_target)**2 <= radius_grid**2)
eps_grid[target_mask] = 1000.0

print(f"  ✓ Target at {TARGET_ANGLE}° ({TARGET_RANGE} m)")
print(f"  ✓ Position: ({target_x:.2f}, {target_y:.2f}) m")
print()

# ============================================================
# Run FDTD
# ============================================================

print(f"[3/5] Running GPU-accelerated FDTD...\n")

solver = BatchedFDTD2D(
    nx=NX,
    ny=NY,
    dx=DX,
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
print(f"\n  ✓ FDTD complete!")
print(f"    Time:       {t_elapsed:.1f} s")
print(f"    Throughput: {throughput:.2f} GCell-steps/s")
print()

# Extract S-matrix
s_matrix_array = np.zeros((NUM_ANTENNAS, NUM_ANTENNAS, TOTAL_STEPS))
for tx in range(NUM_ANTENNAS):
    for rx in range(NUM_ANTENNAS):
        s_matrix_array[tx, rx, :] = to_numpy(s_matrix[tx][rx])

# ============================================================
# FIXED Beamforming for L-Shaped Array
# ============================================================

print(f"[4/5] Performing beamforming (FIXED for L-shaped array)...")

# Find reflection window
reflection_sample = int(ROUND_TRIP / solver.dt)
window_width = 150
sample_start = max(0, reflection_sample - window_width)
sample_end = min(TOTAL_STEPS, reflection_sample + window_width)

print(f"  Reflection at sample {reflection_sample}")
print(f"  Analysis window: [{sample_start}, {sample_end}]")

# Extract monostatic returns
monostatic_signals = s_matrix_array[np.arange(NUM_ANTENNAS), np.arange(NUM_ANTENNAS), :]
windowed_signals = monostatic_signals[:, sample_start:sample_end]

# Beamforming with PROPER 2D steering vector
angles = np.linspace(-90, 90, 360)
power = []

k = 2 * np.pi / WAVELENGTH

for angle in np.deg2rad(angles):
    # FIXED: Use both X and Y coordinates for 2D array
    # For azimuth angle θ in X-Y plane:
    # k_x = k * sin(θ)
    # k_y = k * cos(θ)
    k_x = k * np.sin(angle)
    k_y = k * np.cos(angle)

    # Phase delay for each antenna based on its (x, y) position
    phases = k_x * pos_physical[:, 0] + k_y * pos_physical[:, 1]
    a = np.exp(1j * phases) / np.sqrt(NUM_ANTENNAS)

    # Apply steering vector
    weighted = a[:, np.newaxis] * windowed_signals
    p = np.abs(weighted.sum())
    power.append(p)

power = np.array(power)
power_db = 10 * np.log10(power / power.max() + 1e-10)

# Find peak
peak_idx = np.argmax(power)
estimated_angle = angles[peak_idx]
error = abs(estimated_angle - TARGET_ANGLE)

# 3dB beamwidth
peak_power = power_db[peak_idx]
above_3db = power_db > (peak_power - 3)
if above_3db.sum() > 0:
    indices = np.where(above_3db)[0]
    beamwidth = angles[indices[-1]] - angles[indices[0]]
else:
    beamwidth = 0

print(f"  ✓ Beamforming complete")
print(f"    Ground truth:  {TARGET_ANGLE:.1f}°")
print(f"    Estimated:     {estimated_angle:.1f}°")
print(f"    Error:         {error:.1f}°")
print(f"    3dB beamwidth: {beamwidth:.1f}°")
print()

# ============================================================
# Visualization
# ============================================================

print(f"[5/5] Generating plots...\n")

fig = plt.figure(figsize=(20, 10))

t_ns = np.arange(TOTAL_STEPS) * solver.dt * 1e9

# Plot 1: Geometry
ax1 = plt.subplot(2, 3, 1)
im = ax1.imshow(eps_grid.T, origin='lower', cmap='viridis', vmin=1, vmax=100,
                extent=[0, NX*DX, 0, NY*DX])
ax1.scatter(pos_physical[:, 0], pos_physical[:, 1], c='red', marker='v', s=150,
            edgecolors='white', linewidths=2.5, label='L-array', zorder=10)
ax1.plot([target_x], [target_y], 'y*', markersize=30,
         markeredgecolor='white', markeredgewidth=3, label=f'Target ({TARGET_ANGLE}°)', zorder=11)
ax1.plot([ula_center_x, target_x], [ula_center_y, target_y],
         'y--', linewidth=2, alpha=0.6)
ax1.set_xlabel('X (m)', fontsize=12)
ax1.set_ylabel('Y (m)', fontsize=12)
ax1.set_title('L-Shaped Array Geometry', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.colorbar(im, ax=ax1, label='εᵣ')

# Plot 2: Time signal
ax2 = plt.subplot(2, 3, 2)
signal = monostatic_signals[0, :]
ax2.plot(t_ns, signal, linewidth=1, alpha=0.8)
ax2.axvspan(t_ns[sample_start], t_ns[sample_end-1], color='red', alpha=0.2,
            label='Analysis window')
ax2.axvline(reflection_sample * solver.dt * 1e9, color='red', linestyle='--',
            linewidth=2, label='Expected reflection')
ax2.set_xlabel('Time (ns)', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title('Time-Domain Signal', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Zoomed reflection
ax3 = plt.subplot(2, 3, 3)
zoom_margin = 300
zoom_start = max(0, reflection_sample - zoom_margin)
zoom_end = min(TOTAL_STEPS, reflection_sample + zoom_margin)
ax3.plot(t_ns[zoom_start:zoom_end], signal[zoom_start:zoom_end],
         linewidth=1.5, color='blue')
ax3.axvspan(t_ns[sample_start], t_ns[sample_end-1], color='red', alpha=0.2)
ax3.axvline(reflection_sample * solver.dt * 1e9, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Time (ns)', fontsize=12)
ax3.set_ylabel('Amplitude', fontsize=12)
ax3.set_title('Zoomed: Reflection Region', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Beamforming spectrum (THE KEY RESULT!)
ax4 = plt.subplot(2, 3, 4)
ax4.plot(angles, power_db, 'b-', linewidth=3, label='Beamformer')
ax4.axvline(TARGET_ANGLE, color='red', linestyle='--', linewidth=3,
            label=f'Ground truth ({TARGET_ANGLE}°)', alpha=0.8)
ax4.axvline(estimated_angle, color='green', linestyle=':', linewidth=3.5,
            label=f'Estimated ({estimated_angle:.1f}°)', alpha=0.9)
ax4.axhline(peak_power - 3, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax4.set_xlabel('Angle (degrees)', fontsize=12)
ax4.set_ylabel('Power (dB)', fontsize=12)
ax4.set_title(f'Beamforming Spectrum (Error: {error:.1f}°)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-90, 90)
ax4.set_ylim(-40, 5)

# Plot 5: Array pattern
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(pos_physical[:NUM_HORIZ, 0], pos_physical[:NUM_HORIZ, 1],
            c='red', s=200, marker='v', label='Horizontal arm')
ax5.scatter(pos_physical[NUM_HORIZ:, 0], pos_physical[NUM_HORIZ:, 1],
            c='blue', s=200, marker='^', label='Vertical arm')
for i, pos in enumerate(pos_physical):
    ax5.text(pos[0]+0.02, pos[1]+0.02, f'{i}', fontsize=9)
ax5.set_xlabel('X (m)', fontsize=12)
ax5.set_ylabel('Y (m)', fontsize=12)
ax5.set_title('L-Array Layout', fontsize=14, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)
ax5.axis('equal')

# Plot 6: Results summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

status = "✓ EXCELLENT" if error < 5 else ("✓ GOOD" if error < 10 else "⚠ FAIR")
color = 'green' if error < 5 else ('orange' if error < 10 else 'red')

summary = f"""
RADAR ANALYSIS RESULTS (FIXED)
{'='*50}

CONFIGURATION:
  • Frequency:      {FREQUENCY/1e9:.1f} GHz
  • Array type:     L-shaped ({NUM_ANTENNAS} elements)
  • Grid:           {NX}×{NY} ({DX*1000:.2f}mm)
  • Timesteps:      {TOTAL_STEPS}

TARGET:
  • Ground truth:   {TARGET_ANGLE:.1f}°
  • Range:          {TARGET_RANGE:.1f} m
  • Type:           Metallic sphere

PERFORMANCE:
  • FDTD time:      {t_elapsed:.1f} s
  • Throughput:     {throughput:.2f} GCell-steps/s

BEAMFORMING (FIXED):
  • Estimated:      {estimated_angle:.1f}°
  • Error:          {error:.1f}°
  • 3dB beamwidth:  {beamwidth:.1f}°
  • Peak power:     {peak_power:.1f} dB

ACCURACY: {status}
{'='*50}

FIX APPLIED:
  ✓ Proper 2D steering vector
  ✓ Uses both X and Y antenna coordinates
  ✓ Phase = k_x*x + k_y*y

Previous version only used X (wrong for L-array!)
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('radar_l_shaped_fixed.png', dpi=150, bbox_inches='tight')
plt.show()

print("="*70)
print(" 🎯 RADAR SIMULATION COMPLETE (FIXED)")
print("="*70)
print(f"  Ground truth:     {TARGET_ANGLE:.1f}°")
print(f"  Estimated:        {estimated_angle:.1f}°")
print(f"  Error:            {error:.1f}°")
print(f"  Status:           {status}")
print("="*70)
print(f"\n✓ Results saved to 'radar_l_shaped_fixed.png'")
print(f"✓ Simulation time: {t_elapsed:.1f} seconds")

if error < 5:
    print("\n🎉 EXCELLENT ACCURACY! L-shaped array with proper 2D beamforming!")
elif error < 10:
    print("\n✓ GOOD ACCURACY! Results within acceptable range.")
else:
    print("\n⚠️  Still higher error than expected.")
    print("   Possible reasons:")
    print("   - Target in near-field (plane wave assumption breaks down)")
    print("   - Need more antenna elements")
    print("   - Grid resolution effects")

print("="*70)
print("\nKEY FIX: Steering vector now uses BOTH x and y coordinates:")
print("  phases = k_x * x + k_y * y")
print("  where k_x = k*sin(θ), k_y = k*cos(θ)")
print("\nPrevious version only used X, which is wrong for L-shaped arrays!")
print("="*70)
