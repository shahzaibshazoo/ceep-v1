#!/usr/bin/env python3
"""
Corrected Radar Simulation - Time-Delay Beamforming
====================================================

Fixes applied to radar_smart_complete.py:
1. Uses both X and Y coordinates in steering vector (L-shaped array)
2. Uses time-delay beamforming (shift-and-sum) for real-valued signals
3. Angle convention matches target placement (0° = +Y direction)

For monostatic radar with time-domain signals, the correct approach is:
- For each candidate angle theta, compute expected round-trip delay per antenna
- Shift each antenna's signal by the negative of that delay (align signals)
- Sum the aligned signals; peak power occurs at true target angle

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

# ============================================================
# STEP 1: Configure Parameters (Smart Defaults)
# ============================================================

print("="*70)
print(" CORRECTED RADAR SIMULATION - Time-Delay Beamforming")
print("="*70)
print("\nInitializing...\n")

set_backend('cupy')

# Core parameters
FREQUENCY = 10e9                    # 10 GHz - good for imaging
WAVELENGTH = 3e8 / FREQUENCY        # 3 cm
TARGET_RANGE = 1.5                  # 1.5m
TARGET_ANGLE = 30.0                 # 30 degrees
TARGET_RADIUS = 0.05                # 5cm metallic sphere

# Grid parameters (optimized for T4 GPU)
GRID_RESOLUTION = 20                # 20 points per wavelength (good accuracy)
DX = WAVELENGTH / GRID_RESOLUTION   # 1.5mm grid spacing
DOMAIN_SIZE = 3.0                   # 3m x 3m domain
NX = NY = int(DOMAIN_SIZE / DX)     # ~2000 grid points

# Time parameters (calculated, not hardcoded!)
C = 3e8                             # Speed of light
DT = 0.99 * DX / (C * np.sqrt(2))   # CFL condition for 2D
ROUND_TRIP = 2 * TARGET_RANGE / C   # Time for signal to return
REFLECTION_TIME = ROUND_TRIP + 5e-9 # Add 5ns margin
TOTAL_STEPS = int(REFLECTION_TIME / DT) + 500  # Extra samples after reflection

# L-shaped array (breaks left-right ambiguity!)
NUM_HORIZ = 8                       # 8 antennas horizontal
NUM_VERT = 4                        # 4 antennas vertical
NUM_ANTENNAS = NUM_HORIZ + NUM_VERT # Total: 12

print(f"CONFIGURATION:")
print(f"  Frequency:        {FREQUENCY/1e9:.1f} GHz")
print(f"  Wavelength:       {WAVELENGTH*1000:.1f} mm")
print(f"  Grid size:        {NX} x {NY}")
print(f"  Grid spacing:     {DX*1000:.2f} mm")
print(f"  Domain:           {DOMAIN_SIZE} x {DOMAIN_SIZE} m")
print(f"  Resolution:       {GRID_RESOLUTION} pts/lambda")
print(f"\nTARGET:")
print(f"  Range:            {TARGET_RANGE} m")
print(f"  Angle:            {TARGET_ANGLE} deg")
print(f"  Radius:           {TARGET_RADIUS*100} cm")
print(f"\nTIMING:")
print(f"  Time step:        {DT*1e12:.2f} ps")
print(f"  Round-trip:       {ROUND_TRIP*1e9:.1f} ns")
print(f"  Total timesteps:  {TOTAL_STEPS}")
print(f"  Reflection at:    sample {int(ROUND_TRIP/DT)}")
print(f"\nARRAY:")
print(f"  Type:             L-shaped (breaks left-right ambiguity)")
print(f"  Elements:         {NUM_ANTENNAS} ({NUM_HORIZ} horiz + {NUM_VERT} vert)")
print(f"  Spacing:          lambda/2 = {WAVELENGTH*500:.1f} mm")

# Memory estimate
mem_estimate = NUM_ANTENNAS * NX * NY * 3 * 8 / 1e9
print(f"\nMEMORY:")
print(f"  Estimated GPU:    {mem_estimate:.2f} GB")
print(f"  T4 available:     15.6 GB")
if mem_estimate > 14:
    print("  WARNING: Memory might be tight!")
else:
    print("  OK: Memory within limits")

print(f"\nESTIMATED TIME:   {TOTAL_STEPS/1000*40:.0f} seconds\n")
print("="*70)

# ============================================================
# STEP 2: Create L-Shaped Array Geometry
# ============================================================

print("\n[1/5] Creating L-shaped antenna array...")

center_x = NX // 2
center_y = NY // 4
spacing_grid = int((WAVELENGTH / 2) / DX)

positions = []

# Horizontal arm (along X-axis)
for i in range(NUM_HORIZ):
    x = center_x + int((i - (NUM_HORIZ-1)/2) * spacing_grid)
    y = center_y
    positions.append((x, y))

# Vertical arm (along Y-axis, starting from left end of horizontal)
for i in range(1, NUM_VERT+1):  # Start from 1 to avoid duplicate at center
    x = center_x - int((NUM_HORIZ-1)/2) * spacing_grid  # Align with leftmost horizontal
    y = center_y + int(i * spacing_grid)
    positions.append((x, y))

pos_array = np.array(positions)
print(f"  Created {len(positions)} antenna positions")
print(f"  Horizontal: {NUM_HORIZ} elements")
print(f"  Vertical:   {NUM_VERT} elements")
print(f"  L-shape breaks symmetry for unambiguous angle detection")

# ============================================================
# STEP 3: Create Target Geometry
# ============================================================

print("\n[2/5] Placing target...")

# Target position from array center
# Convention: angle=0 means straight ahead (+Y direction)
#             angle=30 means offset to the right (+X)
ula_center_x = center_x * DX
ula_center_y = center_y * DX
angle_rad = np.deg2rad(TARGET_ANGLE)
target_x = ula_center_x + TARGET_RANGE * np.sin(angle_rad)
target_y = ula_center_y + TARGET_RANGE * np.cos(angle_rad)
ix_target = int(target_x / DX)
iy_target = int(target_y / DX)

print(f"  ULA center:  ({ula_center_x:.2f}, {ula_center_y:.2f}) m")
print(f"  Target:      ({target_x:.2f}, {target_y:.2f}) m")
print(f"  Grid index:  ({ix_target}, {iy_target})")

# Create geometry
eps_grid = np.ones((NX, NY), dtype=np.float64)

# Metallic target
radius_grid = int(TARGET_RADIUS / DX)
x_idx, y_idx = np.ogrid[:NX, :NY]  # FIXED: x_idx for axis 0, y_idx for axis 1
target_mask = ((x_idx - ix_target)**2 + (y_idx - iy_target)**2 <= radius_grid**2)
eps_grid[target_mask] = 1000.0  # Metal

print(f"  Target placed at {TARGET_ANGLE} deg ({TARGET_RANGE}m range)")
print(f"  Target radius: {radius_grid} grid points ({TARGET_RADIUS*100:.1f} cm)")

# ============================================================
# STEP 4: Run Batched FDTD Simulation
# ============================================================

print(f"\n[3/5] Running GPU-accelerated FDTD simulation...")
print(f"  Simulating {NUM_ANTENNAS} TX antennas in parallel...\n")

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
print(f"\n  FDTD complete!")
print(f"    Time:       {t_elapsed:.1f} seconds")
print(f"    Throughput: {throughput:.2f} GCell-steps/s")

# Extract data
s_matrix_array = np.zeros((NUM_ANTENNAS, NUM_ANTENNAS, TOTAL_STEPS))
for tx in range(NUM_ANTENNAS):
    for rx in range(NUM_ANTENNAS):
        s_matrix_array[tx, rx, :] = to_numpy(s_matrix[tx][rx])

# ============================================================
# STEP 5: Time-Delay Beamforming (CORRECTED)
# ============================================================

print(f"\n[4/5] Performing TIME-DELAY beamforming...")

# --- Extract monostatic returns ---
# In monostatic mode, each antenna TX and RX to itself
monostatic_signals = s_matrix_array[np.arange(NUM_ANTENNAS), np.arange(NUM_ANTENNAS), :]

# --- Determine the reflection window ---
reflection_sample = int(ROUND_TRIP / solver.dt)
window_width = 200  # Slightly larger window for delay shifting
sample_start = max(0, reflection_sample - window_width)
sample_end = min(TOTAL_STEPS, reflection_sample + window_width)

print(f"  Reflection expected at sample {reflection_sample}")
print(f"  Analysis window: [{sample_start}, {sample_end}] ({sample_end-sample_start} samples)")

# Safety check
if sample_end <= sample_start:
    print("\n  ERROR: Reflection window invalid!")
    print(f"   Reflection at {reflection_sample} but only {TOTAL_STEPS} steps simulated")
    exit(1)

windowed_signals = monostatic_signals[:, sample_start:sample_end]
n_samples = windowed_signals.shape[1]

# --- Physical positions of antennas ---
pos_physical = pos_array * DX  # Convert grid indices to meters

# Reference point: array centroid
ref_x = pos_physical[:, 0].mean()
ref_y = pos_physical[:, 1].mean()

print(f"  Array centroid: ({ref_x:.4f}, {ref_y:.4f}) m")
print(f"  Using time-delay (shift-and-sum) beamforming")
print(f"  Angle convention: 0 deg = +Y (broadside), positive = +X")

# --- Time-delay beamforming ---
# For each candidate angle theta:
#   - The unit vector toward the target is: k_hat = (sin(theta), cos(theta))
#     (matching the target placement convention)
#   - For far-field approximation, the extra path for antenna i relative to
#     the reference is: delta_d_i = (pos_i - ref) . k_hat
#   - Round-trip delay difference: delta_tau_i = 2 * delta_d_i / c
#   - We shift signal_i BACK by delta_tau_i to align all signals
#   - Then sum: peak when theta matches actual target angle

angles = np.linspace(-90, 90, 361)
power = np.zeros(len(angles))

dt_sim = solver.dt  # Simulation time step

for idx, theta_deg in enumerate(angles):
    theta = np.deg2rad(theta_deg)

    # Unit vector toward candidate target direction
    # Convention: angle measured from +Y axis toward +X axis
    k_hat_x = np.sin(theta)
    k_hat_y = np.cos(theta)

    # Compute differential path length for each antenna relative to centroid
    # delta_d_i = (x_i - ref_x) * sin(theta) + (y_i - ref_y) * cos(theta)
    delta_d = (pos_physical[:, 0] - ref_x) * k_hat_x + \
              (pos_physical[:, 1] - ref_y) * k_hat_y

    # Round-trip delay difference in samples
    # Positive delta_d means antenna is displaced toward target -> shorter path
    # -> echo arrives EARLIER by 2*delta_d/c
    # To align: DELAY the early signal (shift right = subtract from read index)
    delay_samples = 2.0 * delta_d / (C * dt_sim)

    # Shift-and-sum: apply compensating delays via interpolation
    aligned_sum = np.zeros(n_samples)
    sample_indices = np.arange(n_samples, dtype=np.float64)

    for ant in range(NUM_ANTENNAS):
        # To delay signal by delay_samples[ant]:
        #   output[n] = input[n - delay]
        # In interp terms: read from (sample_indices - delay)
        read_indices = sample_indices - delay_samples[ant]

        # Linear interpolation; zero-pad outside valid range
        shifted_sig = np.interp(read_indices, sample_indices,
                                windowed_signals[ant, :],
                                left=0.0, right=0.0)
        aligned_sum += shifted_sig

    # Power is sum of squared aligned signal (energy)
    power[idx] = np.sum(aligned_sum**2)

# Normalize to dB
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

print(f"\n  Beamforming complete (time-delay method)")
print(f"    Ground truth:  {TARGET_ANGLE:.1f} deg")
print(f"    Estimated:     {estimated_angle:.1f} deg")
print(f"    Error:         {error:.1f} deg")
print(f"    3dB beamwidth: {beamwidth:.1f} deg")
print(f"    Peak power:    {peak_power:.1f} dB")

# ============================================================
# STEP 6: Comprehensive Visualization
# ============================================================

print(f"\n[5/5] Generating plots...\n")

fig = plt.figure(figsize=(20, 10))

t_ns = np.arange(TOTAL_STEPS) * solver.dt * 1e9

# Plot 1: Geometry with L-shaped array
ax1 = plt.subplot(2, 3, 1)
im = ax1.imshow(eps_grid.T, origin='lower', cmap='viridis', vmin=1, vmax=100,
                extent=[0, NX*DX, 0, NY*DX])
ax1.scatter(pos_physical[:, 0], pos_physical[:, 1], c='red', marker='v', s=150,
            edgecolors='white', linewidths=2.5, label='L-array', zorder=10)
ax1.plot([target_x], [target_y], 'y*', markersize=30,
         markeredgecolor='white', markeredgewidth=3, label=f'Target ({TARGET_ANGLE} deg)', zorder=11)
ax1.plot([ref_x, target_x], [ref_y, target_y],
         'y--', linewidth=2, alpha=0.6)
ax1.set_xlabel('X (m)', fontsize=12)
ax1.set_ylabel('Y (m)', fontsize=12)
ax1.set_title('L-Shaped Array Geometry', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.colorbar(im, ax=ax1, label='eps_r')

# Plot 2: Full time signal
ax2 = plt.subplot(2, 3, 2)
signal = monostatic_signals[0, :]  # First antenna
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

# Plot 4: Beamforming spectrum
ax4 = plt.subplot(2, 3, 4)
ax4.plot(angles, power_db, 'b-', linewidth=3, label='Time-Delay BF')
ax4.axvline(TARGET_ANGLE, color='red', linestyle='--', linewidth=3,
            label=f'Ground truth ({TARGET_ANGLE} deg)', alpha=0.8)
ax4.axvline(estimated_angle, color='green', linestyle=':', linewidth=3.5,
            label=f'Estimated ({estimated_angle:.1f} deg)', alpha=0.9)
ax4.axhline(peak_power - 3, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax4.set_xlabel('Angle (degrees)', fontsize=12)
ax4.set_ylabel('Power (dB)', fontsize=12)
ax4.set_title(f'Time-Delay Beamforming (Error: {error:.1f} deg)', fontsize=14, fontweight='bold')
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

status = "EXCELLENT" if error < 5 else ("GOOD" if error < 10 else "FAIR")
color = 'green' if error < 5 else ('orange' if error < 10 else 'red')

summary = f"""
RADAR ANALYSIS RESULTS (TIME-DELAY BF)
{'='*50}

CONFIGURATION:
  Frequency:      {FREQUENCY/1e9:.1f} GHz
  Array type:     L-shaped ({NUM_ANTENNAS} elements)
  Grid:           {NX}x{NY} ({DX*1000:.2f}mm)
  Timesteps:      {TOTAL_STEPS}

TARGET:
  Ground truth:   {TARGET_ANGLE:.1f} deg
  Range:          {TARGET_RANGE:.1f} m
  Type:           Metallic sphere

PERFORMANCE:
  FDTD time:      {t_elapsed:.1f} s
  Throughput:     {throughput:.2f} GCell-steps/s
  GPU memory:     {mem_estimate:.2f} GB

BEAMFORMING (TIME-DELAY):
  Estimated:      {estimated_angle:.1f} deg
  Error:          {error:.1f} deg
  3dB beamwidth:  {beamwidth:.1f} deg
  Peak power:     {peak_power:.1f} dB

ACCURACY: {status}
{'='*50}

FIXES APPLIED:
  [1] Both X and Y coords used in steering
  [2] Time-delay beamforming (shift-and-sum)
  [3] Angle convention matches target placement
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('radar_corrected_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("="*70)
print(" RADAR SIMULATION COMPLETE (CORRECTED)")
print("="*70)
print(f"  Ground truth:     {TARGET_ANGLE:.1f} deg")
print(f"  Estimated:        {estimated_angle:.1f} deg")
print(f"  Error:            {error:.1f} deg")
print(f"  Status:           {status}")
print("="*70)
print(f"\n  Results saved to 'radar_corrected_results.png'")
print(f"  Simulation time: {t_elapsed:.1f} seconds")

if error < 5:
    print("\n  EXCELLENT ACCURACY! Time-delay beamforming working perfectly!")
elif error < 10:
    print("\n  GOOD ACCURACY! Results within acceptable range.")
else:
    print("\n  Higher error than expected - investigate signal quality")

print("="*70)
