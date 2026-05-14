# Cell 11: Batched 16-Element Antenna Array — TRUE GPU Parallelism
# =================================================================
# WHY THE PREVIOUS APPROACH WAS SLOW:
#   Sequential: 16 separate simulations × 300×300 = 90K cells per kernel
#   → GPU cores (2560 on T4) mostly idle, kernel launch overhead dominates
#
# THIS APPROACH:
#   Batched: 1 simulation with shape (16, 300, 300) = 1.44M cells per kernel
#   → Every CUDA core has work to do simultaneously
#   → Expected speedup: 10-20x over sequential GPU, 3-5x over CPU
#
# Think of it like: instead of launching 16 rockets one at a time,
# we strap all 16 payloads to one rocket and launch once.

import time
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D

# ============================================================
# Setup: 16-element circular array around a head phantom
# ============================================================

nx, ny = 300, 300
dx = 1e-3  # 1 mm
center = (nx // 2, ny // 2)

# Array configuration
num_antennas = 16
array_radius_cells = 100  # 100 cells = 10 cm

# Compute antenna positions on circle
positions = []
for i in range(num_antennas):
    angle = 2 * np.pi * i / num_antennas
    x = int(center[0] + array_radius_cells * np.cos(angle))
    y = int(center[1] + array_radius_cells * np.sin(angle))
    positions.append((x, y))

frequency = 1.0e9  # 1 GHz
total_steps = 400

print("=" * 60)
print("  BATCHED GPU FDTD: 16 Transmissions in Parallel")
print("=" * 60)
print(f"  Grid: {nx}x{ny} @ {dx*1e3:.1f} mm")
print(f"  Batch size: {num_antennas} (one per TX antenna)")
print(f"  Effective cells per kernel: {num_antennas * nx * ny / 1e6:.2f}M")
print(f"  Steps: {total_steps}")
print(f"  Frequency: {frequency/1e9:.1f} GHz")
print(f"  Array radius: {array_radius_cells * dx * 1e3:.0f} mm")

# ============================================================
# Create batched solver
# ============================================================

solver = BatchedFDTD2D(
    nx=nx, ny=ny, dx=dx,
    total_steps=total_steps,
    cpml_thickness=10,
    source_positions=positions,  # 16 TX antennas
    probe_positions=positions,   # All 16 also receive
    frequency=frequency,
    delay_factor=5.0
)

# Add head phantom (shared across all simulations)
solver.set_material_circle(
    center_x=nx // 2, center_y=ny // 2,
    radius=60,  # 6 cm radius
    eps_r=40.0,
    sigma_e=0.7
)

print(f"  Timestep: {solver.dt:.4e} s")
print(f"  Total time: {solver.dt * total_steps * 1e9:.2f} ns")
print(f"  Head phantom: r=60 cells, eps_r=40, sigma=0.7 S/m")

# ============================================================
# GPU Batched Run
# ============================================================

print(f"\n{'='*60}")
print("  Running BATCHED GPU (all 16 TX simultaneously)...")
print(f"{'='*60}")

t0 = time.time()
s_matrix_gpu = solver.run()
import cupy as cp
cp.cuda.Device().synchronize()
gpu_time = time.time() - t0

cells_per_sec_gpu = num_antennas * nx * ny * total_steps / gpu_time
print(f"  GPU Batched: {gpu_time:.2f}s")
print(f"  Throughput: {cells_per_sec_gpu/1e6:.1f} Mcell-steps/s")

# ============================================================
# CPU Batched Run (same algorithm, NumPy)
# ============================================================

print(f"\n  Running BATCHED CPU (NumPy, same algorithm)...")
t0 = time.time()
s_matrix_cpu = solver.run_cpu()
cpu_time = time.time() - t0

cells_per_sec_cpu = num_antennas * nx * ny * total_steps / cpu_time
print(f"  CPU Batched: {cpu_time:.2f}s")
print(f"  Throughput: {cells_per_sec_cpu/1e6:.1f} Mcell-steps/s")

speedup = cpu_time / gpu_time
print(f"\n{'='*60}")
print(f"  SPEEDUP: {speedup:.1f}x (GPU batched vs CPU batched)")
print(f"{'='*60}")

# ============================================================
# Validate CPU vs GPU
# ============================================================

print("\nValidating CPU vs GPU...")
max_diff = 0
for tx in range(num_antennas):
    for rx in range(num_antennas):
        diff = np.max(np.abs(
            s_matrix_cpu[tx][rx] - s_matrix_gpu[tx][rx]
        ))
        max_diff = max(max_diff, diff)

print(f"  Max CPU-GPU difference: {max_diff:.2e}")

# ============================================================
# Visualize results
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: S-matrix amplitude
s_peak = np.zeros((num_antennas, num_antennas))
for tx in range(num_antennas):
    for rx in range(num_antennas):
        s_peak[tx, rx] = np.max(np.abs(s_matrix_gpu[tx][rx]))

im = axes[0, 0].imshow(
    20 * np.log10(s_peak / np.max(s_peak) + 1e-10),
    cmap='hot', vmin=-40, vmax=0
)
axes[0, 0].set_xlabel('RX Antenna')
axes[0, 0].set_ylabel('TX Antenna')
axes[0, 0].set_title('S-Matrix: Peak Amplitude (dB)')
plt.colorbar(im, ax=axes[0, 0])

# Plot 2: Time-domain signals from TX=0
ax = axes[0, 1]
t_ns = np.arange(total_steps) * solver.dt * 1e9
for rx in [0, 1, 4, 8, 12]:
    signal = s_matrix_gpu[0][rx]
    norm_val = np.max(np.abs(s_matrix_gpu[0][1])) + 1e-30
    ax.plot(t_ns, signal / norm_val, label=f'RX {rx}', linewidth=1.5)
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Normalized Ez')
ax.set_title('TX=0: Received Signals')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Array geometry
ax = axes[1, 0]
theta = np.linspace(0, 2 * np.pi, 100)
ax.plot(60 * np.cos(theta) + nx//2, 60 * np.sin(theta) + ny//2,
        'b-', linewidth=2, label='Head phantom')
ax.plot(array_radius_cells * np.cos(theta) + nx//2,
        array_radius_cells * np.sin(theta) + ny//2,
        'g--', linewidth=1, alpha=0.5)
for i, pos in enumerate(positions):
    ax.plot(pos[0], pos[1], 'r^', markersize=10)
    ax.annotate(f'{i}', (pos[0]+3, pos[1]+3), fontsize=7)
ax.set_xlim(0, nx)
ax.set_ylim(0, ny)
ax.set_aspect('equal')
ax.set_title('Array Geometry')
ax.legend()

# Plot 4: Performance comparison
ax = axes[1, 1]
methods = ['CPU\n(batched)', 'GPU\n(batched)']
times = [cpu_time, gpu_time]
colors = ['steelblue', 'green']
bars = ax.bar(methods, times, color=colors, edgecolor='black')
ax.set_ylabel('Time (seconds)')
ax.set_title(f'Performance: {speedup:.1f}x GPU Speedup')
ax.grid(True, alpha=0.3, axis='y')
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{t:.1f}s', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('batched_antenna_array_16.png', dpi=150)
plt.show()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("  BATCHED 16-ELEMENT ANTENNA ARRAY SUMMARY")
print("=" * 60)
print(f"  Grid: {nx}x{ny} ({nx*dx*100:.0f} cm x {ny*dx*100:.0f} cm)")
print(f"  Batch: {num_antennas} simultaneous simulations")
print(f"  Phantom: circular head, eps_r=40, sigma=0.7 S/m")
print(f"  Array: {num_antennas} monopoles, {array_radius_cells} mm radius")
print(f"  Channels: {num_antennas}x{num_antennas} = {num_antennas**2}")
print(f"  GPU batched: {gpu_time:.2f}s ({cells_per_sec_gpu/1e6:.0f} Mcell-steps/s)")
print(f"  CPU batched: {cpu_time:.2f}s ({cells_per_sec_cpu/1e6:.0f} Mcell-steps/s)")
print(f"  Speedup: {speedup:.1f}x")
print(f"  CPU-GPU error: {max_diff:.2e}")
print(f"\n  WHY batched is faster:")
print(f"    Sequential: 16 kernels × {nx*ny/1e3:.0f}K cells = underutilized GPU")
print(f"    Batched: 1 kernel × {num_antennas*nx*ny/1e6:.1f}M cells = saturated GPU")
print("=" * 60)
