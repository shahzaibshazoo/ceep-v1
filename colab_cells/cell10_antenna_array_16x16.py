# Cell 10: 16-Element Circular Antenna Array Simulation (GPU)
# =============================================================
# Simulates a 16-antenna circular array for microwave head imaging.
# Runs multistatic acquisition: each antenna transmits, all others receive.
# Compares CPU vs GPU performance for the full imaging pipeline.

import time
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from neurowave.core.backend import set_backend, to_numpy
from neurowave.core.config import GridConfig, SimulationConfig
from neurowave.sources.waveforms import GaussianSource
from neurowave.solvers.fdtd_2d import FDTD2D
from neurowave.boundaries.absorbing import CPML
from neurowave.antennas.arrays import CircularArray

# ============================================================
# Setup: 16-element circular array around a head phantom
# ============================================================

# Grid: 300x300 cells at 1mm resolution = 30cm x 30cm domain
nx, ny = 300, 300
dx = 1e-3  # 1 mm
center = (nx // 2, ny // 2)  # Center of domain

# Array configuration
num_antennas = 16
array_radius_mm = 100  # 10 cm radius (20 cm diameter ring)
frequency = 1.0e9  # 1 GHz center frequency
bandwidth = 0.5e9  # 500 MHz bandwidth

# Create circular array
array = CircularArray(
    num_antennas=num_antennas,
    radius_mm=array_radius_mm,
    center=center,
    dx=dx,
    antenna_type='monopole',
    polarization='vertical'
)

positions = array.get_antenna_positions()
print(f"Array Configuration:")
print(f"  Antennas: {num_antennas}")
print(f"  Radius: {array_radius_mm} mm")
print(f"  Grid: {nx}x{ny} @ {dx*1e3:.1f} mm")
print(f"  Frequency: {frequency/1e9:.1f} GHz")
print(f"  Antenna positions (grid):")
for i, pos in enumerate(positions):
    print(f"    Ant {i:2d}: ({pos[0]:3d}, {pos[1]:3d})")

# ============================================================
# Simulation parameters
# ============================================================

grid_config = GridConfig(nx=nx, ny=ny, dx=dx, dy=dx)
# Run enough steps for wave to traverse the domain twice
total_steps = 400
config = SimulationConfig(grid=grid_config, total_steps=total_steps)

print(f"\n  Timestep: {config.dt:.4e} s")
print(f"  Total time: {config.dt * total_steps * 1e9:.2f} ns")
print(f"  Steps: {total_steps}")

# ============================================================
# Run multistatic simulation — GPU
# ============================================================

print("\n" + "=" * 60)
print("  Multistatic Acquisition: 16 TX × 16 RX = 256 channels")
print("=" * 60)

def run_single_tx(tx_idx, backend_name):
    """Run one transmit event, record at all antennas."""
    set_backend(backend_name)

    # Create source at TX antenna
    tx_pos = positions[tx_idx]
    source = GaussianSource(
        x=tx_pos[0], y=tx_pos[1],
        frequency_max=frequency,
        field_component='Ez',
        delay_factor=5.0
    )

    # All antennas are probes (receive)
    probe_points = [(p[0], p[1]) for p in positions]
    cpml = CPML(thickness=10)

    solver = FDTD2D(
        config=config,
        sources=[source],
        boundaries=[cpml],
        probe_points=probe_points,
        record_field='Ez'
    )

    # Add a simple circular "head" phantom (eps_r=40, simulating brain tissue)
    solver.grid.set_material_circle(
        center_x=nx // 2, center_y=ny // 2,
        radius=60,  # 6 cm radius head
        eps_r=40.0,  # Average brain permittivity at 1 GHz
        sigma_e=0.7  # Conductivity ~0.7 S/m
    )

    solver.run()

    # Collect received signals at all antennas
    signals = {}
    for rx_idx, pt in enumerate(probe_points):
        signals[rx_idx] = np.array(solver.probe_data[pt])

    return signals


# --- GPU multistatic run ---
set_backend('cupy')
import cupy as cp

print("\nRunning GPU multistatic acquisition...")
t0 = time.time()

s_matrix_gpu = {}
for tx in range(num_antennas):
    signals = run_single_tx(tx, 'cupy')
    s_matrix_gpu[tx] = signals
    if (tx + 1) % 4 == 0:
        cp.cuda.Device().synchronize()
        elapsed = time.time() - t0
        print(f"  TX {tx+1:2d}/{num_antennas} done ({elapsed:.1f}s)")

cp.cuda.Device().synchronize()
gpu_total = time.time() - t0
print(f"\nGPU Total: {gpu_total:.1f}s for {num_antennas} transmissions")
print(f"GPU Rate: {nx*ny*total_steps*num_antennas/gpu_total/1e6:.1f} Mcell-steps/s")

# --- CPU multistatic run ---
print("\nRunning CPU multistatic acquisition...")
t0 = time.time()

s_matrix_cpu = {}
for tx in range(num_antennas):
    signals = run_single_tx(tx, 'numpy')
    s_matrix_cpu[tx] = signals
    if (tx + 1) % 4 == 0:
        elapsed = time.time() - t0
        print(f"  TX {tx+1:2d}/{num_antennas} done ({elapsed:.1f}s)")

cpu_total = time.time() - t0
print(f"\nCPU Total: {cpu_total:.1f}s for {num_antennas} transmissions")
print(f"CPU Rate: {nx*ny*total_steps*num_antennas/cpu_total/1e6:.1f} Mcell-steps/s")

print(f"\n{'='*60}")
print(f"  SPEEDUP: {cpu_total/gpu_total:.1f}x")
print(f"{'='*60}")

# ============================================================
# Validate CPU vs GPU match
# ============================================================

print("\nValidating CPU vs GPU results...")
max_diff = 0
for tx in range(num_antennas):
    for rx in range(num_antennas):
        diff = np.max(np.abs(s_matrix_cpu[tx][rx] - s_matrix_gpu[tx][rx]))
        max_diff = max(max_diff, diff)

print(f"  Max CPU-GPU difference across all 256 channels: {max_diff:.2e}")

# ============================================================
# Visualize S-matrix (multistatic response)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: S-matrix amplitude (peak response)
s_peak = np.zeros((num_antennas, num_antennas))
for tx in range(num_antennas):
    for rx in range(num_antennas):
        s_peak[tx, rx] = np.max(np.abs(s_matrix_gpu[tx][rx]))

im = axes[0, 0].imshow(20 * np.log10(s_peak / np.max(s_peak) + 1e-10),
                         cmap='hot', vmin=-40, vmax=0)
axes[0, 0].set_xlabel('RX Antenna')
axes[0, 0].set_ylabel('TX Antenna')
axes[0, 0].set_title('S-Matrix: Peak Amplitude (dB)')
plt.colorbar(im, ax=axes[0, 0])

# Plot 2: Time-domain signals from TX=0
ax = axes[0, 1]
t_ns = np.arange(total_steps) * config.dt * 1e9
for rx in [0, 1, 4, 8, 12]:
    signal = s_matrix_gpu[0][rx]
    ax.plot(t_ns, signal / np.max(np.abs(s_matrix_gpu[0][1]) + 1e-30),
            label=f'RX {rx}', linewidth=1.5)
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Normalized Ez')
ax.set_title('TX=0: Received Signals')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Array geometry
ax = axes[1, 0]
theta = np.linspace(0, 2 * np.pi, 100)
# Head outline
ax.plot(60 * np.cos(theta) + nx//2, 60 * np.sin(theta) + ny//2, 'b-', linewidth=2, label='Head phantom')
# Array ring
ax.plot(array_radius_mm * np.cos(theta) + nx//2, array_radius_mm * np.sin(theta) + ny//2,
        'g--', linewidth=1, alpha=0.5)
# Antenna positions
for i, pos in enumerate(positions):
    ax.plot(pos[0], pos[1], 'r^', markersize=10)
    ax.annotate(f'{i}', (pos[0]+3, pos[1]+3), fontsize=7)
ax.set_xlim(0, nx)
ax.set_ylim(0, ny)
ax.set_aspect('equal')
ax.set_title('Array Geometry')
ax.legend()

# Plot 4: Delay profile (time of flight)
ax = axes[1, 1]
tof = np.zeros(num_antennas)
threshold = 0.01
for rx in range(num_antennas):
    signal = np.abs(s_matrix_gpu[0][rx])
    if np.max(signal) > 0:
        sig_norm = signal / np.max(signal)
        arrivals = np.where(sig_norm > threshold)[0]
        if len(arrivals) > 0:
            tof[rx] = arrivals[0] * config.dt * 1e9

ax.bar(range(num_antennas), tof, color='steelblue')
ax.set_xlabel('RX Antenna Index')
ax.set_ylabel('First Arrival (ns)')
ax.set_title('TX=0: Time of Flight to Each RX')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('antenna_array_16_simulation.png', dpi=150)
plt.show()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("  16-ELEMENT ANTENNA ARRAY SIMULATION SUMMARY")
print("=" * 60)
print(f"  Grid: {nx}x{ny} ({nx*dx*100:.0f} cm x {ny*dx*100:.0f} cm)")
print(f"  Phantom: circular head, eps_r=40, sigma=0.7 S/m")
print(f"  Array: {num_antennas} monopoles, {array_radius_mm} mm radius")
print(f"  Frequency: {frequency/1e9:.1f} GHz")
print(f"  Channels: {num_antennas}x{num_antennas} = {num_antennas**2}")
print(f"  GPU time: {gpu_total:.1f}s")
print(f"  CPU time: {cpu_total:.1f}s")
print(f"  Speedup: {cpu_total/gpu_total:.1f}x")
print(f"  CPU-GPU error: {max_diff:.2e}")
print("=" * 60)
