# Cell 6: MEEP Validation — Free Space Propagation
# ==================================================

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from neurowave.core.backend import set_backend
from neurowave.core.config import GridConfig, SimulationConfig
from neurowave.sources.waveforms import GaussianSource
from neurowave.solvers.fdtd_2d import FDTD2D

set_backend('numpy')

resolution = 20  # pixels per um
cell_size = mp.Vector3(10, 10, 0)
fcen = 1.0
df = 0.5

# --- MEEP simulation ---
sources_meep = [mp.Source(
    mp.GaussianSource(frequency=fcen, fwidth=df),
    component=mp.Ez,
    center=mp.Vector3(0, 0)
)]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    sources=sources_meep,
    boundary_layers=[mp.PML(1.0)]
)

monitor_pt = mp.Vector3(2, 0)
meep_data = []

def record_meep(sim):
    meep_data.append(sim.get_field_point(mp.Ez, monitor_pt))

sim.run(mp.at_every(sim.fields.dt, record_meep), until=20)
meep_data = np.array(meep_data)
print(f"MEEP: {len(meep_data)} steps, dt={sim.fields.dt:.6f}")

# --- NeuroWave simulation (same physical setup) ---
dx_meep = 1.0 / resolution  # um
nx, ny = 200, 200
dx = dx_meep * 1e-6  # meters

grid_config = GridConfig(nx=nx, ny=ny, dx=dx, dy=dx)
config = SimulationConfig(grid=grid_config, total_steps=len(meep_data))

# Source at center, probe 2um away (40 cells at 20 px/um)
source = GaussianSource(
    x=100, y=100,
    frequency_max=fcen / (dx_meep * 1e-6) * 3e8,
    field_component='Ez',
    delay_factor=5.0
)

solver = FDTD2D(config=config, sources=[source],
                probe_points=[(140, 100)], record_field='Ez')
solver.run()

nw_data = np.array(solver.probe_data[(140, 100)])

# --- Plot comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

meep_norm = meep_data / np.max(np.abs(meep_data)) if np.max(np.abs(meep_data)) > 0 else meep_data
nw_norm = nw_data / np.max(np.abs(nw_data)) if np.max(np.abs(nw_data)) > 0 else nw_data

axes[0].plot(meep_norm, 'b-', label='MEEP', linewidth=2)
axes[0].plot(nw_norm, 'r--', label='NeuroWave', linewidth=2)
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Normalized Ez')
axes[0].set_title('Probe Signal: MEEP vs NeuroWave')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

meep_peak = np.argmax(np.abs(meep_norm))
nw_peak = np.argmax(np.abs(nw_norm))
axes[1].bar(['MEEP', 'NeuroWave'], [meep_peak, nw_peak], color=['blue', 'red'])
axes[1].set_ylabel('Peak Arrival (timestep)')
axes[1].set_title(f'Wave Arrival: MEEP={meep_peak}, NW={nw_peak}')

plt.tight_layout()
plt.savefig('meep_vs_neurowave_freespace.png', dpi=150)
plt.show()

print(f"\nMEEP peak at step: {meep_peak}")
print(f"NeuroWave peak at step: {nw_peak}")
print(f"Difference: {abs(meep_peak - nw_peak)} steps")
