# Cell 8: MEEP Validation — PML Absorption Quality
# ===================================================

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from neurowave.core.backend import set_backend
from neurowave.core.config import GridConfig, SimulationConfig
from neurowave.sources.waveforms import GaussianSource
from neurowave.solvers.fdtd_2d import FDTD2D
from neurowave.boundaries.absorbing import CPML

set_backend('numpy')

resolution = 20
dx_meep = 1.0 / resolution
dx = dx_meep * 1e-6
fcen = 1.0
df = 0.5

# --- MEEP with PML ---
sim = mp.Simulation(
    cell_size=mp.Vector3(10, 10, 0),
    resolution=resolution,
    sources=[mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(0, 0)
    )],
    boundary_layers=[mp.PML(1.5)]
)

meep_pml_data = []
def record_meep(sim):
    meep_pml_data.append(sim.get_field_point(mp.Ez, mp.Vector3(0, 0)))

sim.run(mp.at_every(sim.fields.dt, record_meep), until=30)
meep_pml_data = np.array(meep_pml_data)

# --- NeuroWave with CPML ---
nx, ny = 200, 200
grid_config = GridConfig(nx=nx, ny=ny, dx=dx, dy=dx)
config = SimulationConfig(grid=grid_config, total_steps=len(meep_pml_data))

source = GaussianSource(
    x=100, y=100,
    frequency_max=fcen / (dx_meep * 1e-6) * 3e8,
    field_component='Ez',
    delay_factor=5.0
)
cpml = CPML(thickness=30)  # ~1.5um at 20 px/um

solver = FDTD2D(config=config, sources=[source], boundaries=[cpml],
                probe_points=[(100, 100)], record_field='Ez')
solver.run()
nw_pml_data = np.array(solver.probe_data[(100, 100)])

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(meep_pml_data, 'b-', label='MEEP', linewidth=1.5)
axes[0].plot(nw_pml_data[:len(meep_pml_data)], 'r--', label='NeuroWave', linewidth=1.5)
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Ez at source')
axes[0].set_title('PML Absorption: Source Point Signal')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Late-time reflection level
late_start = len(meep_pml_data) * 3 // 4
meep_reflection = np.max(np.abs(meep_pml_data[late_start:]))
nw_reflection = np.max(np.abs(nw_pml_data[late_start:len(meep_pml_data)]))
meep_peak_val = np.max(np.abs(meep_pml_data))
nw_peak_val = np.max(np.abs(nw_pml_data))

meep_db = 20 * np.log10(meep_reflection / meep_peak_val) if meep_peak_val > 0 else -np.inf
nw_db = 20 * np.log10(nw_reflection / nw_peak_val) if nw_peak_val > 0 else -np.inf

axes[1].bar(['MEEP', 'NeuroWave'], [meep_db, nw_db], color=['blue', 'red'])
axes[1].set_ylabel('Reflection Level (dB)')
axes[1].set_title('PML Quality: Late-Time Reflection')
axes[1].axhline(y=-40, color='g', linestyle='--', label='-40 dB target')
axes[1].legend()

plt.tight_layout()
plt.savefig('meep_vs_neurowave_pml.png', dpi=150)
plt.show()

print(f"MEEP PML reflection: {meep_db:.1f} dB")
print(f"NeuroWave CPML reflection: {nw_db:.1f} dB")
