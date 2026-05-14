# Cell 7: MEEP Validation — Dielectric Slab
# ============================================

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

resolution = 20
cell_size = mp.Vector3(10, 10, 0)
fcen = 1.0
df = 0.5
dx_meep = 1.0 / resolution
dx = dx_meep * 1e-6

# --- MEEP with dielectric slab (eps_r=4) ---
geometry = [mp.Block(
    size=mp.Vector3(2, mp.inf, mp.inf),
    center=mp.Vector3(2, 0),
    material=mp.Medium(epsilon=4)
)]

sources_meep = [mp.Source(
    mp.GaussianSource(frequency=fcen, fwidth=df),
    component=mp.Ez,
    center=mp.Vector3(0, 0)
)]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    sources=sources_meep,
    geometry=geometry,
    boundary_layers=[mp.PML(1.0)]
)

meep_slab_data = []
def record_meep(sim):
    meep_slab_data.append(sim.get_field_point(mp.Ez, mp.Vector3(3, 0)))

sim.run(mp.at_every(sim.fields.dt, record_meep), until=20)
meep_slab_data = np.array(meep_slab_data)

# --- NeuroWave with dielectric slab ---
nx, ny = 200, 200
grid_config = GridConfig(nx=nx, ny=ny, dx=dx, dy=dx)
config = SimulationConfig(grid=grid_config, total_steps=len(meep_slab_data))

source = GaussianSource(
    x=100, y=100,
    frequency_max=fcen / (dx_meep * 1e-6) * 3e8,
    field_component='Ez',
    delay_factor=5.0
)

solver = FDTD2D(config=config, sources=[source],
                probe_points=[(160, 100)], record_field='Ez')

# Slab from x=120 to x=160 (2um wide, centered at 2um from source)
solver.grid.set_material_region(120, 160, 0, 200, eps_r=4.0)
solver.run()

nw_slab_data = np.array(solver.probe_data[(160, 100)])

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

meep_slab_norm = meep_slab_data / np.max(np.abs(meep_slab_data)) if np.max(np.abs(meep_slab_data)) > 0 else meep_slab_data
nw_slab_norm = nw_slab_data / np.max(np.abs(nw_slab_data)) if np.max(np.abs(nw_slab_data)) > 0 else nw_slab_data

ax.plot(meep_slab_norm, 'b-', label='MEEP', linewidth=2)
ax.plot(nw_slab_norm, 'r--', label='NeuroWave', linewidth=2)
ax.set_xlabel('Timestep')
ax.set_ylabel('Normalized Ez')
ax.set_title('Dielectric Slab (eps_r=4): MEEP vs NeuroWave')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('meep_vs_neurowave_slab.png', dpi=150)
plt.show()

meep_slab_peak = np.argmax(np.abs(meep_slab_norm))
nw_slab_peak = np.argmax(np.abs(nw_slab_norm))
print(f"Through slab — MEEP peak: {meep_slab_peak}, NeuroWave peak: {nw_slab_peak}")
print(f"Expected slowdown factor: {np.sqrt(4):.1f}x (eps_r=4 -> n=2)")
