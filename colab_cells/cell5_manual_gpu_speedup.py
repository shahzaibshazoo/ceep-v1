# Cell 5: Manual GPU Speedup Test
# ==================================

import time
import numpy as np
import sys
sys.path.insert(0, 'src')

from neurowave.core.backend import set_backend, to_numpy
from neurowave.core.config import GridConfig, SimulationConfig
from neurowave.sources.waveforms import GaussianSource
from neurowave.solvers.fdtd_2d import FDTD2D
from neurowave.boundaries.absorbing import CPML

# 1000x1000 grid — 200 steps
grid_config = GridConfig(nx=1000, ny=1000, dx=1e-3, dy=1e-3)
config = SimulationConfig(grid=grid_config, total_steps=200)
source = GaussianSource(x=500, y=500, frequency_max=10e9, field_component='Ez', delay_factor=5.0)
cpml = CPML(thickness=10)

# --- CPU ---
set_backend('numpy')
solver = FDTD2D(config=config, sources=[source], boundaries=[cpml])
t0 = time.time()
solver.run()
cpu_time = time.time() - t0
ez_cpu = np.array(solver.get_field('Ez'))
print(f"CPU: {cpu_time:.2f}s  ({1000*1000*200/cpu_time/1e6:.1f} Mcell-steps/s)")

# --- GPU ---
set_backend('cupy')
import cupy as cp
solver = FDTD2D(config=config, sources=[source], boundaries=[cpml])
# Warmup JIT
for _ in range(3):
    solver.step()
cp.cuda.Device().synchronize()
# Fresh run
solver = FDTD2D(config=config, sources=[source], boundaries=[cpml])
cp.cuda.Device().synchronize()
t0 = time.time()
solver.run()
cp.cuda.Device().synchronize()
gpu_time = time.time() - t0
ez_gpu = to_numpy(solver.get_field('Ez'))
print(f"GPU: {gpu_time:.2f}s  ({1000*1000*200/gpu_time/1e6:.1f} Mcell-steps/s)")

print(f"\nSpeedup: {cpu_time/gpu_time:.1f}x")
print(f"Max CPU-GPU difference: {np.max(np.abs(ez_cpu - ez_gpu)):.2e}")
