"""
Radar Ranging Test: 10 Meter Target Detection
=============================================
Simulates a pulse travel 10 meters, reflect off a target, and return.
Calculates range based on time-of-flight.
"""

import math, time, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from neurowave.core.config    import GridConfig, SimulationConfig, SimulationMode
from neurowave.core.constants import C_0, DEFAULT_COURANT_2D
from neurowave.solvers.fdtd_2d import FDTD2D
from neurowave.sources.waveforms import ModulatedGaussianSource
from neurowave.boundaries.absorbing import CPML

# ─── Shared physical parameters ───────────────────────────────────────────────
DX     = 5e-3        # 5 mm
NX     = 2200        # 2200 * 5mm = 11 meters
NY     = 50          # moderate strip
FC     = 10e9        # 10 GHz
TAU    = 1.0 / (math.pi * FC)   # ~31.8 ps
DELAY  = 10.0        # more delay for better separation
T0     = DELAY * TAU            # ~318 ps (pulse peak)

# Source and Probe at x=100 (500 mm)
SRC_X  = 100
SRC_Y  = 25
# Target at x=2100 (10.5 meters)
TARGET_X = 2100
TARGET_Y = 25

# Distance from Source to Target = (2100 - 100) * 5mm = 2000 * 5mm = 10 meters
DISTANCE = (TARGET_X - SRC_X) * DX
ROUND_TRIP = 2 * DISTANCE
EXPECTED_TOF = ROUND_TRIP / C_0
EXPECTED_ARRIVAL = T0 + EXPECTED_TOF

# Timesteps
# dt = S * dx / (c * sqrt(2))
DT_NW = DEFAULT_COURANT_2D * DX / (C_0 * math.sqrt(2))
# Need to cover at least EXPECTED_ARRIVAL + some margin
STEPS = int((EXPECTED_ARRIVAL * 1.2) / DT_NW)

print("═" * 64)
print(f"  Radar Ranging Setup:")
print(f"  Distance to Target: {DISTANCE:.2f} m")
print(f"  Round Trip Time: {EXPECTED_TOF*1e9:.2f} ns")
print(f"  Expected Echo at: {EXPECTED_ARRIVAL*1e9:.2f} ns")
print(f"  Grid: {NX}x{NY} cells, DX={DX*1e3:.1f}mm")
print(f"  Simulation Steps: {STEPS}")
print("═" * 64)

def waveform_si(t_s: float) -> float:
    env = math.exp(-((t_s - T0) / TAU) ** 2)
    return env * math.sin(2.0 * math.pi * FC * t_s)

class _NWSource(ModulatedGaussianSource):
    def value_at(self, timestep: int, dt: float) -> float:
        return waveform_si(timestep * dt)

def run_neurowave():
    print("\n── Running NeuroWave Radar Test ──────────────────────────────────")
    grid   = GridConfig(nx=NX, ny=NY, dx=DX, dy=DX)
    config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=STEPS)

    # Use CPML to absorb backward wave and side reflections
    pml = CPML(thickness=10)

    src = _NWSource(x=SRC_X, y=SRC_Y, frequency=FC, bandwidth=FC,
                    field_component="Ez", delay_factor=DELAY)

    # Place probe at source to see both incident and reflected wave
    solver = FDTD2D(config=config, sources=[src], boundaries=[pml],
                    record_field="Ez", probe_points=[(SRC_X, SRC_Y)])

    # Target: PEC block or high epsilon
    # We'll use a PEC-like material (high eps and sigma) or just a very high epsilon
    # For now, let's use eps_r=100 as a simple reflector
    solver.grid.set_material_region(TARGET_X, TARGET_X + 2, 0, NY, eps_r=100.0)

    t0w = time.time()
    solver.run()
    elapsed = time.time() - t0w

    hist = np.array(solver.probe_data[(SRC_X, SRC_Y)])
    t    = np.arange(STEPS) * config.dt
    
    # Ranging: find echo peak
    # incident peak is around T0
    # Search for echo tightly around EXPECTED_ARRIVAL
    # Look in window [EXPECTED_ARRIVAL - 2ns, EXPECTED_ARRIVAL + 2ns]
    echo_window = (t > (EXPECTED_ARRIVAL - 2e-9)) & (t < (EXPECTED_ARRIVAL + 2e-9))
    if any(echo_window):
        echo_idx = np.where(echo_window)[0][np.argmax(np.abs(hist[echo_window]))]
        echo_t = t[echo_idx]
        measured_range = (C_0 * (echo_t - T0)) / 2
        print(f"  NeuroWave Detected Echo at: {echo_t*1e9:.2f} ns")
        print(f"  Measured Range: {measured_range:.4f} m (Error: {(measured_range-DISTANCE)*100:.2f} cm)")
    else:
        print("  NeuroWave: No echo detected.")
        echo_t = 0
        measured_range = 0

    return t, hist, measured_range

def run_meep():
    print("\n── Running MEEP Radar Test ───────────────────────────────────────")
    try:
        import meep as mp
    except ImportError:
        print("  MEEP not installed.")
        return None, None, 0

    # a = DX = 5mm
    a = DX
    c_a = C_0 / a 

    def src_func(t_meep):
        return waveform_si(t_meep * a / C_0)

    sources = [mp.Source(
        mp.CustomSource(src_func=src_func, is_integrated=False),
        component=mp.Ez,
        center=mp.Vector3(0, 0, 0), # Source at MEEP origin
    )]

    # MEEP Target position: DISTANCE / a units from source
    target_pos_m = DISTANCE / a
    geometry = [mp.Block(
        size=mp.Vector3(2, mp.inf, mp.inf),
        center=mp.Vector3(target_pos_m, 0, 0),
        material=mp.Medium(epsilon=100.0)
    )]

    cell = mp.Vector3(NX, NY, 0)
    pml_layers = [mp.PML(10)] # 10 cells PML
    
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=1, # 1 pixel per DX
    )

    meep_dt_m = 0.5
    meep_dt_si = meep_dt_m * a / C_0
    run_time_si = STEPS * DT_NW
    run_time_m = run_time_si * c_a

    hist_m = []
    def rec(sim):
        hist_m.append(sim.get_field_point(mp.Ez, mp.Vector3(0, 0, 0)).real)

    t0w = time.time()
    sim.run(mp.at_every(meep_dt_m, rec), until=run_time_m)
    elapsed = time.time() - t0w

    t_m = np.arange(len(hist_m)) * meep_dt_si
    h_m = np.array(hist_m)
    
    echo_window = (t_m > (EXPECTED_ARRIVAL - 2e-9)) & (t_m < (EXPECTED_ARRIVAL + 2e-9))
    if any(echo_window):
        echo_idx = np.where(echo_window)[0][np.argmax(np.abs(h_m[echo_window]))]
        echo_t = t_m[echo_idx]
        measured_range = (C_0 * (echo_t - T0)) / 2
        print(f"  MEEP Detected Echo at: {echo_t*1e9:.2f} ns")
        print(f"  Measured Range: {measured_range:.4f} m (Error: {(measured_range-DISTANCE)*100:.2f} cm)")
    else:
        print("  MEEP: No echo detected.")
        measured_range = 0

    return t_m, h_m, measured_range

if __name__ == "__main__":
    t_nw, h_nw, r_nw = run_neurowave()
    t_m, h_m, r_m = run_meep()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t_nw * 1e9, h_nw / np.max(np.abs(h_nw)), label=f'NeuroWave (Range: {r_nw:.3f}m)')
    if t_m is not None:
        plt.plot(t_m * 1e9, h_m / np.max(np.abs(h_m)), '--', label=f'MEEP (Range: {r_m:.3f}m)')
    
    plt.title(f"Radar Ranging: Target at {DISTANCE:.2f} meters")
    plt.xlabel("Time (ns)")
    plt.ylabel("Normalized Ez")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('radar_ranging_test.png')
    print("\nSaved plot to radar_ranging_test.png")
