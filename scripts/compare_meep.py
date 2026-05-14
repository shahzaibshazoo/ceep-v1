"""
NeuroWave vs MEEP — Free Space & Dielectric Validation
=======================================================
Two tests run sequentially:
  Test 1 (Free Space): No dielectric. Verify wave arrives at t0 + d/c.
  Test 2 (Dielectric): eps_r=4 block. Compare both solvers.
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

# ─── Shared physical parameters ───────────────────────────────────────────────
DX     = 1e-3        # 1 mm
NX = NY = 100
FC     = 10e9        # 10 GHz
TAU    = 1.0 / (math.pi * FC)   # ~31.8 ps
DELAY  = 5.0
T0     = DELAY * TAU             # ~159 ps (pulse peak)
STEPS  = 600

# Source at NW (50,50), probe at NW (80,50) = 30 mm away
SRC_X, SRC_Y   = 50, 50
PROBE_X, PROBE_Y = 80, 50

# Expected free-space arrival: T0 + distance/c
FREE_DIST = (PROBE_X - SRC_X) * DX  # 30 mm
EXPECTED_FS_ARRIVAL = T0 + FREE_DIST / C_0

# Block: NW x=[60,80], y=[30,70]  (probe is ON far edge of block)
# Better: probe outside — put at NW (85,50)
PROBE_DIEL_X = 85
EXPECTED_DIEL_ARRIVAL = (T0
    + (60 - SRC_X) * DX / C_0             # free space to block
    + 20e-3 / (C_0 / 2)                   # through eps_r=4 (n=2)
    + (PROBE_DIEL_X - 80) * DX / C_0)     # free space after block

print("═" * 64)
print(f"  DX={DX*1e3:.0f}mm  NX=NY={NX}  FC={FC/1e9:.0f}GHz")
print(f"  τ={TAU*1e12:.2f}ps  t0={T0*1e12:.1f}ps  steps={STEPS}")
print(f"  dt_NW = {DEFAULT_COURANT_2D*DX/(C_0*math.sqrt(2))*1e12:.4f} ps")
print(f"  Free-space probe: NW({PROBE_X},{PROBE_Y})  d={FREE_DIST*1e3:.0f}mm")
print(f"  Expected FS arrival: {EXPECTED_FS_ARRIVAL*1e12:.1f} ps")
print(f"  Expected diel arrival: {EXPECTED_DIEL_ARRIVAL*1e12:.1f} ps")
# ─── Peak detection (windowed — avoids reflection peaks) ────────────────────
def first_arrival_peak(t: np.ndarray, h: np.ndarray,
                       expected_t: float, window: float = 4 * TAU):
    """Find the peak within ±window of the expected analytical arrival."""
    mask = (t >= expected_t - window) & (t <= expected_t + window)
    if not mask.any():
        idx = np.argmax(np.abs(h))
    else:
        sub = np.abs(h[mask])
        idx = np.where(mask)[0][np.argmax(sub)]
    return t[idx], h[idx]

# ─── Shared waveform (SI seconds) ─────────────────────────────────────────────
def waveform_si(t_s: float) -> float:
    env = math.exp(-((t_s - T0) / TAU) ** 2)
    return env * math.sin(2.0 * math.pi * FC * t_s)


# ─── NeuroWave runner ─────────────────────────────────────────────────────────
class _NWSource(ModulatedGaussianSource):
    def value_at(self, timestep: int, dt: float) -> float:
        return waveform_si(timestep * dt)


def run_neurowave(use_block: bool):
    label = "Dielectric" if use_block else "Free Space"
    print(f"\n── NeuroWave [{label}] " + "─" * 40)
    grid   = GridConfig(nx=NX, ny=NY, dx=DX, dy=DX)
    config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=STEPS)

    py = PROBE_DIEL_Y = 50
    px = PROBE_DIEL_X if use_block else PROBE_X

    src = _NWSource(x=SRC_X, y=SRC_Y, frequency=FC, bandwidth=FC,
                    field_component="Ez", delay_factor=DELAY)

    solver = FDTD2D(config=config, sources=[src],
                    record_field="Ez", probe_points=[(px, py)])

    if use_block:
        solver.grid.set_material_region(60, 80, 30, 70, eps_r=4.0)

    t0w = time.time()
    solver.run()
    elapsed = time.time() - t0w

    hist = np.array(solver.probe_data[(px, py)])
    t    = np.arange(STEPS) * config.dt
    exp  = EXPECTED_DIEL_ARRIVAL if use_block else EXPECTED_FS_ARRIVAL
    peak_t, peak_v = first_arrival_peak(t, hist, exp)
    print(f"  first-arrival peak at {peak_t*1e12:.2f} ps | |Ez|_max={np.max(np.abs(hist)):.3e} | {elapsed:.3f}s")
    return t, hist, elapsed, peak_t


# ─── MEEP runner ──────────────────────────────────────────────────────────────
def run_meep(use_block: bool):
    label = "Dielectric" if use_block else "Free Space"
    print(f"\n── MEEP [{label}] " + "─" * 44)
    try:
        import meep as mp
    except ImportError:
        print("  MEEP not installed.")
        return None, None, None, None

    a   = DX
    c_a = C_0 / a   # 3e11

    def src_func(t_meep):
        return waveform_si(t_meep * a / C_0)

    # Use CustomSource with NO extra envelope (center_frequency/fwidth=0)
    # MEEP applies a Hanning taper if fwidth>0, which delays the effective source.
    sources = [mp.Source(
        mp.CustomSource(src_func=src_func, is_integrated=False),
        component=mp.Ez,
        center=mp.Vector3(0, 0, 0),   # NW (50,50)
    )]

    # MEEP coords = NW coords - 50
    px_meep = (PROBE_DIEL_X if use_block else PROBE_X) - 50
    pt = mp.Vector3(px_meep, 0, 0)

    geometry = []
    if use_block:
        # NW block x=[60,80],y=[30,70] → MEEP centre=(20,0), size=(20,40)
        geometry = [mp.Block(
            size=mp.Vector3(20, 40, mp.inf),
            center=mp.Vector3(20, 0, 0),
            material=mp.Medium(epsilon=4.0),
        )]

    sim = mp.Simulation(
        cell_size=mp.Vector3(NX, NY, 0),
        boundary_layers=[],
        geometry=geometry,
        sources=sources,
        resolution=1,
    )

    meep_dt_m  = 0.5
    meep_dt_si = meep_dt_m * a / C_0
    run_time_si = STEPS * DEFAULT_COURANT_2D * DX / (C_0 * math.sqrt(2))
    run_time_m  = run_time_si * c_a

    hist_m = []
    def rec(sim): hist_m.append(sim.get_field_point(mp.Ez, pt).real)

    t0w = time.time()
    sim.run(mp.at_every(meep_dt_m, rec), until=run_time_m)
    elapsed = time.time() - t0w

    t_m = np.arange(len(hist_m)) * meep_dt_si
    h_m = np.array(hist_m)
    exp  = EXPECTED_DIEL_ARRIVAL if use_block else EXPECTED_FS_ARRIVAL
    peak_t, _ = first_arrival_peak(t_m, h_m, exp)
    print(f"  first-arrival peak at {peak_t*1e12:.2f} ps | |Ez|_max={np.max(np.abs(h_m)):.3e} | {elapsed:.3f}s")
    return t_m, h_m, elapsed, peak_t


# ─── Plot ─────────────────────────────────────────────────────────────────────
def make_plot(results):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    titles = ["Test 1: Free Space (probe 30 mm from source)",
              "Test 2: Dielectric Block ε_r=4 (probe 5 mm past block)"]
    expected = [EXPECTED_FS_ARRIVAL, EXPECTED_DIEL_ARRIVAL]

    for ax, (t_nw, h_nw, t_m, h_m), title, exp in zip(axes, results, titles, expected):
        # Normalize with SIGN preserved (use value at analytic peak, not abs max)
        def norm_signed(t, h):
            idx = np.argmax(np.abs(h))
            return h / np.abs(h[idx])

        hn = norm_signed(t_nw, h_nw)
        ax.plot(t_nw * 1e12, hn, color='steelblue', lw=2, label='NeuroWave')

        if t_m is not None:
            hm = norm_signed(t_m, h_m)
            # Align polarity: flip MEEP if its sign at NW's peak time is opposite
            nw_peak_t = t_nw[np.argmax(np.abs(h_nw))]
            hm_at_nw_peak = np.interp(nw_peak_t, t_m, hm)
            polarity = np.sign(hn[np.argmax(np.abs(hn))]) * np.sign(hm_at_nw_peak)
            if polarity < 0:
                hm = -hm
            ax.plot(t_m * 1e12, hm, '--', color='tomato', lw=2, label='MEEP (polarity aligned)')

        ax.axvline(exp * 1e12, color='green', ls=':', lw=1.5,
                   label=f'Analytical t_arrive={exp*1e12:.0f}ps')
        ax.set_title(title); ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Norm. Ez"); ax.legend(fontsize=9); ax.grid(alpha=0.4)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'meep_comparison.png')
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for use_block in [False, True]:
        t_nw, h_nw, tnw_w, pk_nw = run_neurowave(use_block)
        t_m,  h_m,  tm_w,  pk_m  = run_meep(use_block)

        label = "Dielectric" if use_block else "Free Space"
        exp   = EXPECTED_DIEL_ARRIVAL if use_block else EXPECTED_FS_ARRIVAL
        print(f"\n  [{label}] Analytical expected: {exp*1e12:.1f} ps")
        print(f"  NeuroWave Δt = {(pk_nw-exp)*1e12:+.1f} ps")
        if pk_m: print(f"  MEEP      Δt = {(pk_m-exp)*1e12:+.1f} ps")
        results.append((t_nw, h_nw, t_m, h_m))

    make_plot(results)
