import numpy as np
from ceep.solvers.dft import DFTMonitor
from ceep.core.config import GridConfig, SimulationConfig, SimulationMode
from ceep.solvers.fdtd_2d import FDTD2D
from ceep.sources.waveforms import SinusoidalSource

def test_dft_sinusoidal():
    """A sinusoidal source should have a sharp DFT peak at its frequency."""
    freq = 5e9
    # Simulate a small grid for 200 steps to get a clean sinusoid
    grid = GridConfig(nx=10, ny=10, dx=1e-3, dy=1e-3)
    config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=1000)
    
    src = SinusoidalSource(x=5, y=5, frequency=freq)
    
    # Extract DFT at 3 frequencies: 2.5G, 5.0G, 7.5G
    freqs = [2.5e9, 5e9, 7.5e9]
    monitor = DFTMonitor(frequencies=freqs, region=(5, 5), component="Ez")
    
    solver = FDTD2D(config=config, sources=[src], dft_monitors=[monitor])
    solver.run(1000)
    
    c_field = monitor.get_complex_field()
    
    # Region was a single point (5, 5), so shape should be (num_freqs,)
    assert c_field.shape == (3,)
    
    mags = np.abs(c_field)
    
    # 5 GHz should be the strongest magnitude
    assert mags[1] > mags[0] * 5.0
    assert mags[1] > mags[2] * 5.0
