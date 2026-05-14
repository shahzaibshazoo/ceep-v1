"""
Comprehensive tests for the FDTD 2D solver, sources, boundaries, and grid.
"""

import math
import pytest
import numpy as np

from ceep.core.config import GridConfig, SimulationConfig, SimulationMode
from ceep.core.constants import C_0, EPS_0, MU_0
from ceep.core.grid import Grid2D
from ceep.sources.waveforms import (
    GaussianSource, SinusoidalSource, ModulatedGaussianSource,
)
from ceep.boundaries.absorbing import PEC, MurABC, CPML
from ceep.solvers.fdtd_2d import FDTD2D


# ============================================================
# Grid Tests
# ============================================================

class TestGrid2D:
    """Tests for 2D Yee grid."""

    def _make_config(self, nx=50, ny=50, steps=10):
        grid = GridConfig(nx=nx, ny=ny, dx=1e-3, dy=1e-3)
        return SimulationConfig(grid=grid, total_steps=steps)

    def test_grid_creation(self):
        config = self._make_config()
        grid = Grid2D(config)
        assert grid.ez.shape == (50, 50)
        assert grid.hx.shape == (50, 50)
        assert grid.hy.shape == (50, 50)

    def test_initial_fields_zero(self):
        config = self._make_config()
        grid = Grid2D(config)
        assert np.all(grid.ez == 0.0)
        assert np.all(grid.hx == 0.0)
        assert np.all(grid.hy == 0.0)

    def test_default_free_space(self):
        config = self._make_config()
        grid = Grid2D(config)
        assert np.all(grid.eps_r == 1.0)
        assert np.all(grid.mu_r == 1.0)
        assert np.all(grid.sigma_e == 0.0)

    def test_material_region(self):
        config = self._make_config()
        grid = Grid2D(config)
        grid.set_material_region(10, 20, 10, 20, eps_r=4.0, sigma_e=0.1)
        assert grid.eps_r[15, 15] == 4.0
        assert grid.sigma_e[15, 15] == 0.1
        assert grid.eps_r[0, 0] == 1.0  # Outside region

    def test_material_circle(self):
        config = self._make_config()
        grid = Grid2D(config)
        grid.set_material_circle(25, 25, 5, eps_r=2.0)
        assert grid.eps_r[25, 25] == 2.0  # Center
        assert grid.eps_r[0, 0] == 1.0    # Far corner

    def test_invalid_material(self):
        config = self._make_config()
        grid = Grid2D(config)
        with pytest.raises(ValueError):
            grid.set_material_region(0, 10, 0, 10, eps_r=0.5)  # < 1.0

    def test_reset_fields(self):
        config = self._make_config()
        grid = Grid2D(config)
        grid.ez[10, 10] = 1.0
        grid.reset_fields()
        assert grid.ez[10, 10] == 0.0

    def test_memory_usage(self):
        config = self._make_config(nx=100, ny=100)
        grid = Grid2D(config)
        # 11 arrays × 10000 cells × 8 bytes = 880000 bytes
        assert grid.memory_usage_bytes == 11 * 10000 * 8

    def test_update_coefficients_free_space(self):
        """In free space, Ca=1 and Cb=dt/eps0."""
        config = self._make_config()
        grid = Grid2D(config)
        dt = config.dt
        np.testing.assert_allclose(grid._ca, 1.0)
        np.testing.assert_allclose(grid._cb, dt / EPS_0)
        np.testing.assert_allclose(grid._da, 1.0)
        np.testing.assert_allclose(grid._db, dt / MU_0)


# ============================================================
# Source Tests
# ============================================================

class TestSources:
    """Tests for electromagnetic sources."""

    def test_gaussian_peak(self):
        """Gaussian should peak at t0."""
        src = GaussianSource(x=50, y=50, frequency_max=5e9)
        dt = 1e-12
        t0_step = int(src.t0 / dt)
        val_at_peak = src.value_at(t0_step, dt)
        # Should be close to amplitude (1.0) at peak
        assert val_at_peak > 0.99

    def test_gaussian_symmetry(self):
        """Gaussian should be symmetric around t0."""
        src = GaussianSource(x=0, y=0, frequency_max=1e9)
        dt = 1e-12
        # Use exact float division for symmetric sampling
        t0_step = round(src.t0 / dt)
        offset = 50
        v_before = src.value_at(t0_step - offset, dt)
        v_after = src.value_at(t0_step + offset, dt)
        # Relative symmetry check (int rounding introduces small asymmetry)
        avg = (v_before + v_after) / 2.0
        assert abs(v_before - v_after) / (avg + 1e-30) < 0.01  # < 1% asymmetry

    def test_gaussian_starts_near_zero(self):
        """With delay_factor=6, initial value should be ~0."""
        src = GaussianSource(x=0, y=0, frequency_max=1e9)
        dt = 1e-12
        assert abs(src.value_at(0, dt)) < 1e-10

    def test_sinusoidal_zero_at_start(self):
        """Sinusoidal with ramp should start at ~0."""
        src = SinusoidalSource(x=0, y=0, frequency=1e9)
        dt = 1e-12
        assert abs(src.value_at(0, dt)) < 1e-10

    def test_sinusoidal_reaches_amplitude(self):
        """After ramp-up, sinusoidal should reach full amplitude."""
        src = SinusoidalSource(x=0, y=0, frequency=1e9, amplitude=2.0)
        dt = 1e-12
        # After ~20 periods at 1 GHz (T=1ns, 20T=20ns, 20000 steps at 1ps)
        late_values = [abs(src.value_at(t, dt)) for t in range(20000, 21000)]
        assert max(late_values) > 1.9  # Should reach near amplitude

    def test_waveform_array(self):
        """Test waveform generation produces correct shape."""
        src = GaussianSource(x=0, y=0, frequency_max=1e9)
        wf = src.waveform(1000, 1e-12)
        assert wf.shape == (1000,)
        assert wf.max() > 0

    def test_modulated_gaussian(self):
        """Modulated Gaussian should have carrier oscillation."""
        src = ModulatedGaussianSource(
            x=0, y=0, frequency=2e9, bandwidth=1e9
        )
        wf = src.waveform(5000, 1e-12)
        # Should have both positive and negative values (oscillation)
        assert wf.min() < 0
        assert wf.max() > 0

    def test_source_position(self):
        src = GaussianSource(x=10, y=20, frequency_max=1e9)
        assert src.position == (10, 20)
        assert src.component == "Ez"


# ============================================================
# Boundary Tests
# ============================================================

class TestBoundaries:
    """Tests for boundary conditions."""

    def _make_config(self, nx=50, ny=50, steps=10):
        grid = GridConfig(nx=nx, ny=ny, dx=1e-3, dy=1e-3)
        return SimulationConfig(grid=grid, total_steps=steps)

    def test_pec_zeros_boundary(self):
        """PEC should zero E-field at boundaries."""
        config = self._make_config()
        grid = Grid2D(config)
        grid.ez[:] = 1.0  # Fill with non-zero

        pec = PEC()
        e_dict = {"Ez": grid.ez}
        h_dict = {"Hx": grid.hx, "Hy": grid.hy}
        pec.apply_e_field(e_dict, h_dict, config)

        assert np.all(grid.ez[0, :] == 0.0)
        assert np.all(grid.ez[-1, :] == 0.0)
        assert np.all(grid.ez[:, 0] == 0.0)
        assert np.all(grid.ez[:, -1] == 0.0)
        assert grid.ez[25, 25] == 1.0  # Interior unchanged

    def test_mur_abc_initializes(self):
        """Mur ABC should initialize without error."""
        config = self._make_config()
        grid = Grid2D(config)
        mur = MurABC()
        e_dict = {"Ez": grid.ez}
        h_dict = {"Hx": grid.hx, "Hy": grid.hy}
        mur.apply_e_field(e_dict, h_dict, config)
        assert mur._initialized

    def test_cpml_initializes(self):
        """CPML should initialize and create psi arrays."""
        config = self._make_config(nx=100, ny=100)
        grid = Grid2D(config)
        cpml = CPML(thickness=10)
        e_dict = {"Ez": grid.ez}
        h_dict = {"Hx": grid.hx, "Hy": grid.hy}
        cpml.apply_h_field(e_dict, h_dict, config)
        assert cpml._initialized
        assert cpml._psi_hyx_xlo.shape == (10, 100)


# ============================================================
# FDTD Solver Tests
# ============================================================

class TestFDTD2D:
    """Tests for the 2D FDTD solver."""

    def _make_solver(self, nx=50, ny=50, steps=100, bc=None, sources=None):
        grid = GridConfig(nx=nx, ny=ny, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, total_steps=steps, courant=0.5)
        if sources is None:
            sources = [GaussianSource(x=nx//2, y=ny//2, frequency_max=5e9)]
        if bc is None:
            bc = [MurABC()]
        return FDTD2D(config=config, sources=sources, boundaries=bc)

    def test_solver_creation(self):
        solver = self._make_solver()
        assert solver.current_step == 0
        assert solver.current_time == 0.0

    def test_single_step(self):
        solver = self._make_solver()
        solver.step()
        assert solver.current_step == 1
        assert solver.current_time > 0

    def test_source_injection(self):
        """After stepping, source point should have non-zero field."""
        solver = self._make_solver(steps=50)
        solver.run(10)
        # Source is at center (25, 25)
        assert solver.get_field("Ez")[25, 25] != 0.0

    def test_field_propagation(self):
        """Field should propagate outward from source."""
        solver = self._make_solver(nx=100, ny=100, steps=50)
        solver.run(50)
        ez = solver.get_field("Ez")
        # Field should exist beyond just the source point
        assert np.any(ez[60, :] != 0.0)

    def test_energy_conservation_pec(self):
        """With PEC (no absorption), total energy should be bounded."""
        solver = self._make_solver(nx=50, ny=50, steps=100, bc=[PEC()])
        solver.run(100)
        ez = solver.get_field("Ez")
        # Energy shouldn't blow up (stability check)
        assert np.all(np.isfinite(ez))
        assert np.abs(ez).max() < 100  # Reasonable bound

    def test_field_recording(self):
        """Test field snapshot recording."""
        grid = GridConfig(nx=50, ny=50, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, total_steps=50, courant=0.5)
        src = GaussianSource(x=25, y=25, frequency_max=5e9)
        solver = FDTD2D(
            config=config, sources=[src], boundaries=[MurABC()],
            record_field="Ez", record_interval=10,
        )
        solver.run(50)
        assert len(solver.field_snapshots) == 5  # 50/10 = 5 snapshots

    def test_probe_recording(self):
        """Test probe point time-series recording."""
        grid = GridConfig(nx=50, ny=50, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, total_steps=20, courant=0.5)
        src = GaussianSource(x=25, y=25, frequency_max=5e9)
        solver = FDTD2D(
            config=config, sources=[src], boundaries=[MurABC()],
            record_field="Ez", probe_points=[(30, 25)],
        )
        solver.run(20)
        assert len(solver.probe_data[(30, 25)]) == 20

    def test_dielectric_slab_slows_wave(self):
        """Wave should travel slower in dielectric (eps_r > 1)."""
        # Run two simulations: free space and with dielectric slab
        grid = GridConfig(nx=100, ny=100, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, total_steps=80, courant=0.5)
        src = GaussianSource(x=20, y=50, frequency_max=3e9)

        # Free space
        solver_free = FDTD2D(
            config=config, sources=[src], boundaries=[MurABC()],
            record_field="Ez", probe_points=[(70, 50)],
        )
        solver_free.run(80)

        # With dielectric slab (eps_r=4 → wave speed halved)
        solver_diel = FDTD2D(
            config=config, sources=[src], boundaries=[MurABC()],
            record_field="Ez", probe_points=[(70, 50)],
        )
        solver_diel.grid.set_material_region(40, 60, 0, 100, eps_r=4.0)
        solver_diel.run(80)

        # The wave in dielectric should arrive later (smaller amplitude at
        # early times at the probe point behind the slab)
        free_data = np.array(solver_free.probe_data[(70, 50)])
        diel_data = np.array(solver_diel.probe_data[(70, 50)])

        # Find time of first significant arrival
        threshold = 0.01 * max(np.abs(free_data).max(), 1e-20)
        if threshold > 0:
            free_arrival = np.argmax(np.abs(free_data) > threshold)
            diel_arrival = np.argmax(np.abs(diel_data) > threshold)
            # Dielectric should delay the wave
            assert diel_arrival >= free_arrival

    def test_cpml_absorbs_waves(self):
        """CPML should absorb outgoing waves (field energy should decay)."""
        grid = GridConfig(nx=80, ny=80, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, total_steps=300, courant=0.5)
        src = GaussianSource(x=40, y=40, frequency_max=3e9)

        solver = FDTD2D(
            config=config, sources=[src], boundaries=[CPML(thickness=10)],
            record_field="Ez", record_interval=50,
        )
        solver.run(300)

        # After the pulse has left, field energy should be very small
        ez = solver.get_field("Ez")
        assert np.all(np.isfinite(ez))  # Stability
        final_energy = np.sum(ez ** 2)
        # Energy of early snapshot (when pulse is active) should be higher
        mid_energy = np.sum(solver.field_snapshots[2] ** 2)
        # If source has finished, final energy should be less than mid
        # (pulse has been absorbed by PML)
        assert final_energy < mid_energy or final_energy < 1e-4

    def test_numerical_stability(self):
        """Simulation should remain stable for many timesteps."""
        solver = self._make_solver(nx=50, ny=50, steps=500)
        solver.run(500)
        ez = solver.get_field("Ez")
        assert np.all(np.isfinite(ez))
        assert np.abs(ez).max() < 1e10  # No blow-up

    def test_solver_tez_mode(self):
        """Test TEz mode simulation runs and propagates fields."""
        grid = GridConfig(nx=50, ny=50, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, mode=SimulationMode.TEZ, total_steps=20, courant=0.5)
        # For TEz, use Hz source
        src = GaussianSource(x=25, y=25, frequency_max=5e9, field_component="Hz")
        solver = FDTD2D(config=config, sources=[src], boundaries=[MurABC()])
        solver.run(20)
        
        hz = solver.get_field("Hz")
        ex = solver.get_field("Ex")
        ey = solver.get_field("Ey")
        
        # Check propagation
        assert np.any(hz[30, :] != 0.0)
        assert np.any(ex != 0.0)
        assert np.any(ey != 0.0)
        
    def test_tez_cpml(self):
        """Test TEz mode with CPML boundary condition."""
        grid = GridConfig(nx=60, ny=60, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, mode=SimulationMode.TEZ, total_steps=100, courant=0.5)
        src = GaussianSource(x=30, y=30, frequency_max=5e9, field_component="Hz")
        
        # Run with CPML
        solver = FDTD2D(config=config, sources=[src], boundaries=[CPML(thickness=8)])
        solver.run(100)
        hz = solver.get_field("Hz")
        
        # Ensure stability
        assert np.all(np.isfinite(hz))

class TestDispersiveMaterials:
    """Tests for Debye and other dispersive material models."""
    
    def test_debye_medium_delays_and_attenuates(self):
        """A wave in a Debye medium should be delayed and attenuated vs free space."""
        from ceep.materials.dispersive import DebyePole
        
        # Free space solver
        grid_fs = GridConfig(nx=100, ny=10, dx=1e-3, dy=1e-3)
        config_fs = SimulationConfig(grid=grid_fs, mode=SimulationMode.TMZ, total_steps=200)
        src_fs = GaussianSource(x=10, y=5, frequency_max=20e9)
        solver_fs = FDTD2D(config=config_fs, sources=[src_fs])
        
        # Dispersive solver
        grid_disp = GridConfig(nx=100, ny=10, dx=1e-3, dy=1e-3)
        config_disp = SimulationConfig(grid=grid_disp, mode=SimulationMode.TMZ, total_steps=200)
        src_disp = GaussianSource(x=10, y=5, frequency_max=20e9)
        solver_disp = FDTD2D(config=config_disp, sources=[src_disp])
        
        # Add Debye pole (water-like relaxation at microwave freq)
        pole = DebyePole(delta_eps=70.0, tau=1e-11)
        # Note: set_material_region automatically sets eps_inf
        solver_disp.grid.set_material_region(20, 80, 0, 10, eps_inf=5.0, debye_poles=[pole])
        
        solver_fs.run(200)
        solver_disp.run(200)
        
        ez_fs = solver_fs.get_field("Ez")
        ez_disp = solver_disp.get_field("Ez")
        
        # Check that the wave in dispersive medium has not traveled as far
        peak_idx_fs = np.argmax(np.abs(ez_fs[:, 5]))
        peak_idx_disp = np.argmax(np.abs(ez_disp[:, 5]))
        
        # FS should have traveled further than dispersive
        assert peak_idx_fs > peak_idx_disp
        
        # The wave should be attenuated compared to free space inside the region
        max_fs = np.max(np.abs(ez_fs[25:80, 5]))
        max_disp = np.max(np.abs(ez_disp[25:80, 5]))
        
        assert max_disp < max_fs

    def test_drude_medium_attenuates(self):
        """A wave in a Drude medium (metal) should experience severe attenuation (skin depth)."""
        from ceep.materials.dispersive import DrudePole
        
        grid = GridConfig(nx=100, ny=10, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=200)
        
        # Free space
        solver_fs = FDTD2D(config=config, sources=[GaussianSource(x=10, y=5, frequency_max=20e9)])
        solver_fs.run(200)
        
        # Drude metal
        solver_drude = FDTD2D(config=config, sources=[GaussianSource(x=10, y=5, frequency_max=20e9)])
        # High plasma frequency, low collision -> highly reflective/attenuative metal
        pole = DrudePole(omega_p=1e14, gamma=1e12)
        solver_drude.grid.set_material_region(20, 80, 0, 10, debye_poles=[pole])
        solver_drude.run(200)
        
        ez_fs = solver_fs.get_field("Ez")
        ez_drude = solver_drude.get_field("Ez")
        
        # Wave transmitted into the metal should be near zero (most is reflected)
        max_fs = np.max(np.abs(ez_fs[25:80, 5]))
        max_drude = np.max(np.abs(ez_drude[25:80, 5]))
        
        assert max_drude < 1e-3 * max_fs  # Massive attenuation
        
    def test_lorentz_medium_stability(self):
        """Test that Lorentz poles do not cause instability."""
        from ceep.materials.dispersive import LorentzPole
        
        grid = GridConfig(nx=100, ny=10, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=200)
        
        solver = FDTD2D(config=config, sources=[GaussianSource(x=10, y=5, frequency_max=20e9)])
        pole = LorentzPole(delta_eps=2.0, omega_0=2*np.pi*10e9, delta=1e9)
        solver.grid.set_material_region(20, 80, 0, 10, debye_poles=[pole])
        
        solver.run(200)
        ez = solver.get_field("Ez")
        
        # Should remain stable
        assert np.all(np.isfinite(ez))
        assert np.max(np.abs(ez)) < 10.0


class TestTFSF:
    """Tests for Total-Field/Scattered-Field plane wave injection."""
    
    def test_plane_wave_scattered_field_is_zero_in_free_space(self):
        """In free space, fields outside the TF/SF boundary should remain near zero."""
        from ceep.sources.plane_wave import PlaneWaveSource
        
        grid = GridConfig(nx=100, ny=100, dx=1e-3, dy=1e-3)
        config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=500)
        
        # TF/SF boundary from x=20..80, y=20..80
        src = PlaneWaveSource(
            x_start=20, x_end=80,
            y_start=20, y_end=80,
            frequency_max=10e9, amplitude=1.0,
            delay_factor=2.0
        )
        
        solver = FDTD2D(config=config, sources=[src])
        solver.run(500)
        
        ez = solver.get_field("Ez")
        
        # Inside TF/SF (total field), wave should be present
        assert np.max(np.abs(ez[30:70, 30:70])) > 0.05
        
        # Outside TF/SF (scattered field), wave should be zero (ideally)
        # Due to numerical precision, it might not be exactly 0, but very small
        max_scattered = max(
            np.max(np.abs(ez[0:15, :])),
            np.max(np.abs(ez[85:100, :])),
            np.max(np.abs(ez[:, 0:15])),
            np.max(np.abs(ez[:, 85:100]))
        )
        
        assert max_scattered < 1e-10

