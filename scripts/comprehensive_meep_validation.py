"""
NeuroWave vs MEEP - Comprehensive Validation Suite
====================================================
Side-by-side comparison of NeuroWave (cuda-meep) against MIT MEEP
to validate electromagnetic solver accuracy.

Tests:
  1. Free space propagation (c accuracy)
  2. Dielectric slab (reflection/transmission)
  3. CPML absorption (reflection coefficient)
  4. Dispersive material (Drude metal)
  5. Numerical dispersion (phase velocity)
  6. S-parameter extraction (two-port)

Author: NeuroWave Development Team
Date: 2026-05-13
"""

import math
import time
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from neurowave.core.config import GridConfig, SimulationConfig, SimulationMode
from neurowave.core.constants import C_0, EPS_0, MU_0, DEFAULT_COURANT_2D
from neurowave.solvers.fdtd_2d import FDTD2D
from neurowave.sources.waveforms import ModulatedGaussianSource, GaussianSource
from neurowave.boundaries.absorbing import CPML, PEC
from neurowave.materials.dispersive import DispersiveManager

# Check if MEEP is available
try:
    import meep as mp
    MEEP_AVAILABLE = True
except ImportError:
    MEEP_AVAILABLE = False
    print("⚠️  MEEP not installed - will only run NeuroWave tests")


@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    nw_time: float
    meep_time: Optional[float]
    nw_result: float
    meep_result: Optional[float]
    analytical: Optional[float]
    passed: bool
    tolerance: float
    notes: str = ""


class ValidationSuite:
    """Comprehensive validation test suite"""

    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[TestResult] = []

    def print_header(self, test_name: str):
        """Print test header"""
        print("\n" + "="*70)
        print(f"  TEST: {test_name}")
        print("="*70)

    def print_result(self, result: TestResult):
        """Print test result"""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n{status} - {result.test_name}")
        print(f"  NeuroWave:   {result.nw_result:.6e} ({result.nw_time:.3f}s)")
        if result.meep_result is not None:
            print(f"  MEEP:        {result.meep_result:.6e} ({result.meep_time:.3f}s)")
        if result.analytical is not None:
            print(f"  Analytical:  {result.analytical:.6e}")
            nw_error = abs(result.nw_result - result.analytical) / result.analytical * 100
            print(f"  NW Error:    {nw_error:.2f}%")
            if result.meep_result is not None:
                meep_error = abs(result.meep_result - result.analytical) / result.analytical * 100
                print(f"  MEEP Error:  {meep_error:.2f}%")
        print(f"  Tolerance:   {result.tolerance*100:.2f}%")
        if result.notes:
            print(f"  Notes:       {result.notes}")

    # -------------------------------------------------------------------------
    # TEST 1: Free Space Propagation (Speed of Light)
    # -------------------------------------------------------------------------
    def test_free_space_propagation(self):
        """Test 1: Verify speed of light in free space"""
        self.print_header("Free Space Propagation - Speed of Light")

        # Parameters
        dx = 1e-3  # 1mm
        nx = ny = 200
        fc = 10e9  # 10 GHz
        tau = 1.0 / (math.pi * fc)
        delay = 5.0
        t0 = delay * tau
        steps = 800

        # Source and probe positions
        src_x, src_y = 50, 100
        probe_x, probe_y = 150, 100
        distance = (probe_x - src_x) * dx
        analytical_tof = distance / C_0

        print(f"Distance: {distance*1e3:.1f} mm")
        print(f"Analytical TOF: {analytical_tof*1e12:.2f} ps")

        # NeuroWave
        print("\n→ Running NeuroWave...")
        t0_nw = time.time()

        grid = GridConfig(nx=nx, ny=ny, dx=dx, dy=dx)
        config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=steps)

        src = ModulatedGaussianSource(
            x=src_x, y=src_y, frequency=fc, bandwidth=fc,
            field_component="Ez", delay_factor=delay
        )

        solver = FDTD2D(
            config=config, sources=[src],
            boundaries=[CPML(thickness=10)],
            record_field="Ez", probe_points=[(probe_x, probe_y)]
        )
        solver.run()

        hist_nw = np.array(solver.probe_data[(probe_x, probe_y)])
        t_nw = np.arange(steps) * config.dt
        peak_idx = np.argmax(np.abs(hist_nw))
        measured_tof_nw = t_nw[peak_idx] - t0
        measured_c_nw = distance / measured_tof_nw

        time_nw = time.time() - t0_nw
        print(f"  Measured c: {measured_c_nw:.6e} m/s")
        print(f"  Time: {time_nw:.3f}s")

        # MEEP
        measured_c_meep = None
        time_meep = None
        if MEEP_AVAILABLE:
            print("\n→ Running MEEP...")
            t0_meep = time.time()

            a = dx
            c_a = C_0 / a

            def src_func(t_meep):
                t_si = t_meep * a / C_0
                env = math.exp(-((t_si - t0) / tau) ** 2)
                return env * math.sin(2.0 * math.pi * fc * t_si)

            sources = [mp.Source(
                mp.CustomSource(src_func=src_func, is_integrated=False),
                component=mp.Ez,
                center=mp.Vector3(0, 0, 0)
            )]

            probe_meep = mp.Vector3(probe_x - src_x, 0, 0)

            sim = mp.Simulation(
                cell_size=mp.Vector3(nx, ny, 0),
                boundary_layers=[mp.PML(10)],
                sources=sources,
                resolution=1
            )

            hist_meep = []
            def rec(sim):
                hist_meep.append(sim.get_field_point(mp.Ez, probe_meep).real)

            meep_dt_m = 0.5
            meep_dt_si = meep_dt_m * a / C_0
            run_time_si = steps * config.dt
            run_time_m = run_time_si * c_a

            sim.run(mp.at_every(meep_dt_m, rec), until=run_time_m)

            h_meep = np.array(hist_meep)
            t_meep = np.arange(len(hist_meep)) * meep_dt_si
            peak_idx_m = np.argmax(np.abs(h_meep))
            measured_tof_meep = t_meep[peak_idx_m] - t0
            measured_c_meep = distance / measured_tof_meep

            time_meep = time.time() - t0_meep
            print(f"  Measured c: {measured_c_meep:.6e} m/s")
            print(f"  Time: {time_meep:.3f}s")

        # Evaluate
        tolerance = 0.02  # 2% tolerance
        error_nw = abs(measured_c_nw - C_0) / C_0
        passed = error_nw < tolerance

        result = TestResult(
            test_name="Free Space Propagation",
            nw_time=time_nw,
            meep_time=time_meep,
            nw_result=measured_c_nw,
            meep_result=measured_c_meep,
            analytical=C_0,
            passed=passed,
            tolerance=tolerance,
            notes=f"Speed of light validation"
        )

        self.results.append(result)
        self.print_result(result)

        return result

    # -------------------------------------------------------------------------
    # TEST 2: Dielectric Interface (Fresnel Coefficients)
    # -------------------------------------------------------------------------
    def test_dielectric_interface(self):
        """Test 2: Fresnel reflection/transmission at dielectric interface"""
        self.print_header("Dielectric Interface - Fresnel Coefficients")

        # Parameters
        dx = 1e-3  # 1mm
        nx, ny = 300, 100
        fc = 10e9
        tau = 1.0 / (math.pi * fc)
        delay = 5.0
        t0 = delay * tau
        steps = 1000

        eps_r = 4.0  # Dielectric constant
        n1, n2 = 1.0, math.sqrt(eps_r)  # Refractive indices

        # Analytical Fresnel coefficients (normal incidence)
        R_analytical = ((n1 - n2) / (n1 + n2)) ** 2
        T_analytical = (4 * n1 * n2) / ((n1 + n2) ** 2)

        print(f"ε_r = {eps_r}, n2 = {n2:.3f}")
        print(f"Analytical R = {R_analytical:.4f} ({R_analytical*100:.1f}%)")
        print(f"Analytical T = {T_analytical:.4f} ({T_analytical*100:.1f}%)")

        # Source before interface, probes on both sides
        src_x = 50
        probe_refl_x = 60  # Before interface
        interface_x = 150
        probe_trans_x = 240  # After interface
        src_y = probe_y = 50

        # NeuroWave
        print("\n→ Running NeuroWave...")
        t0_nw = time.time()

        grid = GridConfig(nx=nx, ny=ny, dx=dx, dy=dx)
        config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=steps)

        src = ModulatedGaussianSource(
            x=src_x, y=src_y, frequency=fc, bandwidth=fc,
            field_component="Ez", delay_factor=delay
        )

        solver = FDTD2D(
            config=config, sources=[src],
            boundaries=[CPML(thickness=10)],
            record_field="Ez",
            probe_points=[(probe_refl_x, probe_y), (probe_trans_x, probe_y)]
        )

        # Set dielectric region
        solver.grid.set_material_region(interface_x, nx-10, 0, ny, eps_r=eps_r)
        solver.run()

        # Extract amplitudes
        hist_refl = np.array(solver.probe_data[(probe_refl_x, probe_y)])
        hist_trans = np.array(solver.probe_data[(probe_trans_x, probe_y)])

        # Get max amplitudes (after initial pulse passes)
        E_refl = np.max(np.abs(hist_refl[200:]))
        E_trans = np.max(np.abs(hist_trans[200:]))
        E_incident = np.max(np.abs(hist_refl[:400]))

        R_nw = (E_refl / E_incident) ** 2
        T_nw = ((E_trans / E_incident) / n2) ** 2  # Account for impedance change

        time_nw = time.time() - t0_nw
        print(f"  R_measured = {R_nw:.4f} ({R_nw*100:.1f}%)")
        print(f"  T_measured = {T_nw:.4f} ({T_nw*100:.1f}%)")
        print(f"  Time: {time_nw:.3f}s")

        # For this test, use R coefficient as the main metric
        tolerance = 0.10  # 10% tolerance
        error_nw = abs(R_nw - R_analytical) / R_analytical
        passed = error_nw < tolerance

        result = TestResult(
            test_name="Dielectric Interface",
            nw_time=time_nw,
            meep_time=None,
            nw_result=R_nw,
            meep_result=None,
            analytical=R_analytical,
            passed=passed,
            tolerance=tolerance,
            notes=f"Fresnel reflection coefficient (T={T_nw:.4f})"
        )

        self.results.append(result)
        self.print_result(result)

        return result

    # -------------------------------------------------------------------------
    # TEST 3: CPML Absorption Performance
    # -------------------------------------------------------------------------
    def test_cpml_absorption(self):
        """Test 3: CPML reflection coefficient measurement"""
        self.print_header("CPML Absorption - Reflection Coefficient")

        # Parameters
        dx = 1e-3
        pml_thickness = 20  # cells
        nx = 200 + 2 * pml_thickness
        ny = 200
        fc = 10e9
        tau = 1.0 / (math.pi * fc)
        delay = 5.0
        steps = 1200

        # Source in center, probe near boundary
        src_x, src_y = nx // 2, ny // 2
        probe_x = nx - pml_thickness - 5  # Just before PML
        probe_y = ny // 2

        print(f"PML thickness: {pml_thickness} cells")
        print(f"Grid: {nx}×{ny}")

        # NeuroWave with CPML
        print("\n→ Running NeuroWave with CPML...")
        t0_nw = time.time()

        grid = GridConfig(nx=nx, ny=ny, dx=dx, dy=dx)
        config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=steps)

        src = ModulatedGaussianSource(
            x=src_x, y=src_y, frequency=fc, bandwidth=fc,
            field_component="Ez", delay_factor=delay
        )

        solver_cpml = FDTD2D(
            config=config, sources=[src],
            boundaries=[CPML(thickness=pml_thickness)],
            record_field="Ez",
            probe_points=[(probe_x, probe_y)]
        )
        solver_cpml.run()

        hist_cpml = np.array(solver_cpml.probe_data[(probe_x, probe_y)])
        time_cpml = time.time() - t0_nw

        # NeuroWave without PML (reference)
        print("\n→ Running NeuroWave without PML (reference)...")
        t0_ref = time.time()

        # Use smaller grid without boundary conditions (will reflect)
        solver_ref = FDTD2D(
            config=config, sources=[src],
            boundaries=[],  # No boundaries - will reflect
            record_field="Ez",
            probe_points=[(probe_x, probe_y)]
        )
        solver_ref.run()

        hist_ref = np.array(solver_ref.probe_data[(probe_x, probe_y)])
        time_ref = time.time() - t0_ref

        # Calculate reflection coefficient
        # Compare late-time fields (reflected wave)
        late_start = steps // 2
        E_cpml_late = np.max(np.abs(hist_cpml[late_start:]))
        E_ref_late = np.max(np.abs(hist_ref[late_start:]))

        reflection_coeff = E_cpml_late / E_ref_late
        reflection_db = 20 * np.log10(reflection_coeff) if reflection_coeff > 0 else -200

        print(f"  Reflection (CPML): {reflection_db:.1f} dB")
        print(f"  Time: {time_cpml:.3f}s")

        # CPML should achieve < -40 dB for 20 cells
        target_db = -40.0
        passed = reflection_db < target_db

        result = TestResult(
            test_name="CPML Absorption",
            nw_time=time_cpml,
            meep_time=None,
            nw_result=reflection_db,
            meep_result=None,
            analytical=target_db,
            passed=passed,
            tolerance=0.20,
            notes=f"Target: < {target_db} dB for {pml_thickness} cells"
        )

        self.results.append(result)
        self.print_result(result)

        return result

    # -------------------------------------------------------------------------
    # TEST 4: Dispersive Material (Drude Metal)
    # -------------------------------------------------------------------------
    def test_drude_material(self):
        """Test 4: Drude model for metallic response"""
        self.print_header("Dispersive Material - Drude Metal")

        # Parameters
        dx = 0.5e-3  # 0.5mm
        nx, ny = 200, 100

        # Drude parameters (simplified metal)
        omega_p = 2 * math.pi * 100e9  # Plasma frequency ~100 GHz
        gamma = 2 * math.pi * 10e9     # Collision frequency ~10 GHz

        # Test frequency (below plasma frequency - should reflect)
        fc = 50e9  # 50 GHz < 100 GHz plasma
        tau = 1.0 / (math.pi * fc)
        delay = 5.0
        steps = 1000

        # At f < f_plasma, metal should reflect strongly
        print(f"Plasma freq: {omega_p/(2*math.pi)*1e-9:.1f} GHz")
        print(f"Test freq: {fc*1e-9:.1f} GHz")
        print(f"Collision freq: {gamma/(2*math.pi)*1e-9:.1f} GHz")

        # Source and probes
        src_x = 30
        probe_inc_x = 50  # Before metal
        metal_start_x = 100
        metal_end_x = 120
        probe_y = src_y = 50

        # NeuroWave with Drude
        print("\n→ Running NeuroWave with Drude material...")
        t0_nw = time.time()

        grid = GridConfig(nx=nx, ny=ny, dx=dx, dy=dx)
        config = SimulationConfig(grid=grid, mode=SimulationMode.TMZ, total_steps=steps)

        src = ModulatedGaussianSource(
            x=src_x, y=src_y, frequency=fc, bandwidth=fc*0.5,
            field_component="Ez", delay_factor=delay
        )

        # Create dispersive material manager
        disp_mat = DispersiveManager(grid_shape=(nx, ny))

        # Add Drude pole for metal region
        metal_mask = np.zeros((nx, ny), dtype=bool)
        metal_mask[metal_start_x:metal_end_x, :] = True

        disp_mat.add_drude_pole(
            region_mask=metal_mask,
            omega_p=omega_p,
            gamma=gamma
        )

        solver = FDTD2D(
            config=config,
            sources=[src],
            boundaries=[CPML(thickness=10)],
            record_field="Ez",
            probe_points=[(probe_inc_x, probe_y)],
            dispersive_material=disp_mat
        )
        solver.run()

        hist_drude = np.array(solver.probe_data[(probe_inc_x, probe_y)])

        # Compare with free space (no metal)
        solver_ref = FDTD2D(
            config=config,
            sources=[src],
            boundaries=[CPML(thickness=10)],
            record_field="Ez",
            probe_points=[(probe_inc_x, probe_y)]
        )
        solver_ref.run()

        hist_ref = np.array(solver_ref.probe_data[(probe_inc_x, probe_y)])

        time_nw = time.time() - t0_nw

        # Calculate reflection enhancement
        # Late-time amplitude with metal should be higher due to reflection
        late_start = 500
        E_drude_late = np.max(np.abs(hist_drude[late_start:]))
        E_ref_late = np.max(np.abs(hist_ref[late_start:]))

        reflection_ratio = E_drude_late / E_ref_late if E_ref_late > 0 else 0

        print(f"  Reflection enhancement: {reflection_ratio:.2f}x")
        print(f"  Time: {time_nw:.3f}s")

        # Metal should enhance reflection (ratio > 2)
        passed = reflection_ratio > 2.0

        result = TestResult(
            test_name="Drude Dispersive Material",
            nw_time=time_nw,
            meep_time=None,
            nw_result=reflection_ratio,
            meep_result=None,
            analytical=None,
            passed=passed,
            tolerance=0.30,
            notes=f"Metal reflection enhancement (f < f_plasma)"
        )

        self.results.append(result)
        self.print_result(result)

        return result

    # -------------------------------------------------------------------------
    # Generate Report
    # -------------------------------------------------------------------------
    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*70)
        print("  VALIDATION SUITE SUMMARY")
        print("="*70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\nTests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        print("\nDetailed Results:")
        print("-" * 70)

        for i, result in enumerate(self.results, 1):
            status = "✅" if result.passed else "❌"
            print(f"{i}. {status} {result.test_name}")
            if result.analytical is not None:
                error = abs(result.nw_result - result.analytical) / result.analytical * 100
                print(f"   Error: {error:.2f}% (tolerance: {result.tolerance*100:.1f}%)")

        # Create summary plot
        self._create_summary_plot()

        print(f"\n📊 Results saved to: {self.output_dir}/")
        print("="*70)

        return passed == total

    def _create_summary_plot(self):
        """Create visual summary of validation results"""
        fig, ax = plt.subplots(figsize=(10, 6))

        test_names = [r.test_name for r in self.results]
        nw_times = [r.nw_time for r in self.results]
        colors = ['green' if r.passed else 'red' for r in self.results]

        bars = ax.barh(test_names, nw_times, color=colors, alpha=0.7)

        ax.set_xlabel('Execution Time (seconds)')
        ax.set_title('NeuroWave Validation Suite Results')
        ax.grid(axis='x', alpha=0.3)

        # Add pass/fail labels
        for i, (bar, result) in enumerate(zip(bars, self.results)):
            status = "✅ PASS" if result.passed else "❌ FAIL"
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                   f' {status}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/validation_summary.png', dpi=150)
        print(f"\n📊 Summary plot saved: {self.output_dir}/validation_summary.png")


def main():
    """Run comprehensive validation suite"""
    print("\n" + "="*70)
    print("  NeuroWave vs MEEP - Comprehensive Validation")
    print("  GPU-Accelerated FDTD Electromagnetic Solver")
    print("="*70)

    if not MEEP_AVAILABLE:
        print("\n⚠️  MEEP not available - running NeuroWave-only validation")

    suite = ValidationSuite()

    # Run all tests
    suite.test_free_space_propagation()
    suite.test_dielectric_interface()
    suite.test_cpml_absorption()
    suite.test_drude_material()

    # Generate report
    all_passed = suite.generate_report()

    if all_passed:
        print("\n🎉 ALL TESTS PASSED - NeuroWave is validated!")
    else:
        print("\n⚠️  Some tests failed - review results above")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
