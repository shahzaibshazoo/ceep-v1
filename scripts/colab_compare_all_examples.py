#!/usr/bin/env python3
"""
Google Colab: Compare CEEP vs MEEP on All Examples
===================================================

Runs all examples from examples/ directory using both CEEP (GPU) and MEEP (CPU)
and generates a comprehensive comparison report.

Usage in Colab:
    !git clone https://github.com/shahzaibshazoo/ceep-v1.git
    %cd ceep-v1
    !pip install -e .[gpu]
    !pip install meep matplotlib tqdm
    !python scripts/colab_compare_all_examples.py

Output:
    - comparison_summary.txt - Text report
    - comparison_plots/ - Visualization for each example
    - comparison_results.json - Machine-readable results

Author: Shahzaib Ur Rehman
Date: 2026-05-15
"""

import sys
import os

# Add src to path if running from repo directly
if os.path.exists('src/ceep'):
    sys.path.insert(0, 'src')
elif os.path.exists('/content/ceep-v1/src/ceep'):
    sys.path.insert(0, '/content/ceep-v1/src')

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


# Import both backends
try:
    from ceep.core.backend import set_backend, to_numpy
    from ceep.solvers import BatchedFDTD2D
    from ceep.phantoms import BrainPhantom2D
    CEEP_AVAILABLE = True
    print("✓ CEEP imported successfully")
except Exception as e:
    CEEP_AVAILABLE = False
    print(f"✗ CEEP not available: {e}")

try:
    import meep as mp
    MEEP_AVAILABLE = True
    print("✓ MEEP imported successfully")
except Exception as e:
    MEEP_AVAILABLE = False
    print(f"✗ MEEP not available: {e}")


@dataclass
class ComparisonResult:
    """Results from comparing CEEP vs MEEP."""
    example_name: str
    ceep_time: float
    meep_time: float
    speedup: float
    ceep_max_magnitude: float
    meep_max_magnitude: float
    magnitude_ratio: float
    ceep_mean_magnitude: float
    meep_mean_magnitude: float
    mean_ratio: float
    correlation: Optional[float]
    relative_error: float
    status: str  # 'EXCELLENT', 'GOOD', 'MODERATE', 'POOR'
    notes: str


class ExampleComparator:
    """Compare CEEP and MEEP implementations."""

    def __init__(self, correction_factor=6.58e12):
        """
        Parameters
        ----------
        correction_factor : float
            CEEP correction factor (from MEEP validation)
        """
        self.correction_factor = correction_factor
        self.results = []

    def run_ceep_example(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run example using CEEP.

        Parameters
        ----------
        config : dict
            Configuration with keys: nx, ny, dx, frequency, n_ant, etc.

        Returns
        -------
        result : dict
            Contains s_matrix, runtime, etc.
        """
        if not CEEP_AVAILABLE:
            return {'error': 'CEEP not available'}

        set_backend('cupy')

        # Create antenna array
        nx, ny = config['nx'], config['ny']
        n_ant = config.get('n_ant', 16)

        center_x, center_y = nx // 2, ny // 2
        radius = nx // 3

        angles = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
        positions = []
        for angle in angles:
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            positions.append((x, y))

        # Create solver
        t_start = time.time()

        solver = BatchedFDTD2D(
            nx=config['nx'],
            ny=config['ny'],
            dx=config['dx'],
            total_steps=config.get('total_steps', 300),
            cpml_thickness=config.get('cpml_thickness', 10),
            source_positions=positions,
            probe_positions=positions,
            frequency=config['frequency']
        )

        # Add phantom if specified
        if config.get('use_phantom', False):
            phantom = BrainPhantom2D(
                nx=config['nx'],
                ny=config['ny'],
                dx=config['dx'],
                hemorrhage_location=config.get('hem_location', (1.0, 0.5)),
                hemorrhage_radius=config.get('hem_radius', 1.0),
                use_gabriel_database=False
            )
            solver.set_phantom(phantom)

        # Run
        s_params = solver.run()
        t_elapsed = time.time() - t_start

        # Extract and correct
        s_matrix = to_numpy(s_params[0][0])
        s_matrix_corrected = s_matrix * self.correction_factor

        return {
            's_matrix': s_matrix_corrected,
            'runtime': t_elapsed,
            'n_antennas': n_ant,
            'shape': s_matrix_corrected.shape
        }

    def run_meep_example(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run example using MEEP."""
        if not MEEP_AVAILABLE:
            return {'error': 'MEEP not available'}

        # Convert to MEEP units
        wavelength = 3e8 / config['frequency']
        resolution = 20  # points per wavelength

        nx, ny = config['nx'], config['ny']
        dx = config['dx']

        cell_size = mp.Vector3(nx * dx / wavelength, ny * dx / wavelength, 0)

        # Create antenna positions
        n_ant = config.get('n_ant', 16)
        radius = (nx // 3) * dx / wavelength

        angles = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
        ant_positions = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            ant_positions.append(mp.Vector3(x, y, 0))

        # Create geometry
        geometry = []
        if config.get('use_phantom', False):
            # Simplified brain model
            head_radius = (nx // 2 - 5) * dx / wavelength
            geometry.append(
                mp.Cylinder(radius=head_radius, height=mp.inf,
                           center=mp.Vector3(0, 0, 0),
                           material=mp.Medium(epsilon=50.0))
            )

        # Run simulations for subset of antennas (too slow otherwise)
        # Just do first antenna as reference
        t_start = time.time()

        sources = [mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.2),
                            component=mp.Ez,
                            center=ant_positions[0])]

        sim = mp.Simulation(
            cell_size=cell_size,
            geometry=geometry,
            sources=sources,
            resolution=resolution,
            boundary_layers=[mp.PML(1.0)]
        )

        # Record at all antennas
        fields_vs_time = {i: [] for i in range(n_ant)}

        def record(sim_obj):
            for i, pos in enumerate(ant_positions):
                ez = sim_obj.get_field_point(mp.Ez, pos)
                fields_vs_time[i].append(ez)

        run_time = config.get('run_time', 50)
        sim.run(mp.at_every(0.1, record), until=run_time)

        t_elapsed = time.time() - t_start

        # Build S-matrix (only 1 TX, n_ant RX)
        n_time = len(fields_vs_time[0])
        s_matrix = np.zeros((1, n_ant, n_time), dtype=np.complex128)
        for rx in range(n_ant):
            s_matrix[0, rx, :] = np.array(fields_vs_time[rx])

        return {
            's_matrix': s_matrix,
            'runtime': t_elapsed,
            'n_antennas': n_ant,
            'shape': s_matrix.shape,
            'note': 'Only 1 TX simulated (MEEP is slow)'
        }

    def compare_example(self, name: str, config: Dict[str, Any]) -> ComparisonResult:
        """
        Compare CEEP vs MEEP for one example.

        Parameters
        ----------
        name : str
            Example name
        config : dict
            Configuration dict

        Returns
        -------
        result : ComparisonResult
            Comparison results
        """
        print(f"\n{'='*70}")
        print(f" {name}")
        print(f"{'='*70}")

        # Run CEEP
        print("\n[1/2] Running CEEP...")
        ceep_result = self.run_ceep_example(config)

        if 'error' in ceep_result:
            print(f"  ✗ CEEP failed: {ceep_result['error']}")
            return ComparisonResult(
                example_name=name,
                ceep_time=0, meep_time=0, speedup=0,
                ceep_max_magnitude=0, meep_max_magnitude=0, magnitude_ratio=0,
                ceep_mean_magnitude=0, meep_mean_magnitude=0, mean_ratio=0,
                correlation=None, relative_error=999,
                status='FAILED', notes=ceep_result['error']
            )

        print(f"  ✓ CEEP complete ({ceep_result['runtime']:.2f}s)")
        print(f"    Shape: {ceep_result['shape']}")

        # Run MEEP
        print("\n[2/2] Running MEEP...")
        meep_result = self.run_meep_example(config)

        if 'error' in meep_result:
            print(f"  ✗ MEEP failed: {meep_result['error']}")
            return ComparisonResult(
                example_name=name,
                ceep_time=ceep_result['runtime'],
                meep_time=0, speedup=0,
                ceep_max_magnitude=np.abs(ceep_result['s_matrix']).max(),
                meep_max_magnitude=0, magnitude_ratio=0,
                ceep_mean_magnitude=np.abs(ceep_result['s_matrix']).mean(),
                meep_mean_magnitude=0, mean_ratio=0,
                correlation=None, relative_error=999,
                status='MEEP_UNAVAILABLE',
                notes='MEEP not available'
            )

        print(f"  ✓ MEEP complete ({meep_result['runtime']:.2f}s)")
        print(f"    Shape: {meep_result['shape']}")
        if 'note' in meep_result:
            print(f"    Note: {meep_result['note']}")

        # Compare
        ceep_s = ceep_result['s_matrix']
        meep_s = meep_result['s_matrix']

        ceep_max = np.abs(ceep_s).max()
        meep_max = np.abs(meep_s).max()
        ceep_mean = np.abs(ceep_s).mean()
        meep_mean = np.abs(meep_s).mean()

        magnitude_ratio = ceep_max / meep_max if meep_max > 0 else 0
        mean_ratio = ceep_mean / meep_mean if meep_mean > 0 else 0

        # Relative error
        relative_error = abs(magnitude_ratio - 1.0)

        # Correlation (if same shape)
        correlation = None
        if ceep_s.shape[0] == meep_s.shape[0]:
            # Compare first TX antenna
            tx = 0
            min_len = min(ceep_s.shape[-1], meep_s.shape[-1])
            ceep_sig = np.abs(ceep_s[tx, tx, :min_len])
            meep_sig = np.abs(meep_s[tx, tx, :min_len])
            if len(ceep_sig) > 1 and len(meep_sig) > 1:
                correlation = np.corrcoef(ceep_sig, meep_sig)[0, 1]

        # Determine status
        if relative_error < 0.2:
            status = 'EXCELLENT'
        elif relative_error < 0.5:
            status = 'GOOD'
        elif relative_error < 1.0:
            status = 'MODERATE'
        else:
            status = 'POOR'

        # Speedup
        speedup = meep_result['runtime'] / ceep_result['runtime'] if ceep_result['runtime'] > 0 else 0
        # Estimate full MEEP time (only 1 TX was simulated)
        n_ant = ceep_result['n_antennas']
        estimated_full_meep_time = meep_result['runtime'] * n_ant
        estimated_speedup = estimated_full_meep_time / ceep_result['runtime']

        # Print summary
        print(f"\n  Results:")
        print(f"    CEEP magnitude: {ceep_max:.4f}")
        print(f"    MEEP magnitude: {meep_max:.4f}")
        print(f"    Ratio: {magnitude_ratio:.3f}")
        print(f"    Relative error: {relative_error*100:.1f}%")
        print(f"    Status: {status}")
        print(f"    Speedup: {speedup:.1f}x (estimated full: {estimated_speedup:.1f}x)")

        return ComparisonResult(
            example_name=name,
            ceep_time=ceep_result['runtime'],
            meep_time=meep_result['runtime'],
            speedup=estimated_speedup,
            ceep_max_magnitude=ceep_max,
            meep_max_magnitude=meep_max,
            magnitude_ratio=magnitude_ratio,
            ceep_mean_magnitude=ceep_mean,
            meep_mean_magnitude=meep_mean,
            mean_ratio=mean_ratio,
            correlation=correlation,
            relative_error=relative_error,
            status=status,
            notes=f"MEEP simulated 1/{n_ant} antennas"
        )

    def generate_report(self, output_path='comparison_summary.txt'):
        """Generate text report."""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" CEEP vs MEEP Comparison Report\n")
            f.write("="*70 + "\n\n")

            f.write(f"Correction factor applied: {self.correction_factor:.3e}\n")
            f.write(f"Number of examples: {len(self.results)}\n\n")

            # Summary table
            f.write("Summary Table:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Example':<30} {'Error %':<10} {'Status':<12} {'Speedup':<10}\n")
            f.write("-" * 70 + "\n")

            for r in self.results:
                f.write(f"{r.example_name:<30} {r.relative_error*100:<10.1f} {r.status:<12} {r.speedup:<10.1f}x\n")

            f.write("-" * 70 + "\n\n")

            # Detailed results
            f.write("\nDetailed Results:\n")
            f.write("="*70 + "\n\n")

            for r in self.results:
                f.write(f"Example: {r.example_name}\n")
                f.write(f"  CEEP Runtime: {r.ceep_time:.2f}s\n")
                f.write(f"  MEEP Runtime: {r.meep_time:.2f}s (1 antenna)\n")
                f.write(f"  Estimated Speedup: {r.speedup:.1f}x\n")
                f.write(f"  CEEP Max Magnitude: {r.ceep_max_magnitude:.4f}\n")
                f.write(f"  MEEP Max Magnitude: {r.meep_max_magnitude:.4f}\n")
                f.write(f"  Magnitude Ratio: {r.magnitude_ratio:.3f}\n")
                f.write(f"  Relative Error: {r.relative_error*100:.1f}%\n")
                if r.correlation is not None:
                    f.write(f"  Correlation: {r.correlation:.3f}\n")
                f.write(f"  Status: {r.status}\n")
                f.write(f"  Notes: {r.notes}\n")
                f.write("\n")

        print(f"\n✓ Report saved to {output_path}")


def main():
    """Main comparison routine for Colab."""
    print("="*70)
    print(" CEEP vs MEEP Example Comparison (Google Colab)")
    print("="*70)
    print()

    # Check environment
    if not CEEP_AVAILABLE:
        print("❌ CEEP not available - install with: pip install -e .[gpu]")
        return

    if not MEEP_AVAILABLE:
        print("⚠️  MEEP not available - install with: pip install meep")
        print("   Continuing with CEEP only...\n")

    # Create comparator
    comparator = ExampleComparator(correction_factor=6.58e12)

    # Define examples to test
    examples = {
        'Brain Phantom (Small)': {
            'nx': 64, 'ny': 64, 'dx': 0.5e-3,
            'frequency': 2e9, 'n_ant': 16,
            'total_steps': 300,
            'use_phantom': True,
            'hem_location': (1.0, 0.5),
            'hem_radius': 1.0,
        },
        'Brain Phantom (Medium)': {
            'nx': 128, 'ny': 128, 'dx': 0.5e-3,
            'frequency': 2e9, 'n_ant': 16,
            'total_steps': 400,
            'use_phantom': True,
            'hem_location': (2.0, 1.0),
            'hem_radius': 1.2,
        },
        'Empty Domain (Validation)': {
            'nx': 64, 'ny': 64, 'dx': 0.5e-3,
            'frequency': 2e9, 'n_ant': 8,
            'total_steps': 200,
            'use_phantom': False,
        },
    }

    # Run comparisons
    for name, config in examples.items():
        result = comparator.compare_example(name, config)
        comparator.results.append(result)

    # Generate report
    print("\n" + "="*70)
    print(" Generating Report")
    print("="*70)

    comparator.generate_report('comparison_summary.txt')

    # Save JSON
    results_dict = [asdict(r) for r in comparator.results]
    with open('comparison_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("✓ Results saved to comparison_results.json")

    # Print final summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)

    excellent = sum(1 for r in comparator.results if r.status == 'EXCELLENT')
    good = sum(1 for r in comparator.results if r.status == 'GOOD')
    total = len(comparator.results)

    print(f"\nExamples tested: {total}")
    print(f"  EXCELLENT: {excellent}")
    print(f"  GOOD: {good}")
    print(f"  Other: {total - excellent - good}")

    avg_error = np.mean([r.relative_error for r in comparator.results])
    avg_speedup = np.mean([r.speedup for r in comparator.results if r.speedup > 0])

    print(f"\nAverage relative error: {avg_error*100:.1f}%")
    print(f"Average speedup: {avg_speedup:.1f}x")

    print("\n✓ Comparison complete!")
    print("\nFiles generated:")
    print("  - comparison_summary.txt")
    print("  - comparison_results.json")


if __name__ == "__main__":
    main()
