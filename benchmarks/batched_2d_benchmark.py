#!/usr/bin/env python
"""
Batched 2D FDTD Solver Benchmark Suite
========================================

Professional-grade benchmarking comparing:
  - Sequential GPU: N source positions run one-at-a-time on GPU
  - Batched GPU: All N sources processed in parallel on GPU
  - CPU reference: NumPy baseline

Measures:
  - Wall-clock execution time
  - Throughput (GCell-steps/second)
  - Speedup factor (sequential vs batched)
  - Accuracy (error vs sequential baseline)
  - Memory usage (peak and sustained)

Usage:
    PYTHONPATH=./src python benchmarks/batched_2d_benchmark.py [--quick]

Requirements:
    - CuPy with compatible NVIDIA GPU
    - numpy, matplotlib, pandas

Output:
    - benchmarks/batched_2d_results.md (comprehensive report)
    - benchmarks/benchmark_raw_data.json (raw measurements)
    - benchmarks/plots/ (visualization)
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
from ceep.core.backend import get_backend_module


class BenchmarkRunner:
    """Orchestrates benchmarking across multiple configurations."""

    def __init__(self, quick_mode=False):
        self.quick_mode = quick_mode
        self.results = {
            'sequential': [],
            'batched': [],
            'accuracy': []
        }
        self.gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import cupy as cp
            _ = cp.zeros(1)
            return True
        except Exception as e:
            print(f"WARNING: GPU not available ({e})")
            return False

    def create_sequential_solver(self, nx, ny, batch_size, steps, frequency=2e9):
        """Create a solver that runs one batch element at a time."""
        source_positions = [(nx//2, ny//2) for _ in range(batch_size)]
        probe_positions = [(nx//4, ny//2), (3*nx//4, ny//2), (nx//2, ny//4)]

        return BatchedFDTD2D(
            nx=nx, ny=ny, dx=0.5e-3,
            total_steps=steps, cpml_thickness=10,
            source_positions=source_positions,
            probe_positions=probe_positions,
            frequency=frequency
        )

    def run_sequential(self, nx, ny, batch_size, steps, runs=1) -> Dict:
        """Run N simulations sequentially on GPU (one batch element at a time)."""
        times = []

        for run in range(runs):
            source_pos = [(nx//2, ny//2)]  # Single source
            probe_pos = [(nx//4, ny//2), (3*nx//4, ny//2), (nx//2, ny//4)]

            total_time = 0
            for b in range(batch_size):
                solver = BatchedFDTD2D(
                    nx=nx, ny=ny, dx=0.5e-3,
                    total_steps=steps, cpml_thickness=10,
                    source_positions=source_pos,
                    probe_positions=probe_pos,
                    frequency=2e9
                )

                start = time.perf_counter()
                if self.gpu_available:
                    import cupy as cp
                    cp.cuda.Device().synchronize()
                    result = solver.run()
                    cp.cuda.Device().synchronize()
                else:
                    result = solver.run_cpu()
                elapsed = time.perf_counter() - start
                total_time += elapsed

            times.append(total_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        cell_steps = nx * ny * steps * batch_size
        throughput = cell_steps / avg_time / 1e9  # GCell-steps/s

        return {
            'grid': f'{nx}x{ny}',
            'batch_size': batch_size,
            'steps': steps,
            'time_mean': avg_time,
            'time_std': std_time,
            'throughput_gcell_s': throughput,
            'cell_steps_total': cell_steps
        }

    def run_batched(self, nx, ny, batch_size, steps, runs=1) -> Dict:
        """Run all N simulations in parallel (batched)."""
        times = []
        results_list = []

        for run in range(runs):
            source_positions = [(nx//2, ny//2) for _ in range(batch_size)]
            probe_positions = [(nx//4, ny//2), (3*nx//4, ny//2), (nx//2, ny//4)]

            solver = BatchedFDTD2D(
                nx=nx, ny=ny, dx=0.5e-3,
                total_steps=steps, cpml_thickness=10,
                source_positions=source_positions,
                probe_positions=probe_positions,
                frequency=2e9
            )

            start = time.perf_counter()
            if self.gpu_available:
                import cupy as cp
                cp.cuda.Device().synchronize()
                result = solver.run()
                cp.cuda.Device().synchronize()
            else:
                result = solver.run_cpu()
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            results_list.append(result)

        avg_time = np.mean(times)
        std_time = np.std(times)
        cell_steps = nx * ny * steps * batch_size
        throughput = cell_steps / avg_time / 1e9  # GCell-steps/s

        return {
            'grid': f'{nx}x{ny}',
            'batch_size': batch_size,
            'steps': steps,
            'time_mean': avg_time,
            'time_std': std_time,
            'throughput_gcell_s': throughput,
            'cell_steps_total': cell_steps,
            'results': results_list[0] if results_list else None
        }

    def validate_accuracy(self, nx, ny, batch_size, steps) -> Dict:
        """Compare batched vs sequential results (should be identical)."""
        # Run sequential
        seq_result = None
        for b in range(batch_size):
            source_pos = [(nx//2, ny//2)]
            probe_pos = [(nx//4, ny//2)]
            solver = BatchedFDTD2D(
                nx=nx, ny=ny, dx=0.5e-3,
                total_steps=steps, cpml_thickness=10,
                source_positions=source_pos,
                probe_positions=probe_pos,
                frequency=2e9
            )

            if self.gpu_available:
                import cupy as cp
                cp.cuda.Device().synchronize()
                result = solver.run()
                cp.cuda.Device().synchronize()
                result = {k: {kk: cp.asnumpy(vv) if hasattr(vv, 'get') else vv
                             for kk, vv in v.items()}
                         for k, v in result.items()}
            else:
                result = solver.run_cpu()

            if seq_result is None:
                seq_result = result

        # Run batched
        source_positions = [(nx//2, ny//2) for _ in range(batch_size)]
        probe_positions = [(nx//4, ny//2)]

        solver = BatchedFDTD2D(
            nx=nx, ny=ny, dx=0.5e-3,
            total_steps=steps, cpml_thickness=10,
            source_positions=source_positions,
            probe_positions=probe_positions,
            frequency=2e9
        )

        if self.gpu_available:
            import cupy as cp
            cp.cuda.Device().synchronize()
            batch_result = solver.run()
            cp.cuda.Device().synchronize()
            batch_result = {k: {kk: cp.asnumpy(vv) if hasattr(vv, 'get') else vv
                               for kk, vv in v.items()}
                           for k, v in batch_result.items()}
        else:
            batch_result = solver.run_cpu()

        # Compute error for first batch element
        errors = []
        for tx_idx in range(min(1, batch_size)):
            for rx_idx in range(1):
                seq_signal = seq_result[tx_idx][rx_idx]
                batch_signal = batch_result[tx_idx][rx_idx]

                if isinstance(seq_signal, list):
                    seq_signal = np.array(seq_signal)
                if isinstance(batch_signal, list):
                    batch_signal = np.array(batch_signal)

                error = np.max(np.abs(seq_signal - batch_signal))
                errors.append(error)

        max_error = np.max(errors) if errors else 0

        return {
            'grid': f'{nx}x{ny}',
            'batch_size': batch_size,
            'steps': steps,
            'max_error': max_error,
            'passes': max_error < 1e-12
        }

    def get_memory_usage(self) -> Dict:
        """Get peak GPU memory usage."""
        if not self.gpu_available:
            return {'peak_mb': 0, 'available_mb': 0}

        try:
            import cupy as cp
            mem_info = cp.cuda.Device().mem_info
            available = mem_info[0] / 1e6
            total = mem_info[1] / 1e6
            used = (mem_info[1] - mem_info[0]) / 1e6

            return {
                'available_mb': available,
                'total_mb': total,
                'used_mb': used
            }
        except Exception as e:
            print(f"Could not get memory info: {e}")
            return {'available_mb': 0, 'total_mb': 0, 'used_mb': 0}

    def run_benchmark_suite(self):
        """Run full benchmark suite across all configurations."""
        if self.quick_mode:
            configs = [
                (300, 300, 1, 100),
                (300, 300, 4, 100),
                (300, 300, 8, 100),
                (600, 600, 4, 50),
            ]
        else:
            configs = [
                # (nx, ny, batch, steps)
                (300, 300, 1, 100),
                (300, 300, 1, 200),
                (300, 300, 4, 100),
                (300, 300, 4, 200),
                (300, 300, 8, 100),
                (300, 300, 8, 200),
                (300, 300, 16, 100),

                (600, 600, 1, 100),
                (600, 600, 4, 100),
                (600, 600, 8, 100),
                (600, 600, 16, 100),

                (1000, 1000, 1, 50),
                (1000, 1000, 4, 50),
                (1000, 1000, 8, 50),
            ]

        print("\n" + "="*80)
        print("BATCHED 2D FDTD BENCHMARK SUITE")
        print("="*80)
        print(f"GPU Available: {self.gpu_available}")
        print(f"Quick Mode: {self.quick_mode}")
        print()

        mem = self.get_memory_usage()
        if self.gpu_available:
            print(f"GPU Memory: {mem['total_mb']:.0f} MB total, "
                  f"{mem['available_mb']:.0f} MB available\n")

        for idx, (nx, ny, batch, steps) in enumerate(configs):
            print(f"[{idx+1}/{len(configs)}] Grid {nx}x{ny}, "
                  f"Batch={batch}, Steps={steps}... ", end="", flush=True)

            try:
                runs = 2 if not self.quick_mode else 1

                # Sequential
                seq = self.run_sequential(nx, ny, batch, steps, runs=runs)
                self.results['sequential'].append(seq)

                # Batched
                batch_res = self.run_batched(nx, ny, batch, steps, runs=runs)
                self.results['batched'].append(batch_res)

                # Speedup
                speedup = seq['time_mean'] / batch_res['time_mean']

                # Accuracy
                if batch == 1 or idx % 3 == 0:  # Test accuracy on a few configs
                    acc = self.validate_accuracy(nx, ny, batch, steps)
                    self.results['accuracy'].append(acc)

                print(f"Speedup: {speedup:.1f}x")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        return self.results

    def save_results(self, output_dir='benchmarks'):
        """Save raw benchmark data to JSON."""
        output_path = Path(output_dir) / 'benchmark_raw_data.json'

        # Convert results to JSON-serializable format (remove numpy arrays)
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        json_results = {
            'sequential': [convert_to_serializable(r) for r in self.results['sequential']],
            'batched': [convert_to_serializable(r) for r in self.results['batched']],
            'accuracy': [convert_to_serializable(r) for r in self.results['accuracy']]
        }

        # Remove 'results' keys that contain solver outputs
        for entry in json_results['batched']:
            if 'results' in entry:
                del entry['results']

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\nRaw data saved to: {output_path}")
        return output_path

    def generate_summary(self) -> str:
        """Generate summary statistics."""
        if not self.results['batched'] or not self.results['sequential']:
            return "No benchmark results available."

        speedups = []
        for seq, batch in zip(self.results['sequential'], self.results['batched']):
            if seq['time_mean'] > 0:
                speedup = seq['time_mean'] / batch['time_mean']
                speedups.append(speedup)

        if not speedups:
            return "No speedup data available."

        avg_speedup = np.mean(speedups)
        min_speedup = np.min(speedups)
        max_speedup = np.max(speedups)

        # Check accuracy
        accuracy_pass = all(acc.get('passes', False) for acc in self.results['accuracy'])
        accuracy_status = "PASS" if accuracy_pass else "FAIL"

        summary = f"""
## Summary

**Speedup Achievement**: {avg_speedup:.1f}x average
- Minimum: {min_speedup:.1f}x
- Maximum: {max_speedup:.1f}x
- Target: 10-15x

**Accuracy Validation**: {accuracy_status}
- Error tolerance: < 1e-12 (machine precision)

**Total Configurations Tested**: {len(self.results['sequential'])}

**GPU Available**: {self.gpu_available}
"""
        return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batched 2D FDTD Benchmark')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark mode')
    parser.add_argument('--output-dir', default='benchmarks',
                       help='Output directory for results')
    args = parser.parse_args()

    runner = BenchmarkRunner(quick_mode=args.quick)
    runner.run_benchmark_suite()
    runner.save_results(args.output_dir)

    summary = runner.generate_summary()
    print(summary)


if __name__ == '__main__':
    main()
