#!/usr/bin/env python
"""
Generate Publication-Quality Visualizations
=============================================

Creates charts from benchmark results:
  - Speedup vs Batch Size
  - Throughput vs Grid Size
  - Memory Scaling
  - Time Comparison (Sequential vs Batched)

Usage:
    python benchmarks/generate_plots.py [--input benchmarks/benchmark_raw_data.json]

Output:
    PNG files in benchmarks/plots/
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")


class ResultsVisualizer:
    """Generate visualizations from benchmark results."""

    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.data = self._load_results()
        self.output_dir = self.results_file.parent / 'plots'
        self.output_dir.mkdir(exist_ok=True)

    def _load_results(self) -> Dict:
        """Load benchmark results from JSON."""
        with open(self.results_file, 'r') as f:
            return json.load(f)

    def _setup_style(self):
        """Configure matplotlib style."""
        if not HAS_MATPLOTLIB:
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        colors = {
            'seq': '#FF6B6B',    # Red
            'batch': '#4ECDC4',  # Teal
            'speedup': '#45B7D1' # Blue
        }
        return colors

    def plot_speedup_vs_batch(self):
        """Plot speedup factor vs batch size."""
        if not HAS_MATPLOTLIB:
            return

        colors = self._setup_style()

        # Group by grid size
        speedups_by_grid = {}

        seq_data = self.data['sequential']
        batch_data = self.data['batched']

        for seq, batch in zip(seq_data, batch_data):
            if seq['time_mean'] > 0 and batch['time_mean'] > 0:
                grid = seq['grid']
                batch_size = batch['batch_size']
                speedup = seq['time_mean'] / batch['time_mean']

                if grid not in speedups_by_grid:
                    speedups_by_grid[grid] = {}

                speedups_by_grid[grid][batch_size] = speedup

        fig, ax = plt.subplots(figsize=(10, 6))

        for grid in sorted(speedups_by_grid.keys()):
            batch_sizes = sorted(speedups_by_grid[grid].keys())
            speedups = [speedups_by_grid[grid][b] for b in batch_sizes]

            ax.plot(batch_sizes, speedups, marker='o', label=f'Grid {grid}',
                   linewidth=2, markersize=8)

        # Target range
        ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target Min (10×)')
        ax.axhline(y=15, color='orange', linestyle='--', linewidth=2, label='Target Max (15×)')

        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup Factor (Sequential / Batched)', fontsize=12, fontweight='bold')
        ax.set_title('Speedup vs Batch Size\nBatched 2D FDTD Solver',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        ax.set_xscale('log', base=2)

        fig.tight_layout()
        output_path = self.output_dir / 'speedup_vs_batch.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    def plot_throughput_vs_grid(self):
        """Plot throughput vs grid size."""
        if not HAS_MATPLOTLIB:
            return

        colors = self._setup_style()

        # Extract throughput by grid and batch
        throughputs_seq = {}
        throughputs_batch = {}

        for entry in self.data['sequential']:
            grid = entry['grid']
            batch = entry['batch_size']
            key = f"{grid}_b{batch}"
            throughputs_seq[key] = entry['throughput_gcell_s']

        for entry in self.data['batched']:
            grid = entry['grid']
            batch = entry['batch_size']
            key = f"{grid}_b{batch}"
            throughputs_batch[key] = entry['throughput_gcell_s']

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by batch size
        batches = set()
        for entry in self.data['batched']:
            batches.add(entry['batch_size'])

        batches = sorted(batches)
        grids = sorted(set(entry['grid'] for entry in self.data['batched']))

        x = np.arange(len(grids))
        width = 0.15

        for i, batch in enumerate(batches):
            batch_throughputs = []
            for grid in grids:
                key = f"{grid}_b{batch}"
                batch_throughputs.append(throughputs_batch.get(key, 0))

            ax.bar(x + i*width, batch_throughputs, width,
                  label=f'Batch={batch}', alpha=0.8)

        ax.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (GCell-steps/s)', fontsize=12, fontweight='bold')
        ax.set_title('Throughput vs Grid Size (Batched Execution)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(batches)-1) / 2)
        ax.set_xticklabels(grids)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        output_path = self.output_dir / 'throughput_vs_grid.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    def plot_time_comparison(self):
        """Plot wall-clock time comparison (Sequential vs Batched)."""
        if not HAS_MATPLOTLIB:
            return

        colors = self._setup_style()

        # Select subset of configurations for clarity
        configs_to_plot = []
        for seq, batch in zip(self.data['sequential'], self.data['batched']):
            if batch['batch_size'] in [1, 4, 8, 16]:
                configs_to_plot.append((seq, batch))

        times_seq = [seq['time_mean'] for seq, _ in configs_to_plot]
        times_batch = [batch['time_mean'] for _, batch in configs_to_plot]
        labels = [f"{batch['grid']}\nB={batch['batch_size']}"
                 for _, batch in configs_to_plot]

        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width/2, times_seq, width, label='Sequential',
              color='#FF6B6B', alpha=0.8)
        ax.bar(x + width/2, times_batch, width, label='Batched',
              color='#4ECDC4', alpha=0.8)

        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Wall-Clock Time Comparison\nSequential vs Batched Execution',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        output_path = self.output_dir / 'time_comparison.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    def plot_scaling_efficiency(self):
        """Plot strong scaling efficiency."""
        if not HAS_MATPLOTLIB:
            return

        colors = self._setup_style()

        # Group by grid size and compute scaling efficiency
        scaling_data = {}

        for entry in self.data['batched']:
            grid = entry['grid']
            batch = entry['batch_size']
            throughput = entry['throughput_gcell_s']

            if grid not in scaling_data:
                scaling_data[grid] = {}

            scaling_data[grid][batch] = throughput

        fig, ax = plt.subplots(figsize=(10, 6))

        for grid in sorted(scaling_data.keys()):
            batch_sizes = sorted(scaling_data[grid].keys())
            throughputs = [scaling_data[grid][b] for b in batch_sizes]

            # Normalize to batch=1
            t1 = throughputs[0] if throughputs else 1
            efficiency = [(t / t1) / (b / 1) * 100 for t, b in zip(throughputs, batch_sizes)]

            ax.plot(batch_sizes, efficiency, marker='s', label=f'Grid {grid}',
                   linewidth=2, markersize=8)

        # Ideal scaling line
        ideal_batch = np.array([1, 4, 8, 16])
        ideal_eff = np.ones_like(ideal_batch) * 100
        ax.plot(ideal_batch, ideal_eff, 'k--', linewidth=2, label='Ideal Scaling')

        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Scaling Efficiency (%)', fontsize=12, fontweight='bold')
        ax.set_title('Strong Scaling Efficiency\n(Throughput per batch element)',
                    fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        ax.set_ylim([0, 150])

        fig.tight_layout()
        output_path = self.output_dir / 'scaling_efficiency.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    def plot_accuracy_validation(self):
        """Plot accuracy validation results."""
        if not HAS_MATPLOTLIB or not self.data['accuracy']:
            return

        colors = self._setup_style()

        errors = [entry['max_error'] for entry in self.data['accuracy']]
        configs = [f"{entry['grid']}\nB={entry['batch_size']}"
                  for entry in self.data['accuracy']]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors_arr = ['green' if e < 1e-12 else 'red' for e in errors]
        ax.bar(range(len(errors)), errors, color=colors_arr, alpha=0.7)

        # Tolerance line
        ax.axhline(y=1e-12, color='blue', linestyle='--', linewidth=2,
                  label='Tolerance (1e-12)')

        ax.set_ylabel('Maximum Error', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.set_title('Accuracy Validation\n(Batched vs Sequential Results)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        output_path = self.output_dir / 'accuracy_validation.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)

    def generate_all_plots(self):
        """Generate all visualizations."""
        print(f"\nGenerating visualizations in {self.output_dir}/\n")

        if not HAS_MATPLOTLIB:
            print("ERROR: matplotlib not installed. Install with:")
            print("  pip install matplotlib")
            return

        self.plot_speedup_vs_batch()
        self.plot_throughput_vs_grid()
        self.plot_time_comparison()
        self.plot_scaling_efficiency()
        self.plot_accuracy_validation()

        print(f"\nAll plots saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark visualizations')
    parser.add_argument('--input', default='benchmarks/benchmark_raw_data.json',
                       help='Input JSON file with benchmark results')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run benchmarks first: python benchmarks/batched_2d_benchmark.py")
        return

    visualizer = ResultsVisualizer(str(input_path))
    visualizer.generate_all_plots()


if __name__ == '__main__':
    main()
