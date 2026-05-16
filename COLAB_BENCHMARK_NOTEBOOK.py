# Google Colab Benchmark Notebook
# Run this in Google Colab (colab.research.google.com) to execute benchmarks on T4/A100 GPU
# Copy-paste into a Colab cell and run

# ============================================================================
# SETUP: Install dependencies and clone repo
# ============================================================================

# Cell 1: Install packages and clone repo
!pip install -q cupy-cuda11x numpy scipy scikit-image matplotlib seaborn pandas tabulate -q
!git clone https://github.com/shahzaibshazoo/ceep-v1.git /content/cuda-meep 2>/dev/null || echo \"Repo already cloned\"
%cd /content/cuda-meep

# ============================================================================
# Cell 2: Verify GPU is available
# ============================================================================

import subprocess
import sys

print(\"🔍 Checking GPU availability...\\n\")
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader'],
                       capture_output=True, text=True)
print(result.stdout)

# Import key packages
import os
os.environ['PYTHONPATH'] = '/content/cuda-meep/src'
sys.path.insert(0, '/content/cuda-meep/src')

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(\"✅ CuPy successfully imported - GPU acceleration ready!\\n\")
except ImportError:
    GPU_AVAILABLE = False
    print(\"⚠️  CuPy not available - benchmarks will run on CPU\\n\")

# ============================================================================
# Cell 3: Run Quick Validation (sanity check)
# ============================================================================

print(\"=\" * 70)
print(\"QUICK VALIDATION TEST\")
print(\"=\" * 70)

exec(open('/content/cuda-meep/benchmarks/quick_validation.py').read())

# ============================================================================
# Cell 4: Run Full Benchmark Suite
# ============================================================================

print(\"\\n\" + \"=\" * 70)
print(\"FULL BENCHMARK SUITE - Sequential vs Batched FDTD 2D\")
print(\"=\" * 70)
print(\"\\nThis will take ~30-45 minutes on GPU (T4: ~30 min, A100: ~15 min)\")\nprint(\"CPU-only will take ~5-10 minutes but show no speedup (expected)\\n\")

import json
import time

# Set benchmark configuration
os.chdir('/content/cuda-meep/benchmarks')

# Run the main benchmark
exec(open('batched_2d_benchmark.py').read())

# ============================================================================\n# Cell 5: Analyze Results and Generate Report
# ============================================================================

print(\"\\n\" + \"=\" * 70)
print(\"ANALYZING BENCHMARK RESULTS\")\nprint(\"=\" * 70)\n\nexec(open('analyze_results.py').read())\n\n# ============================================================================\n# Cell 6: Generate Visualizations\n# ============================================================================\n\nprint(\"\\n\" + \"=\" * 70)\nprint(\"GENERATING VISUALIZATIONS\")\nprint(\"=\" * 70)\n\nexec(open('generate_plots.py').read())\n\nprint(\"\\n✅ All visualizations saved to plots/\")\n\n# ============================================================================\n# Cell 7: Display Results Summary\n# ============================================================================\n\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image, display, Markdown\nimport os\n\nprint(\"\\n\" + \"=\" * 70)\nprint(\"BENCHMARK RESULTS SUMMARY\")\nprint(\"=\" * 70)\n\n# Read and display the report\nwith open('batched_2d_results.md', 'r') as f:\n    report_text = f.read()\n    # Convert markdown to display\n    display(Markdown(report_text))\n\n# Display all plots\nplot_dir = 'plots'\nif os.path.exists(plot_dir):\n    print(\"\\n\" + \"=\" * 70)\n    print(\"PERFORMANCE VISUALIZATIONS\")\n    print(\"=\" * 70)\n    \n    plots = sorted([f for f in os.listdir(plot_dir) if f.endswith('.png')])\n    for plot_file in plots:\n        plot_path = os.path.join(plot_dir, plot_file)\n        print(f\"\\n📊 {plot_file}\")\n        display(Image(plot_path))\n\n# ============================================================================\n# Cell 8: Download Results\n# ============================================================================\n\nfrom google.colab import files\nimport shutil\n\nprint(\"\\n\" + \"=\" * 70)\nprint(\"DOWNLOAD RESULTS\")\nprint(\"=\" * 70)\n\n# Create a zip of all results\nshutil.make_archive('/content/benchmark_results', 'zip', '/content/cuda-meep/benchmarks')\n\nprint(\"\\n📥 Downloading benchmark_results.zip...\\n\")\nfiles.download('/content/benchmark_results.zip')\n\nprint(\"✅ Download complete!\")\nprint(\"\\n📋 Contents:\")\nprint(\"   - batched_2d_results.md (full report)\")\nprint(\"   - benchmark_raw_data.json (raw measurements)\")\nprint(\"   - plots/*.png (visualizations)\")\nprint(\"   - BENCHMARK_README.md (user guide)\")\nprint(\"   - BENCHMARK_DESIGN.md (technical specification)\")\n\n# ============================================================================\n# Cell 9: Print Raw Data Table\n# ============================================================================\n\nprint(\"\\n\" + \"=\" * 70)\nprint(\"RAW BENCHMARK DATA\")\nprint(\"=\" * 70)\n\nimport json\nimport pandas as pd\nfrom tabulate import tabulate\n\nwith open('benchmark_raw_data.json', 'r') as f:\n    data = json.load(f)\n\n# Sequential data\nprint(\"\\n🔄 SEQUENTIAL EXECUTION (one source at a time)\")\nseq_rows = []\nfor config in data.get('sequential', []):\n    seq_rows.append([\n        f\"{config['grid_size'][0]}×{config['grid_size'][1]}\",\n        config['batch_size'],\n        config['timesteps'],\n        f\"{config['elapsed_time']:.4f}\",\n        f\"{config['throughput']:.3f}\"\n    ])\nprint(tabulate(seq_rows, \n              headers=['Grid', 'Batch', 'Steps', 'Time (s)', 'Throughput (GCell/s)'],\n              tablefmt='grid'))\n\n# Batched data\nprint(\"\\n⚡ BATCHED EXECUTION (all sources in parallel)\")\nbatch_rows = []\nfor config in data.get('batched', []):\n    batch_rows.append([\n        f\"{config['grid_size'][0]}×{config['grid_size'][1]}\",\n        config['batch_size'],\n        config['timesteps'],\n        f\"{config['elapsed_time']:.4f}\",\n        f\"{config['throughput']:.3f}\"\n    ])\nprint(tabulate(batch_rows,\n              headers=['Grid', 'Batch', 'Steps', 'Time (s)', 'Throughput (GCell/s)'],\n              tablefmt='grid'))\n\n# Speedup comparison\nprint(\"\\n📈 SPEEDUP ANALYSIS\")\nspeedup_rows = []\nfor i, (seq, batch) in enumerate(zip(data.get('sequential', []), data.get('batched', []))):\n    speedup = seq['elapsed_time'] / batch['elapsed_time']\n    speedup_rows.append([\n        f\"{seq['grid_size'][0]}×{seq['grid_size'][1]}\",\n        seq['batch_size'],\n        f\"{seq['elapsed_time']:.4f}\",\n        f\"{batch['elapsed_time']:.4f}\",\n        f\"{speedup:.2f}×\"\n    ])\nprint(tabulate(speedup_rows,\n              headers=['Grid', 'Batch', 'Seq (s)', 'Batch (s)', 'Speedup'],\n              tablefmt='grid'))\n\n# Accuracy validation\nprint(\"\\n✅ ACCURACY VALIDATION\")\naccuracy_rows = []\nfor config in data.get('accuracy_validation', []):\n    status = '✅ PASS' if config['error'] < 1e-12 else '❌ FAIL'\n    accuracy_rows.append([\n        f\"{config['grid_size'][0]}×{config['grid_size'][1]}\",\n        config['batch_size'],\n        f\"{config['error']:.2e}\",\n        '< 1e-12',\n        status\n    ])\nprint(tabulate(accuracy_rows,\n              headers=['Grid', 'Batch', 'Max Error', 'Tolerance', 'Status'],\n              tablefmt='grid'))\n\n# ============================================================================\n# Cell 10: Interpretation & Next Steps\n# ============================================================================\n\nprint(\"\\n\" + \"=\" * 70)\nprint(\"INTERPRETATION\")\nprint(\"=\" * 70)\n\navg_speedup = np.mean([s['elapsed_time'] / b['elapsed_time'] \n                       for s, b in zip(data.get('sequential', []), data.get('batched', []))])\n\nif avg_speedup >= 10:\n    status = \"✅ EXCELLENT - Target achieved!\"\nelif avg_speedup >= 8:\n    status = \"✅ GOOD - Approaching target\"\nelif avg_speedup >= 5:\n    status = \"🟡 MODERATE - Acceptable for some use cases\"\nelse:\n    status = \"⚠️  LOW - Check GPU utilization (nvidia-smi)\"\n\nprint(f\"\\n📊 Average Speedup: {avg_speedup:.2f}×\")\nprint(f\"   Status: {status}\")\nprint(f\"   Target: 10-15×\")\n\nif GPU_AVAILABLE:\n    print(\"\\n💡 GPU is available and being used.\")\n    print(\"   If speedup < 10×:\")\n    print(\"   1. Check nvidia-smi (GPU utilization)\")\n    print(\"   2. Increase batch size (16, 32, 64)\")\n    print(\"   3. Increase grid size (1000×1000+)\")\nelse:\n    print(\"\\n💡 GPU not available - running on CPU.\")\n    print(\"   Expected: <1× speedup (normal for CPU)\")\n    print(\"   To enable GPU: Restart runtime with GPU enabled\")\n\nprint(\"\\n\" + \"=\" * 70)\nprint(\"NEXT STEPS\")\nprint(\"=\" * 70)\nprint(\"\\n1. ✅ Download benchmark_results.zip\")\nprint(\"2. 📊 Review batched_2d_results.md for full analysis\")\nprint(\"3. 📈 Share speedup metrics with team\")\nprint(\"4. 🚀 Use optimal batch size for production\")\nprint(\"\\nFor 3D batched solver: Agent 4 is working on implementation...\")\n"}}]
</invoke>