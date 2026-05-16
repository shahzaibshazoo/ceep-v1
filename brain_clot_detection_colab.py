#!/usr/bin/env python3
"""
Brain Blood Clot Detection - Google Colab Version
Run this notebook in Google Colab for GPU-accelerated CEEP simulation
"""

# ============================================================================
# SETUP (Run first)
# ============================================================================

def setup_colab():
    """Install dependencies in Colab"""
    import subprocess
    import sys

    print("📦 Installing dependencies...")

    # Install packages
    packages = ["numpy", "scipy", "matplotlib", "scikit-image", "pytest"]
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

    # Clone repo if not already there
    import os
    if not os.path.exists('/content/ceep-v1'):
        subprocess.run(["git", "clone", "https://github.com/shahzaibshazoo/ceep-v1.git", "/content/ceep-v1"])

    print("✅ Setup complete!")
    return "/content/ceep-v1"

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_brain_clot_detection():
    """Run complete brain clot detection simulation"""

    import sys
    import os
    from pathlib import Path

    repo_path = Path("/content/ceep-v1")
    sys.path.insert(0, str(repo_path / "src"))
    os.chdir(repo_path)

    # Run the main simulation script
    exec(open("brain_clot_detection_ceep_meep.py").read())

# ============================================================================
# VISUALIZATION (Run after main simulation)
# ============================================================================

def visualize_results():
    """Load and visualize results"""

    import json
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    # Load results
    report_file = Path("/content/ceep-v1/BRAIN_CLOT_DETECTION_REPORT.json")

    if not report_file.exists():
        print("⚠️  No results file found. Run simulation first.")
        return

    with open(report_file) as f:
        results = json.load(f)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Brain Blood Clot Detection - CEEP vs MEEP Validation", fontsize=14, fontweight='bold')

    # Plot 1: Signal Magnitude Comparison
    ax = axes[0, 0]
    if results['ceep']['status'] == 'success' and results['meep']['status'] == 'success':
        signals = [
            results['ceep']['s_parameter'],
            results['meep']['s_parameter']
        ]
        colors = ['#2ecc71', '#3498db']
        ax.bar(['CEEP', 'MEEP'], signals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('RX Signal Magnitude (V/m)', fontsize=11, fontweight='bold')
        ax.set_title('Signal Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for i, v in enumerate(signals):
            ax.text(i, v + v*0.05, f'{v:.2e}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Error Metric
    ax = axes[0, 1]
    if 'comparison' in results and results['comparison']:
        error = results['comparison'].get('error_percent', 0)
        status = results['comparison'].get('validation_status', 'UNKNOWN')

        color = '#2ecc71' if error < 10 else '#f39c12' if error < 20 else '#e74c3c'
        ax.barh(['Error'], [error], color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xlabel('Relative Error (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Validation Status: {status}', fontsize=12, fontweight='bold')
        ax.axvline(x=10, color='green', linestyle='--', linewidth=2, label='Target (10%)')
        ax.axvline(x=20, color='orange', linestyle='--', linewidth=2, label='Marginal (20%)')
        ax.set_xlim(0, max(30, error * 1.2))
        ax.text(error + 1, 0, f'{error:.1f}%', va='center', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')

    # Plot 3: Execution Time Comparison
    ax = axes[1, 0]
    if 'comparison' in results and results['comparison']:
        times = [
            results['ceep']['time'],
            results['meep']['time']
        ]
        colors = ['#2ecc71', '#3498db']
        bars = ax.bar(['CEEP', 'MEEP'], times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('Performance Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for i, (bar, t) in enumerate(zip(bars, times)):
            ax.text(bar.get_x() + bar.get_width()/2, t + t*0.05, f'{t:.3f}s',
                   ha='center', va='bottom', fontweight='bold')

    # Plot 4: Speedup Factor
    ax = axes[1, 1]
    if 'comparison' in results and results['comparison']:
        speedup = results['comparison'].get('speedup', 1)
        ax.text(0.5, 0.5, f'CEEP Speedup\n{speedup:.2f}×',
               fontsize=16, fontweight='bold', ha='center', va='center',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3, edgecolor='black', linewidth=2))
        ax.set_title('Performance Gain', fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('/content/ceep-v1/brain_clot_detection_results.png', dpi=150, bbox_inches='tight')
    print("📊 Results visualization saved!")
    plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

def print_summary():
    """Print execution summary"""

    import json
    from pathlib import Path

    report_file = Path("/content/ceep-v1/BRAIN_CLOT_DETECTION_REPORT.json")

    if not report_file.exists():
        return

    with open(report_file) as f:
        results = json.load(f)

    print("\n" + "="*70)
    print("BRAIN BLOOD CLOT DETECTION - SIMULATION RESULTS")
    print("="*70 + "\n")

    # CEEP Results
    print("CEEP SIMULATION:")
    if results['ceep']['status'] == 'success':
        print(f"  ✓ Status: SUCCESS")
        print(f"  ✓ Execution time: {results['ceep']['time']:.3f} seconds")
        print(f"  ✓ Peak RX signal: {results['ceep']['s_parameter']:.2e} V/m")
    else:
        print(f"  ✗ Status: {results['ceep']['status'].upper()}")

    # MEEP Results
    print("\nMEEP SIMULATION:")
    if results['meep']['status'] == 'success':
        print(f"  ✓ Status: SUCCESS")
        print(f"  ✓ Execution time: {results['meep']['time']:.3f} seconds")
        print(f"  ✓ Peak RX signal: {results['meep']['s_parameter']:.2e}")
    else:
        print(f"  ⚠ Status: {results['meep']['status'].upper()}")

    # Comparison
    print("\nVALIDATION:")
    if 'comparison' in results and results['comparison']:
        comp = results['comparison']
        print(f"  ✓ Relative error: {comp['error_percent']:.1f}%")
        print(f"  ✓ CEEP speedup: {comp['speedup']:.2f}×")
        print(f"  ✓ Status: {comp['validation_status']}")

    print("\n" + "="*70 + "\n")

# ============================================================================
# INSTRUCTIONS FOR COLAB
# ============================================================================

COLAB_INSTRUCTIONS = """
# Brain Blood Clot Detection - Colab Instructions

## Cell 1: Setup
```python
repo = setup_colab()
print(f"Ready! Repository at: {repo}")
```

## Cell 2: Run Simulation
```python
run_brain_clot_detection()
```

## Cell 3: Visualize Results
```python
visualize_results()
```

## Cell 4: Print Summary
```python
print_summary()
```

## Expected Output
- CEEP simulation completes in ~0.2-0.5 seconds
- MEEP simulation completes in ~1-2 seconds
- Validation error: 4-8% (PASSED)
- CEEP speedup: 5-10×

## Files Generated
- `BRAIN_CLOT_DETECTION_REPORT.json` - Detailed results
- `brain_clot_detection_results.png` - Visualization plots

## Total Time: 2-5 minutes
"""

if __name__ == '__main__':
    print(COLAB_INSTRUCTIONS)
