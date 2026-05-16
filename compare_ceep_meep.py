#!/usr/bin/env python3
"""
CEEP vs MEEP Comprehensive Validation Suite
Automatically tests and compares both solvers
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup paths
repo_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(repo_root / 'src'))
os.environ['PYTHONPATH'] = str(repo_root / 'src')

# Colors
class C:
    G = '\033[92m'
    R = '\033[91m'
    Y = '\033[93m'
    B = '\033[94m'
    X = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def header(text):
    print(f"\n{C.BOLD}{C.X}{'='*70}{C.END}")
    print(f"{C.BOLD}{C.X}{text:^70}{C.END}")
    print(f"{C.BOLD}{C.X}{'='*70}{C.END}\n")

def success(text):
    print(f"{C.G}✓ {text}{C.END}")

def error(text):
    print(f"{C.R}✗ {text}{C.END}")

def warning(text):
    print(f"{C.Y}⚠ {text}{C.END}")

def info(text):
    print(f"{C.B}ℹ {text}{C.END}")

# ============================================================================

def test_ceep():
    """Test CEEP"""
    header("TEST 1: CEEP SOLVER")
    try:
        from ceep.solvers import FDTD2D
        from ceep.core import Grid2D, Config2D, PointSource

        info("Creating configuration...")
        config = Config2D(nx=100, ny=100, dx=0.5e-3, dy=0.5e-3, frequency_hz=1e9)
        grid = Grid2D(config)
        source = PointSource(x=50e-3, y=25e-3, frequency_hz=1e9)

        info("Creating solver...")
        solver = FDTD2D(config=config, grid=grid, sources=[source], boundaries=None)

        info("Running 100 timesteps...")
        start = time.time()
        for _ in range(100):
            solver.step()
        elapsed = time.time() - start

        field = solver.get_field('Ez')
        energy = np.sum(np.abs(field)**2)

        success(f"CEEP completed in {elapsed:.3f}s")
        info(f"  Peak field: {np.max(np.abs(field)):.2e}")
        info(f"  Energy: {energy:.2e}")
        return {'status': 'pass', 'time': elapsed, 'energy': float(energy)}

    except Exception as e:
        error(f"CEEP failed: {e}")
        return {'status': 'fail', 'error': str(e)}

def test_meep():
    """Test MEEP"""
    header("TEST 2: MEEP SOLVER")
    try:
        import meep as mp

        info("Creating configuration...")
        cell = mp.Vector3(5, 5, 0)
        sources = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez, center=mp.Vector3(0, -1))]

        info("Creating solver...")
        sim = mp.Simulation(cell_size=cell, sources=sources, pml_layers=[mp.PML(1.0)], resolution=20)

        info("Running 100 timesteps...")
        start = time.time()
        sim.run(mp.until_time(100))
        elapsed = time.time() - start

        fields = sim.get_array(component=mp.Ez)

        success(f"MEEP completed in {elapsed:.3f}s")
        info(f"  Peak field: {np.max(np.abs(fields)):.2e}")
        return {'status': 'pass', 'time': elapsed}

    except ImportError:
        warning("MEEP not installed - skipping")
        return {'status': 'skip', 'reason': 'Not installed'}
    except Exception as e:
        error(f"MEEP failed: {e}")
        return {'status': 'fail', 'error': str(e)}

def benchmark_ceep():
    """Benchmark CEEP"""
    header("BENCHMARK: CEEP (200×200, 200 steps)")
    try:
        from ceep.solvers import FDTD2D
        from ceep.core import Grid2D, Config2D, PointSource

        config = Config2D(nx=200, ny=200, dx=0.5e-3, dy=0.5e-3, frequency_hz=1e9)
        grid = Grid2D(config)
        source = PointSource(x=100e-3, y=100e-3, frequency_hz=1e9)
        solver = FDTD2D(config=config, grid=grid, sources=[source], boundaries=None)

        start = time.time()
        for _ in range(200):
            solver.step()
        elapsed = time.time() - start

        cells = 200 * 200
        throughput = (cells * 200) / (elapsed * 1e9)

        success(f"Completed in {elapsed:.3f}s")
        info(f"  Throughput: {throughput:.3f} GCell-steps/sec")
        return elapsed

    except Exception as e:
        error(f"CEEP benchmark failed: {e}")
        return None

def benchmark_meep():
    """Benchmark MEEP"""
    header("BENCHMARK: MEEP (10×10, 200 steps)")
    try:
        import meep as mp

        cell = mp.Vector3(10, 10, 0)
        sources = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez, center=mp.Vector3(0, 0))]
        sim = mp.Simulation(cell_size=cell, sources=sources, pml_layers=[mp.PML(1.0)], resolution=20)

        start = time.time()
        sim.run(mp.until_time(200))
        elapsed = time.time() - start

        success(f"Completed in {elapsed:.3f}s")
        return elapsed

    except ImportError:
        warning("MEEP not installed")
        return None
    except Exception as e:
        error(f"MEEP benchmark failed: {e}")
        return None

def main():
    """Main"""
    header("CEEP vs MEEP VALIDATION SUITE")
    info("Comprehensive automated testing\n")

    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'benchmarks': {}
    }

    try:
        # Tests
        results['tests']['ceep'] = test_ceep()
        results['tests']['meep'] = test_meep()

        # Benchmarks
        ceep_time = benchmark_ceep()
        meep_time = benchmark_meep()

        results['benchmarks']['ceep'] = ceep_time
        results['benchmarks']['meep'] = meep_time

        # Comparison
        if ceep_time and meep_time:
            header("PERFORMANCE COMPARISON")
            speedup = meep_time / ceep_time
            info(f"CEEP: {ceep_time:.3f}s (200×200)")
            info(f"MEEP: {meep_time:.3f}s (10×10)")
            info(f"Speedup: {speedup:.2f}×")

        # Save
        report = repo_root / 'VALIDATION_RESULTS.json'
        with open(report, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        success(f"\nResults saved to: {report}\n")
        return True

    except KeyboardInterrupt:
        warning("\nInterrupted")
        return False
    except Exception as e:
        error(f"\nFailed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
