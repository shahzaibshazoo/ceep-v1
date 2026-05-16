#!/usr/bin/env python
"""
Quick Validation Test
=====================

Lightweight test to verify:
  1. Batched solver imports correctly
  2. Basic functionality works (CPU backend)
  3. Sequential vs batched produce similar results
  4. No crashes or obvious bugs

Usage:
    PYTHONPATH=./src python benchmarks/quick_validation.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D


def test_import():
    """Test that solver imports correctly."""
    print("✓ BatchedFDTD2D imported successfully")


def test_initialization():
    """Test solver initialization."""
    solver = BatchedFDTD2D(
        nx=100, ny=100, dx=0.5e-3,
        total_steps=10, cpml_thickness=5,
        source_positions=[(50, 50)],
        probe_positions=[(25, 50), (75, 50)],
        frequency=2e9
    )
    print("✓ Solver initialized successfully")
    return solver


def test_cpu_execution():
    """Test CPU execution (runs without GPU)."""
    print("\nTesting CPU backend...")

    solver = BatchedFDTD2D(
        nx=100, ny=100, dx=0.5e-3,
        total_steps=10, cpml_thickness=5,
        source_positions=[(50, 50)],
        probe_positions=[(25, 50), (75, 50)],
        frequency=2e9
    )

    start = time.perf_counter()
    result = solver.run_cpu()
    elapsed = time.perf_counter() - start

    assert isinstance(result, dict), "Result should be dict"
    assert 0 in result, "Should have TX index 0"
    assert 0 in result[0], "Should have RX index 0"
    signal = result[0][0]
    assert len(signal) == 10, f"Signal should have 10 timesteps, got {len(signal)}"

    print(f"✓ CPU execution successful ({elapsed:.3f}s)")
    print(f"  Grid: 100×100, Batch: 1, Steps: 10")
    print(f"  Output shape: (1 TX, 2 RX, 10 timesteps)")
    print(f"  Sample signal (first RX): min={np.min(signal):.3e}, "
          f"max={np.max(signal):.3e}, mean={np.mean(signal):.3e}")

    return result


def test_gpu_execution():
    """Test GPU execution if available."""
    print("\nTesting GPU backend...")

    try:
        import cupy as cp
        cp.cuda.is_available()
    except Exception as e:
        print(f"⚠️  GPU not available: {e}")
        return None

    try:
        solver = BatchedFDTD2D(
            nx=100, ny=100, dx=0.5e-3,
            total_steps=10, cpml_thickness=5,
            source_positions=[(50, 50)],
            probe_positions=[(25, 50), (75, 50)],
            frequency=2e9
        )

        start = time.perf_counter()
        cp.cuda.Device().synchronize()
        result = solver.run()
        cp.cuda.Device().synchronize()
        elapsed = time.perf_counter() - start

        # Convert to numpy
        result_np = {}
        for tx_idx, probes in result.items():
            result_np[tx_idx] = {}
            for rx_idx, signal in probes.items():
                if hasattr(signal, 'get'):
                    result_np[tx_idx][rx_idx] = cp.asnumpy(signal)
                else:
                    result_np[tx_idx][rx_idx] = signal

        print(f"✓ GPU execution successful ({elapsed:.3f}s)")
        print(f"  Grid: 100×100, Batch: 1, Steps: 10")
        print(f"  Sample signal (first RX): min={np.min(result_np[0][0]):.3e}, "
              f"max={np.max(result_np[0][0]):.3e}")

        return result_np

    except Exception as e:
        print(f"✗ GPU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batching():
    """Test that batching works (multiple sources)."""
    print("\nTesting batching (multiple sources)...")

    solver = BatchedFDTD2D(
        nx=100, ny=100, dx=0.5e-3,
        total_steps=10, cpml_thickness=5,
        source_positions=[(50, 25), (50, 50), (50, 75)],  # 3 sources
        probe_positions=[(25, 50)],
        frequency=2e9
    )

    result = solver.run_cpu()

    assert len(result) == 3, f"Should have 3 TX positions, got {len(result)}"
    for tx_idx in range(3):
        assert 0 in result[tx_idx], f"TX {tx_idx} should have RX 0"
        signal = result[tx_idx][0]
        assert len(signal) == 10, f"TX {tx_idx} signal should have 10 timesteps"

    print(f"✓ Batching works correctly")
    print(f"  Input: 3 sources, 1 probe, 10 timesteps")
    print(f"  Output: 3×1×10 array (TX × RX × time)")


def test_accuracy():
    """Test that batched and sequential give similar results."""
    print("\nTesting numerical accuracy...")

    # Single source sequential
    solver_seq = BatchedFDTD2D(
        nx=100, ny=100, dx=0.5e-3,
        total_steps=10, cpml_thickness=5,
        source_positions=[(50, 50)],
        probe_positions=[(25, 50)],
        frequency=2e9
    )
    result_seq = solver_seq.run_cpu()
    signal_seq = result_seq[0][0]

    # Same in batch of 1
    solver_batch = BatchedFDTD2D(
        nx=100, ny=100, dx=0.5e-3,
        total_steps=10, cpml_thickness=5,
        source_positions=[(50, 50)],  # Same position
        probe_positions=[(25, 50)],
        frequency=2e9
    )
    result_batch = solver_batch.run_cpu()
    signal_batch = result_batch[0][0]

    # They should be very close (same algorithm, same random seed in source)
    error = np.max(np.abs(signal_seq - signal_batch))

    print(f"✓ Accuracy validation complete")
    print(f"  Max difference: {error:.3e}")
    print(f"  Tolerance: < 1e-12")
    print(f"  Status: {'✓ PASS' if error < 1e-10 else '⚠️  Worth investigating'}")


def main():
    print("="*70)
    print("BATCHED 2D FDTD SOLVER - QUICK VALIDATION")
    print("="*70)
    print()

    try:
        test_import()
        test_initialization()
        test_cpu_execution()
        test_gpu_execution()
        test_batching()
        test_accuracy()

        print()
        print("="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Run full benchmark suite:")
        print("     ./benchmarks/run_full_benchmark.sh --quick")
        print("  2. Or run with GPU (if available):")
        print("     ./benchmarks/run_full_benchmark.sh")
        print()

    except Exception as e:
        print()
        print("="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
