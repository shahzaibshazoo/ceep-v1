#!/usr/bin/env python3
"""
GPU/CUDA Environment Verification Script
=========================================

Checks system readiness for NeuroWave development:
- Python version
- CUDA toolkit availability
- GPU device information
- PyTorch CUDA support
- CuPy availability
- NumPy/Numba availability

Usage:
    python scripts/check_gpu.py
"""

import sys
import platform


def check_python() -> None:
    """Check Python version."""
    print("=" * 60)
    print("Python Environment")
    print("=" * 60)
    print(f"  Version:  {sys.version}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Arch:     {platform.machine()}")
    v = sys.version_info
    if v.major >= 3 and v.minor >= 9:
        print("  Status:   ✅ Python 3.9+ requirement met")
    else:
        print("  Status:   ❌ Python 3.9+ required")
    print()


def check_numpy() -> None:
    """Check NumPy availability."""
    print("-" * 60)
    print("NumPy")
    print("-" * 60)
    try:
        import numpy as np
        print(f"  Version:  {np.__version__}")
        print("  Status:   ✅ Available")
    except ImportError:
        print("  Status:   ❌ Not installed (pip install numpy)")
    print()


def check_pytorch() -> None:
    """Check PyTorch and CUDA support."""
    print("-" * 60)
    print("PyTorch")
    print("-" * 60)
    try:
        import torch
        print(f"  Version:       {torch.__version__}")
        print(f"  CUDA available:{torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version:  {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  GPU count:     {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}:         {props.name}")
                print(f"    Memory:      {props.total_mem / 1e9:.1f} GB")
                print(f"    Compute:     {props.major}.{props.minor}")
                print(f"    SM count:    {props.multi_processor_count}")
            print("  Status:        ✅ CUDA support available")
        else:
            print("  Status:        ⚠️  No CUDA support (CPU-only mode)")
    except ImportError:
        print("  Status:        ❌ Not installed (pip install torch)")
    print()


def check_cupy() -> None:
    """Check CuPy availability."""
    print("-" * 60)
    print("CuPy")
    print("-" * 60)
    try:
        import cupy as cp
        print(f"  Version:  {cp.__version__}")
        print(f"  CUDA:     {cp.cuda.runtime.runtimeGetVersion()}")
        print("  Status:   ✅ Available")
    except ImportError:
        print("  Status:   ⚠️  Not installed (optional: pip install cupy-cuda12x)")
    except Exception as e:
        print(f"  Status:   ⚠️  Installed but error: {e}")
    print()


def check_numba() -> None:
    """Check Numba availability."""
    print("-" * 60)
    print("Numba")
    print("-" * 60)
    try:
        import numba
        print(f"  Version:  {numba.__version__}")
        try:
            from numba import cuda as numba_cuda
            cuda_available = numba_cuda.is_available()
            print(f"  CUDA:     {cuda_available}")
            print(f"  Status:   {'✅' if cuda_available else '⚠️'} "
                  f"{'CUDA available' if cuda_available else 'No CUDA (CPU only)'}")
        except (ImportError, AttributeError):
            print("  CUDA:     Not available (numba.cuda module missing)")
            print("  Status:   ⚠️  Numba installed without CUDA support")
    except ImportError:
        print("  Status:   ⚠️  Not installed (optional: pip install numba)")
    print()


def check_system_cuda() -> None:
    """Check system CUDA toolkit."""
    import subprocess
    print("-" * 60)
    print("System CUDA Toolkit")
    print("-" * 60)
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Extract version from nvcc output
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    print(f"  {line.strip()}")
            print("  Status:   ✅ nvcc available")
        else:
            print("  Status:   ⚠️  nvcc not found in PATH")
    except FileNotFoundError:
        print("  Status:   ⚠️  nvcc not found in PATH")
    except Exception as e:
        print(f"  Status:   ⚠️  Error: {e}")

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  nvidia-smi: {result.stdout.strip()}")
        else:
            print("  nvidia-smi: Not available")
    except (FileNotFoundError, Exception):
        print("  nvidia-smi: Not available")
    print()


def main() -> None:
    """Run all environment checks."""
    print()
    print("🧠⚡ NeuroWave — GPU Environment Check")
    print("=" * 60)
    print()

    check_python()
    check_numpy()
    check_pytorch()
    check_cupy()
    check_numba()
    check_system_cuda()

    print("=" * 60)
    print("Environment check complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
