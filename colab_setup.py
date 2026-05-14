#!/usr/bin/env python3
"""
CEEP Colab Setup Script
=======================

Run this at the start of any Google Colab session to install CEEP with GPU support.

Usage in Colab:
---------------
!wget https://raw.githubusercontent.com/shahzaibshazoo/ceep-v1/master/colab_setup.py
!python colab_setup.py

Or one-liner:
!pip install git+https://github.com/shahzaibshazoo/ceep-v1.git && pip install cupy-cuda12x

Author: Shahzaib Ur Rehman
Date: 2026-05-14
"""

import subprocess
import sys

def run_command(cmd):
    """Run shell command and print output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode

def main():
    print("="*60)
    print(" CEEP Colab Setup")
    print("="*60)
    print()

    # 1. Check GPU
    print("[1/4] Checking GPU availability...")
    ret = run_command("nvidia-smi")
    if ret != 0:
        print("⚠️  WARNING: No GPU detected!")
        print("   Go to Runtime → Change runtime type → GPU")
        sys.exit(1)
    print("✓ GPU detected\n")

    # 2. Install CuPy
    print("[2/4] Installing CuPy for GPU acceleration...")
    ret = run_command("pip install -q cupy-cuda12x")
    if ret != 0:
        print("❌ CuPy installation failed")
        sys.exit(1)
    print("✓ CuPy installed\n")

    # 3. Clone and install CEEP
    print("[3/4] Installing CEEP from GitHub...")
    run_command("rm -rf /content/ceep-v1")
    ret = run_command("git clone -q https://github.com/shahzaibshazoo/ceep-v1.git /content/ceep-v1")
    if ret != 0:
        print("❌ Git clone failed")
        sys.exit(1)

    ret = run_command("cd /content/ceep-v1 && pip install -q -e .")
    if ret != 0:
        print("❌ CEEP installation failed")
        sys.exit(1)
    print("✓ CEEP installed\n")

    # 4. Verify installation
    print("[4/4] Verifying installation...")
    verify_code = """
import sys
sys.path.insert(0, '/content/ceep-v1/src')
from ceep.core.backend import set_backend, print_backend_info
set_backend('cupy')
print_backend_info()
"""

    with open('/tmp/verify.py', 'w') as f:
        f.write(verify_code)

    ret = run_command("python /tmp/verify.py")
    if ret != 0:
        print("❌ Verification failed")
        sys.exit(1)

    print("\n" + "="*60)
    print(" ✓ CEEP Setup Complete!")
    print("="*60)
    print()
    print("Quick test:")
    print("  from ceep.core.backend import set_backend")
    print("  set_backend('cupy')")
    print("  print('GPU ready!')")
    print()

if __name__ == "__main__":
    main()
