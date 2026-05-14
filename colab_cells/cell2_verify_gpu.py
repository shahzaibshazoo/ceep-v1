# Cell 2: Verify GPU
# ===================

import cupy as cp
import numpy as np
import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("  GPU Information")
print("=" * 60)
print(f"  CuPy version: {cp.__version__}")
props = cp.cuda.runtime.getDeviceProperties(0)
print(f"  GPU: {props['name'].decode()}")
mem = cp.cuda.Device().mem_info
print(f"  Memory: {mem[0]/1e9:.1f} GB free / {mem[1]/1e9:.1f} GB total")
print("=" * 60)
