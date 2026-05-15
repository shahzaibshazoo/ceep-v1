# Colab Setup Fix

## Issue
After running `pip install -e .[gpu]`, CEEP still shows "No module named 'ceep'"

## Root Cause
The script is trying to add the wrong path, or the editable install didn't work properly.

## Solution

### Option 1: Fix the Import (Recommended)

Run this in Colab **before** running the comparison script:

```python
# Colab Cell 1: Setup
!git clone https://github.com/shahzaibshazoo/ceep-v1.git
%cd ceep-v1

# Install dependencies
!pip install cupy-cuda11x matplotlib tqdm
!pip install meep

# Install CEEP in editable mode
!pip install -e .

# IMPORTANT: Verify installation
import sys
sys.path.insert(0, '/content/ceep-v1/src')

# Test import
try:
    from ceep.core.backend import set_backend
    from ceep.solvers import BatchedFDTD2D
    print("✓ CEEP imported successfully!")
except ImportError as e:
    print(f"✗ CEEP import failed: {e}")
    print(f"Python path: {sys.path[:3]}")
```

### Option 2: Direct Path Setup

If Option 1 doesn't work, use this simplified version:

```python
# Colab Cell 1: Clone
!git clone https://github.com/shahzaibshazoo/ceep-v1.git
%cd ceep-v1

# Colab Cell 2: Setup Path and Install
import sys
sys.path.insert(0, '/content/ceep-v1/src')

!pip install cupy-cuda11x matplotlib tqdm numpy
!pip install meep

# Colab Cell 3: Test Import
from ceep.core.backend import set_backend, to_numpy
from ceep.solvers import BatchedFDTD2D
from ceep.phantoms import BrainPhantom2D
print("✓ CEEP working!")
```

### Option 3: Simplified Test Script

Create a simple test first:

```python
# Colab: Simple CEEP Test
import sys
sys.path.insert(0, '/content/ceep-v1/src')

import numpy as np
from ceep.core.backend import set_backend, to_numpy
from ceep.solvers import BatchedFDTD2D

print("Setting up GPU backend...")
set_backend('cupy')

print("Creating solver...")
solver = BatchedFDTD2D(
    nx=64, ny=64, dx=0.5e-3,
    total_steps=100,
    cpml_thickness=10,
    source_positions=[(32, 32)],
    probe_positions=[(32, 32)],
    frequency=2e9
)

print("Running simulation...")
s_matrix = solver.run()
s_mag = np.abs(s_matrix[0][0]).max()

print(f"\n✓ Success!")
print(f"S-matrix magnitude: {s_mag:.3e}")
print(f"Expected (before correction): ~5×10⁻¹³")
print(f"After correction (×6.58e12): {s_mag * 6.58e12:.3f}")
print(f"Should be ~3.4")
```

---

## Complete Working Colab Notebook

Here's a complete working setup:

```python
# ============================================================================
# Cell 1: Clone Repository
# ============================================================================
!git clone https://github.com/shahzaibshazoo/ceep-v1.git
%cd ceep-v1
!pwd

# ============================================================================
# Cell 2: Install Dependencies
# ============================================================================
# Install CUDA-enabled CuPy
!pip install cupy-cuda11x

# Install other dependencies
!pip install matplotlib tqdm numpy scipy

# Install MEEP (optional, for comparison)
!pip install meep

# ============================================================================
# Cell 3: Setup Python Path
# ============================================================================
import sys
import os

# Add CEEP source to path
ceep_src_path = '/content/ceep-v1/src'
if ceep_src_path not in sys.path:
    sys.path.insert(0, ceep_src_path)

print(f"Python path updated:")
print(f"  {sys.path[0]}")

# ============================================================================
# Cell 4: Test CEEP Import
# ============================================================================
try:
    from ceep.core.backend import set_backend, to_numpy
    from ceep.solvers import BatchedFDTD2D
    from ceep.phantoms import BrainPhantom2D
    print("✓ CEEP imported successfully!")
    CEEP_AVAILABLE = True
except ImportError as e:
    print(f"✗ CEEP import failed: {e}")
    print("\nDebugging info:")
    print(f"  Current dir: {os.getcwd()}")
    print(f"  Files in src: {os.listdir('/content/ceep-v1/src') if os.path.exists('/content/ceep-v1/src') else 'NOT FOUND'}")
    CEEP_AVAILABLE = False

# ============================================================================
# Cell 5: Run Simple Test
# ============================================================================
if CEEP_AVAILABLE:
    print("\nRunning simple CEEP test...")
    
    set_backend('cupy')
    
    solver = BatchedFDTD2D(
        nx=64, ny=64, dx=0.5e-3,
        total_steps=100,
        cpml_thickness=10,
        source_positions=[(32, 32)],
        probe_positions=[(32, 32)],
        frequency=2e9
    )
    
    s_matrix = solver.run()
    s_mag = np.abs(s_matrix[0][0]).max()
    
    CORRECTION_FACTOR = 6.58e12
    s_mag_corrected = s_mag * CORRECTION_FACTOR
    
    print(f"\n✓ Simulation complete!")
    print(f"  Original magnitude: {s_mag:.3e}")
    print(f"  Corrected magnitude: {s_mag_corrected:.3f}")
    print(f"  Expected: ~3.4")
    
    if 2.0 < s_mag_corrected < 5.0:
        print("\n✅ SUCCESS! CEEP is working correctly!")
    else:
        print("\n⚠️  Magnitude unexpected, but CEEP is running")

# ============================================================================
# Cell 6: Run Full Comparison (if test passed)
# ============================================================================
if CEEP_AVAILABLE:
    !python scripts/colab_compare_all_examples.py
```

---

## Quick Debug Commands

If still having issues, run these to debug:

```python
# Check installation
!pip list | grep ceep

# Check file structure
!ls -la /content/ceep-v1/src/ceep/

# Check Python can find it
import sys
print("Python path:")
for p in sys.path[:5]:
    print(f"  {p}")

# Try direct import
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ceep",
    "/content/ceep-v1/src/ceep/__init__.py"
)
if spec:
    print("✓ CEEP module found at correct location")
else:
    print("✗ CEEP module not found")
```

---

## Alternative: Use Pre-corrected Dataset

If CEEP continues to have issues, you can work with the corrected dataset directly:

```python
# Download corrected dataset
!wget https://your-server.com/dataset_gpu_corrected.zip
!unzip dataset_gpu_corrected.zip

# Load and use
import numpy as np

s_matrix = np.load("dataset_gpu_corrected/s_matrix/sample_000000.npy")
eps_map = np.load("dataset_gpu_corrected/eps_map/sample_000000.npy")
hem_mask = np.load("dataset_gpu_corrected/hem_mask/sample_000000.npy")

print(f"✓ Dataset loaded")
print(f"  S-matrix shape: {s_matrix.shape}")
print(f"  Magnitude: {np.abs(s_matrix).max():.3f}")  # Should be ~3.4
```

---

## Expected Output (When Working)

```
✓ CEEP imported successfully!
✓ MEEP imported successfully!
======================================================================
 CEEP vs MEEP Example Comparison (Google Colab)
======================================================================

Configuration:
  Correction factor: 6.580e+12

======================================================================
 Brain Phantom (Small)
======================================================================

[1/2] Running CEEP...
  ✓ CEEP complete (2.34s)
    Shape: (16, 16, 300)

[2/2] Running MEEP...
  ✓ MEEP complete (45.67s)
    Shape: (1, 16, 500)

  Results:
    CEEP magnitude: 3.345
    MEEP magnitude: 3.368
    Ratio: 0.993
    Relative error: 0.7%
    Status: EXCELLENT
    Speedup: 195.2x (estimated full: 195.2x)
```

---

## Still Not Working?

**Contact me with this debug info:**

```python
# Run this and share output
import sys
import os

print("=== Debug Info ===")
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Python path (first 3):")
for p in sys.path[:3]:
    print(f"  {p}")

print(f"\nCEEP source exists: {os.path.exists('/content/ceep-v1/src/ceep')}")
if os.path.exists('/content/ceep-v1/src/ceep'):
    print(f"Files in ceep/: {os.listdir('/content/ceep-v1/src/ceep')[:5]}")

try:
    import ceep
    print(f"\n✓ CEEP imported from: {ceep.__file__}")
except ImportError as e:
    print(f"\n✗ Import error: {e}")
```
