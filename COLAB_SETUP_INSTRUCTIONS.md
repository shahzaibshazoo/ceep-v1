# CEEP Library - Complete Colab Setup

## For New Users - No Prior Knowledge Required

This guide will help you run all CEEP examples in Google Colab without any errors.

---

## 🚀 Quick Start (5 Minutes)

Copy and paste these cells into a new Colab notebook **in order**:

### Cell 1: Clone Repository
```python
# Go to content directory and clone CEEP
import os
os.chdir('/content')

# Remove old version if exists
!rm -rf ceep-v1

# Clone fresh from GitHub
!git clone https://github.com/shahzaibshazoo/ceep-v1.git

# Enter directory
%cd ceep-v1

# Verify location
!pwd
```

**Expected output:**
```
/content/ceep-v1
```

---

### Cell 2: Install Dependencies
```python
# Install CUDA 12 compatible CuPy and other dependencies
!pip install cupy-cuda12x matplotlib numpy tqdm -q

print("✅ Dependencies installed")
print("⚠️  If you see CUDA version warnings, restart runtime:")
print("   Runtime → Restart Runtime")
```

**Important:** If you see errors about CUDA versions, go to:
- **Runtime → Restart Runtime** (not Restart and run all)
- Then continue from Cell 3

---

### Cell 3: Run Complete Test Suite
```python
# Run all examples and verify correctness
!python COLAB_NEW_USER_TEST.py
```

This will:
- Test 8 different configurations
- Compare results with MEEP reference
- Show detailed performance statistics
- Verify S-parameters are correct (no correction factors needed)

**Expected output:**
```
================================================================================
 CEEP Library - Complete Test Suite
================================================================================

[1/4] Setting up environment...
  ✓ Found CEEP at: /content/ceep-v1/src
  ✓ CEEP imported successfully

[2/4] Configuring GPU backend...
  ✓ CuPy GPU backend ready

[3/4] Loading MEEP reference...
  MEEP reference: 3.368
  Tolerance: ±5.0%

[4/4] Running test suite...
================================================================================

📊 Test 1: Empty Domain (Basic Validation)
--------------------------------------------------------------------------------
  Config: 64×64 grid, 1 antenna
  Runtime: 0.85s
  S-parameter: 3.367
  MEEP reference: 3.368
  Error: 0.0%
  Status: ✅ PASS

... (more tests) ...

================================================================================
 FINAL SUMMARY
================================================================================

Test Results: 8/8 tests passed

... (detailed results) ...

✅ ALL TESTS PASSED!

CEEP library is validated and production-ready!

No correction factors needed - S-parameters are correct out of the box.
```

---

## 📋 What Gets Tested

1. **Empty Domain** - Basic validation against MEEP (should be 3.368)
2. **Large Grid** - 128×128 grid performance
3. **2-Antenna Array** - Multi-antenna configuration
4. **4-Antenna Square** - Square array pattern
5. **8-Antenna Circular** - Circular array pattern
6. **Brain Phantom Simple** - Single antenna in brain tissue
7. **Brain Phantom + 8-Antenna** - Full imaging configuration
8. **High Frequency** - 5 GHz operation

---

## ✅ Success Criteria

You'll know it's working when you see:

```
Test Results: 8/8 tests passed

CEEP vs MEEP Validation:
  MEEP reference: 3.368
  CEEP (Test 1): 3.367
  Error: 0.0%
  🎯 EXCELLENT - Within 1% of MEEP!

✅ ALL TESTS PASSED!
```

---

## 🔧 Troubleshooting

### Problem 1: "No module named 'ceep'"

**Solution:**
```python
# Check current directory
!pwd
# Should show: /content/ceep-v1

# If not, run:
%cd /content/ceep-v1
```

---

### Problem 2: "libnvrtc.so.11.2: cannot open shared object file"

**Cause:** Wrong CuPy version installed

**Solution:**
```python
# Uninstall old version
!pip uninstall cupy-cuda11x -y

# Install correct version for CUDA 12
!pip install cupy-cuda12x -q

# Restart runtime (important!)
# Runtime → Restart Runtime

# Then run test again
!python COLAB_NEW_USER_TEST.py
```

---

### Problem 3: Tests failing with wrong magnitudes

**Cause:** Old version of repository

**Solution:**
```python
# Pull latest changes
%cd /content/ceep-v1
!git pull origin master

# Run tests again
!python COLAB_NEW_USER_TEST.py
```

---

### Problem 4: Import errors for BrainPhantom2D

**Cause:** Repository not updated

**Solution:**
```python
# Re-clone fresh copy
%cd /content
!rm -rf ceep-v1
!git clone https://github.com/shahzaibshazoo/ceep-v1.git
%cd ceep-v1
!python COLAB_NEW_USER_TEST.py
```

---

## 📊 Using CEEP for Your Own Simulations

After all tests pass, you can use CEEP like this:

```python
from ceep.core.backend import set_backend
from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
from ceep.phantoms import BrainPhantom2D
import numpy as np

# Setup GPU
set_backend('cupy')

# Create brain phantom
phantom = BrainPhantom2D(
    nx=64, ny=64,
    dx=0.5e-3,
    hemorrhage_location=(1.0, 0.5),  # cm from center
    hemorrhage_radius=1.0,           # cm
    use_gabriel_database=False
)

# Setup antenna array (8 antennas in circle)
nx, ny = 64, 64
cx, cy = nx // 2, ny // 2
radius = 25  # grid cells

angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
positions = [
    (int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a))) 
    for a in angles
]

# Create solver
solver = BatchedFDTD2D(
    nx=nx, ny=ny,
    dx=0.5e-3,
    total_steps=150,
    cpml_thickness=10,
    source_positions=positions,
    probe_positions=positions,
    frequency=2e9  # 2 GHz
)

# Set phantom
solver.set_phantom(phantom)

# Run simulation
s_matrix = solver.run()

# Extract results
# s_matrix[tx_idx][rx_idx] = time-domain signal
s11 = s_matrix[0][0]  # Reflection at antenna 0
s12 = s_matrix[0][1]  # Transmission from 0 to 1

print(f"S11 magnitude: {np.abs(s11).max():.3f}")
print(f"S12 magnitude: {np.abs(s12).max():.3f}")
```

**Important:** S-parameters are already correct! No correction factors needed.

---

## 🎯 Expected Performance

On a typical Colab GPU (T4):

- **64×64 grid, 100 steps**: ~0.5-1.0 seconds
- **128×128 grid, 200 steps**: ~2-3 seconds
- **8-antenna array, 150 steps**: ~3-5 seconds

Total test suite: ~10-15 seconds

---

## 📚 What You Get

1. **MEEP-validated results** - S-parameters match industry-standard MEEP solver
2. **No correction factors** - Results are correct out of the box
3. **GPU acceleration** - 10-18x faster than sequential simulation
4. **Production-ready** - Use directly for dataset generation and research

---

## 🆘 Still Having Issues?

Run this diagnostic cell and share the output:

```python
!pwd
!ls -la | head -10
!python --version
!pip show cupy-cuda12x
!nvidia-smi

import sys
print("\nPython path:")
for p in sys.path[:5]:
    print(f"  {p}")
```

---

## ✨ Next Steps

After all tests pass:

1. **Generate datasets** - Use CEEP to create training data
2. **Train models** - Feed S-parameters to neural networks
3. **Validate results** - Compare with MEEP reference
4. **Publish research** - CEEP outputs are publication-quality

---

## 📞 Support

- **GitHub Issues:** https://github.com/shahzaibshazoo/ceep-v1/issues
- **Documentation:** See README.md in repository
- **Examples:** See examples/ directory for more use cases

---

**Last Updated:** 2026-05-15  
**Version:** 1.0 (Production-ready with MEEP validation)
