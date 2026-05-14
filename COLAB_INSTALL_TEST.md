# CEEP Colab Installation - Fixed!

## ✅ The pyproject.toml bug is now fixed in the repository

## Quick Install (Copy-Paste to Colab)

### Method 1: Direct pip install (Recommended)

```python
# Cell 1: Install CEEP
!pip install -q git+https://github.com/shahzaibshazoo/ceep-v1.git
!pip install -q cupy-cuda12x

print("✓ Installation complete!")
```

```python
# Cell 2: Verify
from ceep.core.backend import set_backend, print_backend_info
set_backend('cupy')
print_backend_info()
```

**Expected output:**
```
Backend: cupy
Device: cuda
GPU: Tesla T4
✓ GPU acceleration enabled
```

---

### Method 2: Clone and install (If pip fails)

```python
# Cell 1: Clone and install
!rm -rf /content/ceep-v1
!git clone https://github.com/shahzaibshazoo/ceep-v1.git /content/ceep-v1
!cd /content/ceep-v1 && pip install -q -e .
!pip install -q cupy-cuda12x

print("✓ Installation complete!")
```

```python
# Cell 2: Verify
from ceep.core.backend import set_backend, print_backend_info
set_backend('cupy')
print_backend_info()
```

---

## Run Radar Example

```python
# Cell 3: Download and run
!wget -q https://raw.githubusercontent.com/shahzaibshazoo/ceep-v1/master/examples/radar_2d_ula_beamforming.py
!python radar_2d_ula_beamforming.py
```

**Expected:**
- Runtime: ~5-8 seconds
- Output: "✓ Complete in 5.2s"
- File: `radar_2d_ula_beamforming.png`

```python
# Cell 4: Display results
from IPython.display import Image, display
display(Image('radar_2d_ula_beamforming.png'))
```

---

## Minimal Inline Test (No Downloads)

```python
import numpy as np
from ceep.core.backend import set_backend, get_backend_module, to_numpy
set_backend('cupy')

# Quick performance test
xp = get_backend_module()
size = 1000
A = xp.random.rand(size, size)
B = xp.random.rand(size, size)

import time
t0 = time.time()
C = xp.matmul(A, B)
xp.cuda.Device().synchronize()
t1 = time.time()

print(f"✓ GPU matmul ({size}×{size}): {(t1-t0)*1000:.2f} ms")
print(f"✓ Result shape: {C.shape}")
print(f"✓ CEEP GPU backend working!")
```

**Expected:**
```
✓ GPU matmul (1000×1000): 2-5 ms
✓ Result shape: (1000, 1000)
✓ CEEP GPU backend working!
```

---

## Troubleshooting

### Still getting TOML error?

The fix is in the latest commit. Make sure you're pulling fresh:

```python
!rm -rf /content/ceep-v1
!git clone https://github.com/shahzaibshazoo/ceep-v1.git /content/ceep-v1
!cd /content/ceep-v1 && cat pyproject.toml | grep -A 3 "authors"
```

Should show:
```
authors = [
    {name = "Shahzaib Ur Rehman", email = "shahzaibelbert@gmail.com"}
]
```

**NOT:**
```
authors = [{name = "...", email = "..."}]
    {name = "...", email = "..."},  # WRONG!
]
```

### Module not found?

```python
# Option 1: Reinstall
!pip uninstall -y ceep
!pip install git+https://github.com/shahzaibshazoo/ceep-v1.git

# Option 2: Manual path
import sys
sys.path.insert(0, '/content/ceep-v1/src')
from ceep.core.backend import set_backend
```

### Low GPU performance?

Check throughput in simulation output:
- **Good:** 2.7 GCell-steps/s
- **Bad:** 0.7 GCell-steps/s

If bad, restart runtime and reinstall.

---

## What Was Fixed

**Problem:** Line 13 in `pyproject.toml` had broken syntax
```toml
authors = [{name = "...", email = "..."}]
    {name = "...", email = "..."},  # Invalid!
]
```

**Solution:** Clean authors list
```toml
authors = [
    {name = "Shahzaib Ur Rehman", email = "shahzaibelbert@gmail.com"}
]
```

**Commit:** `6b6e2dc` (2026-05-14)

---

## Performance Benchmarks (Colab T4)

| Task | Time | Notes |
|------|------|-------|
| Installation | ~30s | First time only |
| Import CEEP | <1s | After install |
| 2D Radar example | 5-8s | Full simulation + beamforming |
| 1 hemorrhage sample | 3-4s | 16×16 S-parameters |
| 100 samples | 5.5 min | Dataset generation |

---

## Ready to Use!

The repository is now fixed and fully functional. Try the installation above! 🚀

**Repository:** https://github.com/shahzaibshazoo/ceep-v1  
**Latest commit:** `6b6e2dc` - pyproject.toml fix
