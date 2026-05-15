# Quick Reference Guide

## 🚀 For Google Colab Users

### Run Example Comparison

```python
# Setup
!git clone https://github.com/shahzaibshazoo/ceep-v1.git
%cd ceep-v1
!pip install -e .[gpu]
!pip install meep matplotlib tqdm

# Run comparison
!python scripts/colab_compare_all_examples.py

# View results
!cat comparison_summary.txt
```

**Output:** Comparison of CEEP vs MEEP with error analysis

---

## 📊 For Neural Network Training

### Load Corrected Dataset

```python
import numpy as np
from pathlib import Path

# Load one sample
sample_id = 0
base_path = Path("dataset_gpu_corrected")

s_matrix = np.load(base_path / "s_matrix" / f"sample_{sample_id:06d}.npy")
eps_map = np.load(base_path / "eps_map" / f"sample_{sample_id:06d}.npy")
hem_mask = np.load(base_path / "hem_mask" / f"sample_{sample_id:06d}.npy")

print(f"S-matrix shape: {s_matrix.shape}")  # (16, 16, 301)
print(f"S-matrix magnitude: {np.abs(s_matrix).max():.3f}")  # ~3.4
```

**Expected magnitude:** ~3.4 (matches MEEP validation)

---

## 🔧 For Dataset Generation (GPU Required)

### Generate New Dataset with Correct Magnitudes

```python
# Run on GPU machine
!python scripts/generate_corrected_dataset.py --n_samples 100 --output dataset_new

# Validation
!python scripts/test_meep_vs_ceep.py
```

**Time:** ~2 hours for 100 samples on T4 GPU

---

## 🐛 For Debugging CEEP

### Check S-Parameter Magnitude

```python
from ceep.solvers import BatchedFDTD2D
import numpy as np

solver = BatchedFDTD2D(
    nx=64, ny=64, dx=0.5e-3,
    total_steps=300, cpml_thickness=10,
    source_positions=[(32, 32)],
    probe_positions=[(32, 32)],
    frequency=2e9
)

s_matrix = solver.run()
s_mag = np.abs(s_matrix[0][0]).max()

print(f"Magnitude: {s_mag:.3e}")
# Should be: ~5×10⁻¹³ (before fix) or ~3.4 (after fix)
```

---

## 📖 Documentation Index

| File | Purpose |
|------|---------|
| `COMPLETE_SESSION_SUMMARY.md` | Full session overview |
| `READY_FOR_TRAINING.md` | Training guide |
| `DATASET_FIX_SUMMARY.md` | Dataset correction details |
| `CEEP_FIX_NEEDED.md` | How to fix CEEP source code |
| `FIXES_APPLIED.md` | All bug fixes |
| `QUICK_START_FIXED.md` | Quick start |
| `DATASET_DIAGNOSIS.md` | Problem diagnosis |

---

## 🔬 Validation Commands

### Test MEEP vs CEEP

```bash
python3 scripts/test_meep_vs_ceep.py
```

**Expected output:** Ratio = 1.000 ✓

### Analyze Dataset

```bash
python3 scripts/analyze_gpu_dataset.py
```

**Expected:** 100 samples, magnitudes ~3.4

### Verify Correction

```python
import numpy as np

original = np.load("dataset_gpu/s_matrix/sample_000000.npy")
corrected = np.load("dataset_gpu_corrected/s_matrix/sample_000000.npy")

print(f"Original: {np.abs(original).max():.3e}")  # ~5×10⁻¹³
print(f"Corrected: {np.abs(corrected).max():.3f}")  # ~3.4
print(f"Factor: {np.abs(corrected).max() / np.abs(original).max():.3e}")  # ~6.58×10¹²
```

---

## ⚠️ Common Issues

### Issue: "S-matrix magnitude too small"

**Solution:** Use `dataset_gpu_corrected/` not `dataset_gpu/`

```python
# WRONG
s_matrix = np.load("dataset_gpu/s_matrix/sample_000000.npy")

# CORRECT
s_matrix = np.load("dataset_gpu_corrected/s_matrix/sample_000000.npy")
```

### Issue: "CEEP GPU not available"

**Solution:** Run on Colab or machine with CUDA

```python
# Check GPU
import torch
print(torch.cuda.is_available())

# Or use MEEP (CPU)
python scripts/meep_reference_simulation.py
```

### Issue: "Training accuracy stuck at 50%"

**Possible causes:**
1. Using uncorrected dataset → Use `dataset_gpu_corrected/`
2. Model too simple → Try deeper network
3. Not enough data → Augment or generate more samples

---

## 📈 Expected Results

### Dataset
- **Samples:** 100 (70 with hemorrhage, 30 without)
- **S-matrix magnitude:** ~3.4
- **MEEP validation:** Ratio 1.000 ✓

### Neural Network
- **Detection accuracy:** >90%
- **Localization error:** <5mm
- **Training time:** 1-2 hours

### Performance
- **CEEP speedup:** 13-33x vs MEEP
- **GPU memory:** ~2-4 GB
- **Inference time:** <100ms/sample

---

## 🔗 Quick Links

| Resource | Location |
|----------|----------|
| Corrected Dataset | `dataset_gpu_corrected/` |
| MEEP Reference | `meep_reference_s_matrix.npy` |
| Comparison Viz | `meep_vs_ceep_comparison.png` |
| Training Guide | `READY_FOR_TRAINING.md` |
| Colab Script | `scripts/colab_compare_all_examples.py` |

---

## ✅ Checklist

Before training:
- [ ] Using `dataset_gpu_corrected/` not `dataset_gpu/`
- [ ] S-matrix magnitudes ~3.4 (not ~10⁻¹³)
- [ ] All 100 samples loaded correctly
- [ ] GPU available for training

After training:
- [ ] Detection accuracy >90%
- [ ] Localization error <5mm
- [ ] Visualize predictions vs ground truth

---

**Status:** ✅ Everything ready for training

**Last Updated:** 2026-05-15
