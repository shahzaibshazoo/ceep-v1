# GPU Dataset Diagnosis & Fix

## Date: 2026-05-15

## Summary

The GPU dataset (`dataset_gpu (4).zip`) has **critical issues** with S-parameter magnitudes being ~10 orders of magnitude too small.

---

## Dataset Analysis Results

### ✓ Structure: CORRECT
- 100 samples total
- 4 directories: `s_matrix/`, `eps_map/`, `hem_mask/`, `metadata/`
- All files present and loadable
- Hemorrhage distribution: ~70% with, ~30% without

### ❌ S-Matrix Values: TOO SMALL

```
Expected S-parameter magnitude: ~1e-3 to 1e-1
Actual values in dataset:       ~5e-13

Ratio: 5.1e-10 (10 orders of magnitude too small!)
```

### Specific Numbers

| Sample | S-Matrix Max | Has Hemorrhage |
|--------|-------------|----------------|
| 000000 | 5.117e-13   | Yes            |
| 000001 | 5.069e-13   | Yes            |
| 000010 | 5.111e-13   | Yes            |
| 000050 | 5.060e-13   | Yes            |
| 000099 | 5.062e-13   | Yes            |

**Statistics:**
- Mean: 5.096e-13
- Std:  1.591e-15
- All samples have similar (wrong) magnitude

---

## Root Cause Analysis

### Possible Issues in CEEP Dataset Generation

1. **Source Amplitude Scaling**
   - FDTD source may have amplitude of 1.0 but need normalization
   - Missing scaling factor when converting fields to S-parameters

2. **S-Parameter Computation**
   - Formula: `S_ij = E_scattered / E_incident`
   - May be missing incident field normalization
   - Time-domain integration may be incorrect

3. **Field Extraction Timing**
   - S-parameters extracted before fields reach steady state
   - Time window too short for proper signal capture

4. **Geometry/Material Issues**
   - Permittivity values may be set incorrectly
   - Source not properly exciting the domain

---

## Verification Strategy

### 1. MEEP Reference Simulation ✓ IN PROGRESS

Running classical MEEP simulation with:
- Same geometry (16 antennas, circular array)
- Same frequency (2 GHz)
- Same head model (layered brain)
- Hemorrhage present

Expected output:
- S-parameter magnitude: ~1e-3
- This will confirm the expected scale

**Status:** Script running in background
- File: `scripts/meep_reference_simulation.py`
- Output: `meep_reference_s_matrix.npy`
- ETA: ~10-15 minutes for 16 antennas

### 2. Comparison Results

Will compare:
```
MEEP S-matrix magnitude:    [TO BE MEASURED]
CEEP S-matrix magnitude:    5.117e-13
Ratio (CEEP/MEEP):          [TO BE CALCULATED]
```

---

## Recommended Fixes

### Option 1: Fix CEEP Source Implementation ⭐ BEST

Look at `src/ceep/solvers/fdtd_2d_batched.py`:

```python
# Check _add_source() method
# Ensure source amplitude is properly scaled

# Example fix might be:
source_amplitude = 1.0  # Current
source_amplitude = 1.0e10  # May need scaling based on MEEP comparison
```

### Option 2: Post-Process Scaling ⚠️ WORKAROUND

If CEEP source cannot be easily fixed:

```python
# Scale dataset by measured correction factor
correction_factor = meep_magnitude / ceep_magnitude

for sample_file in dataset:
    s_matrix = np.load(sample_file)
    s_matrix_corrected = s_matrix * correction_factor
    np.save(sample_file, s_matrix_corrected)
```

**Pros:** Quick fix
**Cons:** Doesn't address root cause, may have frequency-dependent errors

### Option 3: Regenerate Dataset ✅ PROPER FIX

After fixing CEEP source:
1. Update `BatchedFDTD2D` with correct scaling
2. Re-run dataset generation script
3. Verify one sample against MEEP
4. Generate full 100 samples

---

## Testing Procedure

### Step 1: Single Sample Validation

```bash
# After MEEP completes
python3 scripts/compare_ceep_meep_sample.py
```

This will:
- Load MEEP reference result
- Generate one CEEP sample with same parameters
- Compare magnitudes side-by-side
- Calculate correction factor

### Step 2: Verify Hemorrhage Detection

```python
# Apply DAS imaging to corrected S-parameters
python3 scripts/verify_dataset_with_das.py
```

Check:
- Can hemorrhage be localized?
- Is SNR sufficient?
- Does it match ground truth `hem_mask`?

### Step 3: Full Dataset Generation

Once validated:
```bash
python3 scripts/generate_full_dataset.py --n_samples 100 --output dataset_gpu_v2
```

---

## Current Status

### ✅ Completed
1. API fixes for `BrainPhantom` and `set_phantom()`
2. Radar example bug fixes (X/Y swap)
3. Dataset structure validation
4. DAS imaging analysis script
5. MEEP reference simulation (running)

### 🔄 In Progress
1. MEEP simulation running (ETA: ~10-15 min)
2. Waiting for magnitude comparison

### 📋 Next Steps
1. Compare MEEP vs CEEP magnitudes
2. Identify exact correction factor
3. Fix CEEP source scaling
4. Regenerate one sample for validation
5. Full dataset regeneration

---

## Files Created

### Analysis Scripts
- `scripts/analyze_gpu_dataset.py` - Dataset validation
- `scripts/verify_dataset_with_das.py` - DAS imaging verification  
- `scripts/meep_reference_simulation.py` - MEEP reference
- `scripts/compare_ceep_meep_sample.py` - Side-by-side comparison

### Visualizations
- `dataset_analysis_sample_000000.png` - Detailed sample analysis
- `meep_reference_result.png` - MEEP simulation results (pending)

### Documentation
- `FIXES_APPLIED.md` - All bug fixes applied
- `QUICK_START_FIXED.md` - Quick start guide
- `DATASET_DIAGNOSIS.md` - This file

---

## Expected Timeline

1. **MEEP simulation completes:** ~15 minutes
2. **Identify fix in CEEP:** ~30 minutes  
3. **Test single sample:** ~5 minutes
4. **Regenerate dataset (GPU):** ~1-2 hours
5. **Validation:** ~30 minutes

**Total:** ~2-3 hours to corrected dataset

---

## Impact on Neural Network Training

### Current Dataset: ❌ NOT USABLE

The S-parameters are essentially noise-level signals (~1e-13). Training on this will result in:
- Model learns nothing (no signal to learn from)
- Random predictions
- Cannot distinguish hemorrhage vs no hemorrhage

### Corrected Dataset: ✓ WILL WORK

With proper S-parameter magnitudes:
- Clear signal difference between hemorrhage/no-hemorrhage
- Spatial information preserved
- Network can learn to localize hemorrhages

---

## References

**CEEP Source Code:**
- `src/ceep/solvers/fdtd_2d_batched.py` - Main solver
- `src/ceep/phantoms/head_models.py` - Brain phantom

**MEEP Documentation:**
- https://meep.readthedocs.io/en/latest/
- S-parameter computation examples

**Expected S-parameter Values:**
- Microwave imaging literature: 1e-3 to 1e-1
- Brain tissue at 2 GHz: ε_r = 40-68

---

## Contact

For questions about this diagnosis:
- Developer: Shahzaib Ur Rehman
- Analysis: Claude (Anthropic)
- Date: 2026-05-15

---

**Status:** Awaiting MEEP reference simulation results to determine exact correction factor.
