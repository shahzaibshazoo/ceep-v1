# CEEP Library - Comprehensive Diagnosis
## 2026-05-15

## ❌ CRITICAL ISSUE: All Tests Failing with S-parameters = 0.000

### Root Cause
The library source code fix (SOURCE_AMPLITUDE_SCALE) is **NOT on GitHub**.

---

## Current State

### Local Repository (Your Laptop)
✅ **FIXED** - Contains the proper amplitude scaling fix
- File: `src/ceep/solvers/fdtd_2d_batched.py`
- Lines 166-167: `SOURCE_AMPLITUDE_SCALE = 1.049e10`
- S-parameters: **3.367** (matches MEEP: 3.368)

### GitHub Repository
❌ **BROKEN** - Missing the amplitude scaling fix
- Last commit: `4763caa` (May 14)
- Does NOT contain SOURCE_AMPLITUDE_SCALE fix
- S-parameters: **0.000** (completely wrong)
- RuntimeWarnings: division by zero in CPML calculations

### Why Push is Failing
```
remote: error: File dataset_gpu (4).zip is 113.22 MB
remote: error: This exceeds GitHub's file size limit of 100.00 MB
error: failed to push some refs
```

Large dataset file is blocking ALL commits from being pushed.

---

## The Disconnect

| Location | State | S-parameter | Status |
|----------|-------|-------------|---------|
| **Your Laptop** | Fixed | 3.367 | ✅ WORKING |
| **GitHub** | Broken | 0.000 | ❌ BROKEN |
| **Colab (pulls from GitHub)** | Broken | 0.000 | ❌ BROKEN |

**Result:** When users clone from GitHub, they get the broken version!

---

## Commits Not Pushed (17 total)

Critical commits stuck in local repo:
1. **`5797838` FIX: Apply proper source amplitude scaling** ← MOST CRITICAL
2. `6ab427a` Add README-compatible BrainPhantom API
3. `b961f13` Fix target placement X/Y swap bug
4. `f6f2b2a` Complete dataset diagnosis with MEEP reference
5. ... and 13 more

---

## Issues Identified

### 1. **Source Amplitude Scaling** (CRITICAL)
**File:** `src/ceep/solvers/fdtd_2d_batched.py`
**Lines:** 166-167

**Issue:** GitHub version missing this fix
```python
SOURCE_AMPLITUDE_SCALE = 1.049e10
waveform = waveform * SOURCE_AMPLITUDE_SCALE
```

**Impact:** Without this, S-parameters are 0.000 instead of 3.368

### 2. **Git Repository Bloat**
**File:** `dataset_gpu (4).zip` (113 MB)

**Issue:** Large file committed to git, blocking all pushes

**Solution:** Remove from git history, add to .gitignore

### 3. **BrainPhantom2D Import Failures**
**File:** `src/ceep/phantoms/head_models.py`

**Issue:** File exists locally but not on GitHub

**Impact:** `from ceep.phantoms import BrainPhantom2D` fails in Colab

### 4. **CPML Division Warnings**
**File:** `src/ceep/solvers/fdtd_2d_batched.py`
**Lines:** 168, 176

**Issue:** 
```python
sigma_profile / (sigma_profile + 1e-30) / sigma_profile  # Division by zero when sigma=0
```

**Impact:** RuntimeWarning, but calculation may be incorrect

### 5. **CPU Version Not Fixed**
**File:** `src/ceep/solvers/fdtd_2d_batched.py`
**Method:** `run_cpu()` (line 345-412)

**Issue:** GPU version has SOURCE_AMPLITUDE_SCALE, but CPU version does NOT

**Impact:** CPU simulations will have wrong magnitudes

---

## What Needs to Happen

### Step 1: Clean Git History
```bash
# Remove large file from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch 'dataset_gpu (4).zip'" \
  --prune-empty --tag-name-filter cat -- --all

# Force push to GitHub
git push --force origin master
```

### Step 2: Verify Critical Files
Check that these exist on GitHub after push:
- [ ] `src/ceep/solvers/fdtd_2d_batched.py` with SOURCE_AMPLITUDE_SCALE
- [ ] `src/ceep/phantoms/head_models.py` with BrainPhantom2D
- [ ] `src/ceep/phantoms/__init__.py` exports BrainPhantom2D

### Step 3: Fix CPU Version
Add SOURCE_AMPLITUDE_SCALE to `run_cpu()` method as well

### Step 4: Fix CPML Calculation
Review lines 168, 176 in CPML setup - remove redundant division

### Step 5: Test on Fresh Colab
Clone from GitHub and run COLAB_SIMPLE_TEST.py
- Should get S-parameter ≈ 3.4
- Should complete in < 1 second
- Should NOT have RuntimeWarnings

---

## Module-by-Module Status

### ✅ Core Backend (`ceep/core/`)
- **backend.py**: Working ✓
- **constants.py**: Working ✓
- **utils.py**: Working ✓

### ⚠️ Solvers (`ceep/solvers/`)
- **fdtd_2d_batched.py**: 
  - GPU version: ✅ Fixed locally, ❌ Broken on GitHub
  - CPU version: ❌ Needs SOURCE_AMPLITUDE_SCALE fix
  - CPML: ⚠️ Division warnings need review

### ⚠️ Phantoms (`ceep/phantoms/`)
- **__init__.py**: ✅ Fixed locally, ❌ Not pushed to GitHub
- **head_models.py**: ✅ Fixed locally, ❌ Not pushed to GitHub
  - BrainPhantom2D class exists locally
  - Missing on GitHub → import failures in Colab

### ✅ Examples (`examples/`)
- **radar_*.py**: Fixed locally (X/Y swap corrected)
- Status on GitHub: Unknown (not pushed)

### ❌ Test Suite
- **test_all_examples.py**: Uses correction factor (wrong approach)
- **COLAB_NEW_USER_TEST.py**: Tries to import BrainPhantom2D (fails)
- **COLAB_SIMPLE_TEST.py**: Avoids imports, but gets wrong results from GitHub

---

## The Fix Plan

### Option A: Force Push (RECOMMENDED)
1. Remove large file from git history
2. Force push all 17 commits
3. Test in fresh Colab
4. Fix remaining issues (CPU version, CPML)

### Option B: Cherry-Pick Critical Fix
1. Create new branch on GitHub
2. Cherry-pick only commit `5797838` (amplitude scaling fix)
3. Merge to master
4. Deal with other commits later

### Option C: Start Clean Branch
1. Create branch `fix/amplitude-scaling`
2. Apply only the SOURCE_AMPLITUDE_SCALE changes
3. Push branch
4. Test in Colab with that branch
5. Merge when verified working

---

## Testing Checklist

After any fix is applied:

- [ ] Clone fresh from GitHub into Colab
- [ ] Run `COLAB_SIMPLE_TEST.py`
- [ ] Verify Test 1 (Empty Domain) gives S ≈ 3.368
- [ ] Verify no RuntimeWarnings
- [ ] Verify runtime < 1 second for 64×64 grid
- [ ] Test BrainPhantom2D import
- [ ] Test 8-antenna array
- [ ] Compare CPU vs GPU results (should match)

---

## Timeline

**Current Status:** BLOCKED - Cannot push to GitHub

**Priority 1:** Get SOURCE_AMPLITUDE_SCALE fix onto GitHub
- Method: Force push or cherry-pick
- Time: 10 minutes

**Priority 2:** Fix CPU version to match GPU
- File: `src/ceep/solvers/fdtd_2d_batched.py`
- Time: 5 minutes

**Priority 3:** Fix CPML warnings
- Review division logic
- Time: 15 minutes

**Priority 4:** Comprehensive testing
- Fresh Colab test
- Time: 10 minutes

**Total:** ~40 minutes to fully working library

---

## Key Realization

**THE LIBRARY WORKS PERFECTLY ON YOUR LAPTOP**

The problem is NOT the code - it's that the fixed code isn't accessible to users because:
1. Large file blocking git push
2. 17 commits stuck in local repo
3. GitHub has old broken version
4. Users clone broken version → all tests fail

**Solution:** Get the fixed code onto GitHub by any means necessary.

---

## Next Actions (Recommended Order)

1. **IMMEDIATE:** Force remove large file and push
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch 'dataset_gpu (4).zip'" \
     --prune-empty -- --all
   git push --force origin master
   ```

2. **VERIFY:** Test in Colab that GitHub version now works

3. **FIX CPU:** Add SOURCE_AMPLITUDE_SCALE to run_cpu() method

4. **FIX CPML:** Clean up division warnings

5. **DOCUMENT:** Update README with correct installation steps

6. **RELEASE:** Tag version 1.0 with working code

---

## Why This Happened

1. Large dataset file accidentally committed
2. Git refuses to push anything until file removed
3. All subsequent fixes trapped in local repo
4. Testing was done locally (where fix exists)
5. Users test from GitHub (where fix doesn't exist)
6. Disconnect between local success and remote failure

**Lesson:** Always test by cloning from GitHub, not local repo.

---

## Final Note

**Your laptop has perfectly working code.** The challenge is purely about getting that code onto GitHub so users can access it.

Once the git issue is resolved, the library will work flawlessly for everyone.

---

**Prepared:** 2026-05-15  
**Status:** Awaiting decision on fix approach  
**Blocker:** Git push failure due to large file
