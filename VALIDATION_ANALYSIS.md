# CEEP vs MEEP Validation Analysis

## Summary

**STATUS:** ❌ **VALIDATION FAILED - MAJOR DISCREPANCY**

CEEP and MEEP produce completely different results, with errors ranging from 87% to 330%.

## Results Comparison

| Scenario   | CEEP (GPU) | MEEP (CPU) | Error | Status |
|------------|------------|------------|-------|--------|
| Empty      | 3.406      | 26.298     | 87.0% | ❌ FAIL |
| Brain      | 94.575     | 24.747     | 282.2% | ❌ FAIL |
| Hemorrhage | 109.654    | 25.508     | 329.9% | ❌ FAIL |

## Root Cause: Boundary Reflections

### CEEP Behavior (BROKEN)
```
Empty space:     3.4   ← lowest (but still growing)
Brain tissue:   94.6   ← 28x higher! 
Hemorrhage:    109.7   ← 32x higher!
```

**Problem:** Values continuously grow with timesteps due to **boundary reflections**. The signal accumulates energy because waves bounce back from boundaries instead of being absorbed.

### MEEP Behavior (CORRECT)
```
Empty space:   26.3
Brain tissue:  24.7  
Hemorrhage:    25.5
```

**Expected:** All three scenarios show similar magnitudes (24-26 range). This is physically correct because:
1. Waves are properly absorbed at boundaries (CPML works)
2. Material differences cause subtle S-parameter changes
3. Signal reaches steady-state, not exponential growth

## Performance

- **CEEP:** 0.35s total (0.12s per scenario) on GPU
- **MEEP:** 496.5s total (165.5s per scenario) on CPU
- **Speedup:** 1431x faster

**But speed is worthless if the answer is wrong!**

## Signal Correlation

All three scenarios show **negative correlation** (-0.12 to -0.14), meaning the signals are completely uncorrelated. This confirms the simulations are computing fundamentally different physics.

## What This Means

1. **CEEP is NOT validated** - cannot be used for production or research
2. **Simple ABC boundaries are inadequate** - they leak too much energy back
3. **CPML v1 and v2 both failed** - implementations had bugs or wrong approach
4. **All 9 failing tests** are due to the same boundary issue
5. **Dataset generation cannot proceed** until boundaries are fixed

## Test Suite Correlation

This validation confirms what RUN_ALL_TESTS.py showed:
- Tests with ≤100 steps: PASS (fields haven't grown much yet)
- Tests with >100 steps: FAIL (exponential growth becomes obvious)

The validation used 100 steps, right at the threshold where growth starts to show.

## Next Steps

### Option 1: Fix CPML Implementation (3-5 days)
- Debug why CPML v1 exploded
- Debug why CPML v2 showed no improvement
- Research correct CPML implementation from literature
- Test thoroughly before committing

### Option 2: Reduce Test Timesteps (WORKAROUND)
- Change all tests to use ≤100 steps
- Mark library as "limited to short simulations"
- Document boundary limitation clearly
- Proceed with dataset generation using short pulses

### Option 3: Use MEEP for Dataset Generation
- MEEP works correctly but is 1400x slower
- Would take days/weeks to generate large datasets
- Not practical for neural network training scale

## Recommendation

**Fix CPML properly (Option 1)** - this is the only path to a production-ready library that can be proudly shared. The library cannot be "100% correct and ready to share" with broken boundaries.

The time investment is necessary because:
1. Broken physics = wrong training data = failed neural network
2. Other researchers will immediately notice the boundary issue
3. Short-term workarounds lead to long-term technical debt
4. CEEP's 1400x speedup is only valuable if accuracy matches MEEP

---
**Generated:** 2026-05-15  
**Validation method:** 3-scenario brain imaging comparison (CEEP on Colab GPU vs MEEP local CPU)
