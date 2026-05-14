# Known Issues - NeuroWave

**Last Updated**: 2026-05-14

---

## Resolved: 3D Wave Propagation Timing Anomaly

**Status**: ✅ **RESOLVED** (2026-05-13)  
**Test**: `tests/test_fdtd_3d.py::test_3d_wave_propagation` — now PASSING  
**Root cause**: Test methodology error, not a solver bug

### Resolution

The 3D field update equations are **correct**. The apparent timing anomaly was caused by
placing a probe too close to the soft source (5 cells). In the near-field region, the
accumulated source injection energy dominates the pulse's propagating wavefront, making
`argmax(abs(data))` return the wrong timestep.

**Fix applied**: Moved test probes to 15 and 35 cells from the source (far-field region)
and used wavefront arrival detection instead of peak detection.

**Validation result**: Wavefront delay = 71 steps vs expected 69.3 steps (2.5% error).

### Previous Misdiagnosis

The original test placed a probe 5 cells from a soft source. With soft source injection
(`field[x,y,z] += value`), the source point acts as a continuously driven antenna.
Probes within ~10 cells see superposition of the outgoing wave AND the local source
field, making peak timing unreliable.

### Lesson Learned

When testing FDTD wave propagation with soft sources:
- Place probes at least 15 cells from the source
- Use wavefront arrival (threshold crossing) rather than absolute peak detection
- Use `delay_factor >= 5.0` so the pulse starts near zero

---

## Secondary Issue: 3D CPML Instability

**Status**: ✅ **RESOLVED** (2026-05-14)  
**Test**: `tests/test_fdtd_3d.py::test_3d_cpml_stability` — now PASSING  
**Root cause**: Two bugs — (1) H-field psi used backward diffs instead of forward, (2) CPML wrote corrections to cells outside the main update domain

### Resolution

Two bugs were identified and fixed:

**Bug 1: H-field finite difference direction mismatch**
```python
# Before (WRONG - backward differences):
psi = be * psi + ce * (grid.ez[:, j, :] - grid.ez[:, j-1, :])

# After (CORRECT - forward differences matching main solver):
psi = be * psi + ce * (grid.ez[:, j+1, :nz-1] - grid.ez[:, j, :nz-1])
```

**Bug 2: Index range overflow**
CPML corrections were applied to cells outside the main solver's update domain. For example, H-field updates at `hx[:, :-1, :-1]` but CPML was writing to `hx[:, j, nz-1]` which is never updated by the main loop — creating unmatched field values that fed back into E-field updates.

Fix: All CPML corrections restricted to exact same index ranges as main update:
- H-field: `hx[:, j, :nz-1]`, `hy[:nx-1, :, k]`, `hz[:nx-1, j, :]`
- E-field: corrections only at `[1:, 1:, 1:]`

### Validation

- 3D CPML stable over 3000+ steps
- Energy bounded < 1e-10 after 600 steps (source fully absorbed)
- All 6 field components remain finite
- test_3d_cpml_stability passes in full suite (95/95)

---

## Test Status Summary

| Test Category | Passing | Total | Status |
|--------------|---------|-------|--------|
| 2D Core FDTD | 37 | 37 | ✅ Perfect |
| Advanced FDTD | 37 | 37 | ✅ Perfect |
| 3D FDTD | 9 | 9 | ✅ Perfect |
| Biomedical | 12 | 12 | ✅ Perfect |
| **TOTAL** | **95** | **95** | **✅ 100%** |

**All tests passing.**

---

## Impact Assessment

### What Users Can Do Right Now ✅

1. **All 2D electromagnetic simulations** - Production ready
2. **All dispersive materials** - Working perfectly
3. **Complete biomedical imaging pipeline** - Fully functional
4. **3D FDTD with CPML boundaries** - Fully working and stable
5. **GPU acceleration of 2D** - Ready to implement

### No Known Blockers

All core FDTD functionality (2D and 3D) is production-ready.

---

## Next Steps

1. ✅ **Proceed to GPU acceleration** (Phase 6) — all solvers production-ready
2. 📊 MEEP validation comparison
3. Phase 5: AI integration (differentiable solver)

---

**Last verified**: 2026-05-14  
**Next review**: After GPU implementation complete
