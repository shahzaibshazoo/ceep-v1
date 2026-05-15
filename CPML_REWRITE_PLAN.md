# CPML Rewrite Plan

## Current Problem

The CPML implementation is fundamentally flawed:
1. Applies CPML as "corrections" after standard update
2. Unclear coefficient definitions
3. Has failed 3 times now (original, fix #1, fix #2)

## Root Cause

**CPML is not a correction - it's a modified update equation!**

The standard FDTD update in PML regions must be **replaced**, not corrected.

## Standard FDTD (no PML)

```
H_x^{n+1/2} = H_x^{n-1/2} - (dt/μ) * dE_z/dy
H_y^{n+1/2} = H_y^{n-1/2} + (dt/μ) * dE_z/dx
E_z^{n+1}   = E_z^n + (dt/ε) * (dH_y/dx - dH_x/dy)
```

## CPML FDTD (in PML regions)

Instead of using derivatives directly, we use **convolution integrals** accumulated in psi arrays:

```
# Update psi memory variables:
ψ_Hx_y^{n+1/2} = b_y * ψ_Hx_y^{n-1/2} + c_y * dE_z/dy
ψ_Hy_x^{n+1/2} = b_x * ψ_Hy_x^{n-1/2} + c_x * dE_z/dx

# H-field update IN PML:
H_x^{n+1/2} = H_x^{n-1/2} - (dt/μ) * ψ_Hx_y^{n+1/2}
H_y^{n+1/2} = H_y^{n-1/2} + (dt/μ) * ψ_Hy_x^{n+1/2}

# Similar for E-field with its own psi arrays
```

## Coefficients (Roden & Gedney, 2000)

```
σ(x) = σ_max * ((n - x) / n)^m   # Polynomial grading, m=3 typical
κ(x) = 1 + (κ_max - 1) * ((n - x) / n)^m   # Stretching, κ_max=1-15
α(x) = α_max * (x / n)^m_α       # CFS, α_max ≈ 0.05-0.24

b = exp(-(σ/κ + α) * dt / ε₀)
c = (b - 1) / (σ/κ + κ*α) * σ/κ   # When α=0: c = σ/κ * (b-1) / (σ/κ) = b-1
```

For simplified CPML (α=0, κ=1):
```
b = exp(-σ * dt / ε₀)
c = b - 1
```

## Key Insight

When α=0, κ=1:  **c = b - 1**

Since 0 < b < 1, we have **-1 < c < 0**.

This negative c is CORRECT! It provides damping because:
```
ψ^{n+1} = b * ψ^n + c * derivative
        = b * ψ^n + (b-1) * derivative
        = b * (ψ^n + derivative) - derivative
```

The (b-1) term creates exponential decay of the accumulated integral.

## What's Wrong in Current Code?

Looking at line 256:
```python
self.hy[:, i, :] += self.db[0, i, :] * self.psi_hxy_lo[:, i, :]
```

This **ADDS** psi to the already-updated field. But the update should be:
```python
self.hy[:, i, :] = self.da * self.hy_old - self.db * psi
```

NOT:
```python
self.hy[:, i, :] = self.da * self.hy_old - self.db * normal_derivative
self.hy[:, i, :] += self.db * psi  # WRONG!
```

## The Fix

We need to:
1. Store old field values before update
2. In PML regions, use psi instead of direct derivatives
3. Not "add corrections" but use psi AS the derivative term

OR simpler:
1. Compute derivatives everywhere as normal
2. In PML: psi = b*psi + c*derivative, then replace derivative with psi
3. Continue with standard update using psi

## Implementation Strategy

### Option A: Minimal Change (safest)
Keep current structure but fix the application:
```python
# Current:
self.hy[:, i, :] += self.db[0, i, :] * self.psi_hxy_lo[:, i, :]

# Fixed:
# The psi already contains the time-integrated derivative
# We need to SUBTRACT the normal derivative contribution first
normal_deriv = (self.ez[:, i+1, :] - self.ez[:, i, :]) * self.inv_dx
self.hy[:, i, :] -= self.db[0, i, :] * normal_deriv  # Remove normal contrib
self.hy[:, i, :] += self.db[0, i, :] * self.psi_hxy_lo[:, i, :]  # Add CPML
```

### Option B: Rewrite (cleanest)
Separate PML and non-PML regions completely:
```python
# Non-PML regions: standard update
self.hy[:, n:nx-n, :] = ...  # standard FDTD

# PML regions: CPML update
for i in range(n):
    # Update psi
    deriv = (self.ez[:, i+1, :] - self.ez[:, i, :]) * self.inv_dx
    self.psi[:, i, :] = b[i] * self.psi[:, i, :] + c[i] * deriv
    
    # Use psi for update
    self.hy[:, i, :] = da * self.hy[:, i, :] + db * self.psi[:, i, :]
```

### Option C: Reference Implementation
Copy MEEP's CPML exactly, line-by-line.

## Recommendation

Start with **Option A** - it's a 2-line fix to test if the concept is right.
If that works, refactor to Option B for clarity.
If both fail, we need Option C and careful verification against MEEP source.

## Testing

After any change:
1. Run TEST_CPML_FIX.py - should show < 2x growth
2. Run RUN_ALL_TESTS.py - all 15 should pass
3. Re-run CEEP vs MEEP validation - error should be < 10%
4. If all pass: CPML is fixed, proceed to dataset generation

If any fail: Debug further, possibly need Option C (MEEP source study).
