# CPML Reference Implementation
## Based on Roden & Gedney (2000) - IEEE TAP

## Standard CPML Formulation

### Coordinate Stretching in Frequency Domain

CPML uses complex coordinate stretching in the frequency domain:
```
s(x) = κ(x) + σ(x)/(α(x) + jωε₀)
```

Where:
- κ(x) = stretching factor (≥1)
- σ(x) = conductivity profile
- α(x) = complex frequency shift (CFS)

### Time Domain Convolution

The frequency-domain division by `jω` becomes a time-domain integral (convolution).
CPML uses auxiliary differential equations (ADE) to avoid storing history.

### Psi Variables (Memory of Convolution)

For each field component affected by PML, we need a psi variable:
```
∂ψ/∂t + (σ/κ + α) ψ = σ/(κ·ε₀) · ∂F/∂x
```

Where F is the field derivative being absorbed.

### Discretized Update (Leap-Frog)

```
ψⁿ⁺¹ = bₑ·ψⁿ + cₑ·(∂F/∂x)ⁿ⁺¹/²

Where:
bₑ = exp(-(σ/κ + α)·Δt/ε₀)
cₑ = σ/(σ·κ + κ²·α) · (bₑ - 1)
```

### Simplified CPML (α=0, κ=1)

For basic absorption (no CFS):
```
bₑ = exp(-σ·Δt/ε₀)
cₑ = (bₑ - 1) · σ/σ = bₑ - 1
```

**Key insight**: cₑ is NEGATIVE (since 0 < bₑ < 1)

## Field Update in PML

### Standard FDTD (no PML):
```
Hₓⁿ⁺¹/² = Hₓⁿ⁻¹/² - (Δt/μ)·∂Eᵧ/∂z
```

### With CPML:
```
# Update psi
ψ_Hx^{n+1/2} = bₑ·ψ_Hx^{n-1/2} + cₑ·(∂Eᵧ/∂z)ⁿ

# Use psi instead of derivative
Hₓⁿ⁺¹/² = Hₓⁿ⁻¹/² - (Δt/μ)·κₓ·ψ_Hx^{n+1/2}
```

The key is that **ψ replaces the derivative**, not added as correction!

## 2D TM Mode (Ez, Hx, Hy)

### H-field update in PML:

**Standard:**
```
Hₓ = Hₓ - (Δt/μ)·∂Ez/∂y
Hᵧ = Hᵧ + (Δt/μ)·∂Ez/∂x
```

**CPML (Y-direction PML affects Hx):**
```
∂ψ_Hx/∂t = (σᵧ/ε₀)·∂Ez/∂y - (σᵧ/κᵧ + αᵧ)·ψ_Hx

Discrete:
ψ_Hx^{n+1/2} = b_y·ψ_Hx^{n-1/2} + c_y·(∂Ez/∂y)ⁿ
Hₓⁿ⁺¹/² = Hₓⁿ⁻¹/² - (Δt/μ)·ψ_Hx^{n+1/2}
```

**CPML (X-direction PML affects Hy):**
```
ψ_Hy^{n+1/2} = b_x·ψ_Hy^{n-1/2} + c_x·(∂Ez/∂x)ⁿ
Hᵧⁿ⁺¹/² = Hᵧⁿ⁻¹/² + (Δt/μ)·ψ_Hy^{n+1/2}
```

### E-field update in PML:

**Standard:**
```
Ez = Ez + (Δt/ε)·(∂Hy/∂x - ∂Hx/∂y)
```

**CPML (X-direction PML):**
```
ψ_Ez_x^{n+1} = b_x·ψ_Ez_x^n + c_x·(∂Hy/∂x)^{n+1/2}
```

**CPML (Y-direction PML):**
```
ψ_Ez_y^{n+1} = b_y·ψ_Ez_y^n + c_y·(∂Hx/∂y)^{n+1/2}
```

**Full E update in PML:**
```
Ez^{n+1} = Ez^n + (Δt/ε)·(ψ_Ez_x^{n+1} - ψ_Ez_y^{n+1})
```

## Coefficient Calculation

### σ Profile (Polynomial Grading)
```python
m = 3  # grading order
σ_max = 0.8 · (m+1) / (dx · √μ/ε)

For depth d from 0 (at boundary) to n (interior):
σ(d) = σ_max · ((n-1-d)/(n-1))^m
```

### b and c Profiles
```python
b[d] = exp(-σ[d] · dt / ε₀)
c[d] = (b[d] - 1.0)  # For simplified CPML (α=0, κ=1)
```

### Important Notes
1. σ is largest at boundary (d=0), smallest at interior (d=n-1)
2. b approaches 0 at boundary (strong damping), approaches 1 at interior
3. c is most negative at boundary, approaches 0 at interior
4. The negative sign in c provides the damping behavior

## Implementation Checklist

✓ Psi arrays initialized to zero
✓ σ profile computed with polynomial grading
✓ b = exp(-σ·dt/ε₀)
✓ c = b - 1 (will be negative)
✓ In PML regions: compute derivative
✓ Update psi: ψ^new = b·ψ^old + c·derivative
✓ Replace standard derivative with psi in field update
✗ Do NOT add psi as correction on top of standard update

## Common Mistakes

1. **Adding correction instead of replacing**
   ```python
   # WRONG:
   H = H - dt/mu * deriv  # standard update
   H = H + dt/mu * psi    # add correction ❌
   
   # RIGHT:
   psi = b*psi + c*deriv
   H = H - dt/mu * psi    # use psi as derivative ✓
   ```

2. **Wrong coefficient formula**
   ```python
   # WRONG:
   c = sigma / (something complex)  # ❌
   
   # RIGHT (α=0, κ=1):
   c = b - 1  # Simple! ✓
   ```

3. **Incorrect grading direction**
   ```python
   # WRONG:
   sigma[i] = sigma_max * (i/n)^m  # grows inward ❌
   
   # RIGHT:
   sigma[i] = sigma_max * ((n-1-i)/(n-1))^m  # shrinks inward ✓
   ```

4. **Not handling both X and Y PML**
   - Each direction needs its own psi variables
   - Corner regions need BOTH X and Y psi contributions

## Testing

A working CPML should show:
- S-parameter grows to steady state, then stabilizes
- No exponential growth over time
- 500-step simulation ≈ same magnitude as 100-step
- Accurate results vs reference (MEEP) within 10%
