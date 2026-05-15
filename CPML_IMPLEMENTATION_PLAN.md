# CPML Implementation Plan

## Problem
Current solver uses simple ABC (zero boundaries) which causes reflections.
Tests fail for >100 timesteps because energy bounces back.

## Solution
Implement Convolutional PML (CPML) - the same approach MEEP uses.

## MEEP's Approach (from Taflove & Hagness)

### Key Concept
CPML splits fields into:
1. **Regular update** (what we have now)
2. **PML correction** applied in boundary regions

### Mathematical Formulation

In CPML regions, field updates become:
```
H_new = H_old + dt * curl(E) + CPML_correction
E_new = E_old + dt * curl(H) + CPML_correction
```

Where CPML_correction uses auxiliary "psi" variables that accumulate history.

### Psi Variables (Memory of Past Fields)

For each PML face, we need:
- `psi_Hx` - remembers dE/dy for Hx update
- `psi_Hy` - remembers dE/dx for Hy update  
- `psi_Ez` - remembers curl(H) for Ez update

### Update Algorithm (MEEP's way)

**Step 1: Update H field (magnetic)**
```python
# Regular update (entire domain)
Hx = Hx - dt/μ * dEz/dy
Hy = Hy + dt/μ * dEz/dx

# CPML correction (only in PML regions)
for each PML layer i:
    psi_Hx[i] = b[i] * psi_Hx[i] + c[i] * dEz/dy
    Hx[PML_region] += psi_Hx[i]
    
    psi_Hy[i] = b[i] * psi_Hy[i] + c[i] * dEz/dx
    Hy[PML_region] += psi_Hy[i]
```

**Step 2: Update E field (electric)**
```python
# Regular update (entire domain)
Ez = Ez + dt/ε * (dHy/dx - dHx/dy)

# CPML correction (only in PML regions)
for each PML layer i:
    curl_H = dHy/dx - dHx/dy
    psi_Ez[i] = b[i] * psi_Ez[i] + c[i] * curl_H
    Ez[PML_region] += psi_Ez[i]
```

### Coefficients b and c

These are computed in `_setup_cpml()`:
```python
b = exp(-σ * dt / ε₀)
c = (b - 1.0)  # Simplified for basic CPML
```

Where σ varies across PML depth (polynomial grading).

## Implementation Strategy

### Option 1: Full CPML (Like MEEP)
- Implement psi updates for all 8 arrays
- Apply corrections after each field update
- Most accurate, but complex

### Option 2: Simplified CPML (Fast to implement)
- Use exponential damping in PML regions
- Simpler than full CPML
- 80-90% as effective

### Option 3: Use MEEP Directly
- Wrap MEEP's PML implementation
- But MEEP is CPU-only, we need GPU

## Recommended: Option 1 (Full CPML)

Because:
1. We already have psi arrays allocated
2. We have b and c coefficients computed
3. Just need to apply them in the update loop
4. Will match MEEP exactly

## Code Structure

```python
def _apply_cpml(self):
    """Apply CPML corrections after field updates."""
    cp = self.xp
    n = self.cpml_n
    
    # X-direction PML (left and right boundaries)
    # Left boundary (x=0 to x=n)
    for i in range(n):
        # Update psi for Hy
        dEz_dx = (self.ez[:, i+1, :] - self.ez[:, i, :]) * self.inv_dx
        self.psi_hyx_lo[:, i, :] = (
            self.cpml_b_x[i] * self.psi_hyx_lo[:, i, :]
            + self.cpml_c_x[i] * dEz_dx
        )
        self.hy[:, i, :] += self.psi_hyx_lo[:, i, :]
    
    # Right boundary (x=nx-n to x=nx)
    for i in range(n):
        x_idx = self.nx - n + i
        dEz_dx = (self.ez[:, x_idx+1, :] - self.ez[:, x_idx, :]) * self.inv_dx
        self.psi_hyx_hi[:, i, :] = (
            self.cpml_b_x[n-1-i] * self.psi_hyx_hi[:, i, :]
            + self.cpml_c_x[n-1-i] * dEz_dx
        )
        self.hy[:, x_idx, :] += self.psi_hyx_hi[:, i, :]
    
    # Similar for Y-direction and Ez field...
```

## Testing Strategy

1. **Test 1:** Verify basic CPML reduces reflections
2. **Test 2:** Run 500 steps - should NOT explode
3. **Test 3:** Compare with MEEP at 200 steps
4. **Test 4:** Brain tissue simulation at 150 steps

## Expected Results After Fix

| Test | Before | After |
|------|--------|-------|
| 100 steps | 3.367 ✓ | 3.367 ✓ |
| 120 steps | 7.5 ✗ | 3.4 ✓ |
| 150 steps | 24-681 ✗ | 3.5 ✓ |
| 500 steps | 843,022 ✗ | 3.6 ✓ |

## Implementation Time

- Full CPML: 2-3 hours
- Testing: 1 hour  
- Debugging: 1-2 hours
- **Total: ~5 hours** for production-quality CPML

## Alternative: Ask User to Wait

Since this is complex and will take hours, we could:
1. Document the 100-step limitation clearly
2. Tell users "CPML coming in v1.1"
3. For now, use 100 steps (which works perfectly)

**But user wants ALL 15 TESTS TO PASS NOW.**

So let's implement full CPML!
