# Correct CPML Implementation

## The Problem

The current implementation has fundamental design flaws:
1. Mixing material damping (da, db coefficients) with CPML psi updates
2. Incomplete E-field CPML boundaries
3. Wrong indexing for PML/interior boundary
4. Not handling corners (both X and Y PML)

## Root Cause of Instability

When fused kernels were disabled, the buggy CPML code started running, causing:
- 100 steps: 10^158 (catastrophic growth)
- 500 steps: NaN (complete numerical overflow)

This is because the update equations are fundamentally wrong.

## Correct Approach

CPML in 2D TM mode requires:
1. Replace derivatives with psi in PML regions ONLY
2. Use simple H = H + dt/mu * psi (no da damping in PML)
3. Apply material losses in psi update, not field update
4. Handle all 4 boundaries + 4 corners correctly

## Recommended Action

Given the complexity and multiple failed attempts, the best path forward is:

### Option 1: Use MEEP for dataset generation
- MEEP works correctly (validated)
- It's 1400x slower but produces correct results
- For a dataset of 10,000 samples, would take ~40 hours on CPU
- This is acceptable for a one-time dataset generation

### Option 2: Hire a computational electromagnetics expert
- CPML implementation requires deep FDTD expertise
- An expert could fix this in 1-2 days
- Cost: ~$500-1000 for consulting

### Option 3: Use simpler ABC boundaries
- Give up on CPML, use first-order Mur ABC
- Will have ~5-10% reflections but might be acceptable
- Much simpler to implement correctly

### Option 4: Fork a working FDTD-CPML codebase
- Find an open-source Python FDTD with working CPML
- Adapt it to your batched GPU architecture
- Examples: gprMax, meep-python internals

## Apology

I apologize for the multiple failed attempts. CPML is more complex than I initially assessed, with many subtle interactions between:
- Coordinate systems (Yee grid staggering)
- Update equation ordering
- Material vs PML damping
- Boundary indexing
- Corner regions

Each "fix" created new bugs because the implementation needs to be correct as a whole, not incrementally patched.

For production use, I recommend Option 1 (MEEP) or Option 2 (expert) rather than continuing to debug this implementation.
