# Maxwell's Equations and the FDTD Method

> NeuroWave Theory Documentation

---

## 1. Maxwell's Equations (Differential Form)

Maxwell's equations in a linear, isotropic medium with electric conductivity σ:

$$\nabla \times \mathbf{E} = -\mu \frac{\partial \mathbf{H}}{\partial t}$$

$$\nabla \times \mathbf{H} = \varepsilon \frac{\partial \mathbf{E}}{\partial t} + \sigma \mathbf{E}$$

$$\nabla \cdot (\varepsilon \mathbf{E}) = \rho_v$$

$$\nabla \cdot (\mu \mathbf{H}) = 0$$

Where:
- **E** = electric field intensity [V/m]
- **H** = magnetic field intensity [A/m]
- ε = permittivity [F/m] = ε₀·εᵣ
- μ = permeability [H/m] = μ₀·μᵣ
- σ = electric conductivity [S/m]

## 2. TMz Mode (2D)

For 2D TMz polarization (fields independent of z), only three components remain:

**Ez**, **Hx**, **Hy**

The curl equations reduce to:

$$\mu \frac{\partial H_x}{\partial t} = -\frac{\partial E_z}{\partial y}$$

$$\mu \frac{\partial H_y}{\partial t} = \frac{\partial E_z}{\partial x}$$

$$\varepsilon \frac{\partial E_z}{\partial t} = \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y} - \sigma E_z$$

## 3. The Yee Algorithm

### 3.1 Staggered Grid

The Yee algorithm places field components at staggered spatial positions:

```
    (i,j+1) ---- Hy(i,j+1) ---- (i+1,j+1)
       |                            |
       |                            |
    Hx(i,j)      Ez(i,j)        Hx(i+1,j)
       |                            |
       |                            |
    (i,j) ------ Hy(i,j) ------- (i+1,j)
```

- **Ez** at integer points: (i·Δx, j·Δy)
- **Hx** at half-y: (i·Δx, (j+½)·Δy)
- **Hy** at half-x: ((i+½)·Δx, j·Δy)

### 3.2 Leapfrog Time-Stepping

E and H are staggered in time by half a timestep:

- **H** is updated at t = (n+½)·Δt
- **E** is updated at t = (n+1)·Δt

This gives second-order temporal accuracy.

### 3.3 Update Equations

**H-field update** (from time n-½ to n+½):

```
Hx^{n+½}(i,j) = Da(i,j)·Hx^{n-½}(i,j) - Db(i,j)/Δy · [Ez^n(i,j+1) - Ez^n(i,j)]

Hy^{n+½}(i,j) = Da(i,j)·Hy^{n-½}(i,j) + Db(i,j)/Δx · [Ez^n(i+1,j) - Ez^n(i,j)]
```

**E-field update** (from time n to n+1):

```
Ez^{n+1}(i,j) = Ca(i,j)·Ez^n(i,j) + Cb(i,j) · [
    (Hy^{n+½}(i,j) - Hy^{n+½}(i-1,j)) / Δx
  - (Hx^{n+½}(i,j) - Hx^{n+½}(i,j-1)) / Δy
]
```

### 3.4 Update Coefficients

For lossy dielectric media:

```
Ca = (1 - σ·Δt/(2ε)) / (1 + σ·Δt/(2ε))
Cb = (Δt/ε) / (1 + σ·Δt/(2ε))

Da = (1 - σ_m·Δt/(2μ)) / (1 + σ_m·Δt/(2μ))
Db = (Δt/μ) / (1 + σ_m·Δt/(2μ))
```

For lossless free space (σ=0, σ_m=0): Ca=Da=1, Cb=Δt/ε₀, Db=Δt/μ₀.

## 4. CFL Stability Condition

The FDTD method is **conditionally stable**. The timestep must satisfy the Courant-Friedrichs-Lewy (CFL) condition:

**2D:**

$$\Delta t \leq \frac{1}{c_0 \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}}}$$

**3D:**

$$\Delta t \leq \frac{1}{c_0 \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

For uniform grid (Δx = Δy = Δ):
- 2D: Δt ≤ Δ/(c₀·√2)
- 3D: Δt ≤ Δ/(c₀·√3)

The **Courant number** S = c₀·Δt·√(1/Δx² + 1/Δy²) must satisfy S ≤ 1 for stability.

### 4.1 Practical Choice

We typically use S = 0.5 (well below the stability limit) for:
- Safety margin against roundoff errors
- Reduced numerical dispersion
- Compatibility with dispersive materials

## 5. Numerical Dispersion

The FDTD method introduces artificial dispersion — the numerical phase velocity differs from the physical phase velocity:

$$\frac{v_{num}}{c} = \frac{1}{S} \cdot \frac{\arcsin\left(S\sqrt{\sin^2\left(\frac{k_x \Delta x}{2}\right)/\Delta x^2 + \sin^2\left(\frac{k_y \Delta y}{2}\right)/\Delta y^2} \cdot \frac{1}{\sqrt{1/\Delta x^2 + 1/\Delta y^2}}\right)}{\sqrt{\sin^2(k_x \Delta x/2)/\Delta x^2 + \sin^2(k_y \Delta y/2)/\Delta y^2} / \sqrt{1/\Delta x^2 + 1/\Delta y^2}}$$

**Rule of thumb**: Use Δx ≤ λ_min/10 (10 points per wavelength minimum).
For high-accuracy work: Δx ≤ λ_min/20.

## 6. Spatial Resolution Guidelines

| Application | Points/wavelength | Notes |
|------------|-------------------|-------|
| Basic propagation | 10 | Minimum for reasonable accuracy |
| Dielectric interfaces | 15-20 | Higher contrast needs finer grid |
| Dispersive materials | 20+ | Frequency-dependent resolution |
| Biomedical tissue | 20+ | High-contrast, lossy media |
| Near-field coupling | 15+ | Evanescent fields need resolution |

## 7. References

1. K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.*, vol. 14, no. 3, pp. 302-307, 1966.
2. A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House, 2005.
3. J.-P. Berenger, "A perfectly matched layer for the absorption of electromagnetic waves," *J. Comput. Phys.*, vol. 114, pp. 185-200, 1994.

---

*Part of the NeuroWave Theory Documentation series.*
