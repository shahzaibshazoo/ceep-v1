# Convolutional Perfectly Matched Layer (CPML)

> NeuroWave Theory Documentation

---

## 1. Introduction

The Perfectly Matched Layer (PML) is an artificial absorbing layer for wave equations. Unlike traditional boundary conditions (like Mur) which attempt to model a one-way wave equation at the boundary, the PML creates a non-physical material layer that strongly absorbs incoming waves while having an intrinsic impedance perfectly matched to the interior medium.

The Convolutional PML (CPML) formulation by Roden and Gedney is considered the "gold standard" because it:
1. Does not require field splitting (unlike Berenger's original PML).
2. Is highly efficient in memory and computation.
3. Can handle anisotropic media and dispersive materials easily.

## 2. Coordinate Stretching

The PML is mathematically derived by stretching the spatial coordinates into the complex frequency domain. A standard spatial derivative $\frac{\partial}{\partial x}$ is replaced by $\frac{1}{s_x}\frac{\partial}{\partial x}$, where $s_x$ is the complex stretching variable:

$$s_x = \kappa_x + \frac{\sigma_x}{\alpha_x + j\omega\varepsilon_0}$$

Where:
- $\kappa_x$ controls the phase velocity and is used to absorb evanescent waves (Complex Frequency Shifted or CFS-PML).
- $\sigma_x$ provides the actual wave attenuation.
- $\alpha_x$ shifts the pole away from $\omega=0$, drastically improving late-time stability and low-frequency absorption.

## 3. The CPML Formulation

When transformed back to the time domain, the $1/s_x$ term becomes a convolution. For example, Ampere's law for $H_y$:

$$\varepsilon_0 \frac{\partial E_z}{\partial t} = \frac{\partial H_y}{\partial x} \Rightarrow \varepsilon_0 \frac{\partial E_z}{\partial t} = \bar{s}_x(t) * \frac{\partial H_y}{\partial x}$$

Where $\bar{s}_x(t)$ is the inverse Fourier transform of $1/s_x$.

The CPML handles this convolution recursively. The continuous derivative is approximated as:

$$\frac{1}{s_x}\frac{\partial H_y}{\partial x} \approx \frac{1}{\kappa_x}\frac{\partial H_y}{\partial x} + \psi_{ezx}$$

Where the auxiliary variable $\psi_{ezx}$ (psi) accumulates the historical convolutional terms:

$$\psi_{ezx}^{n} = b_x \psi_{ezx}^{n-1} + c_x \left(\frac{H_y^n(i) - H_y^n(i-1)}{\Delta x}\right)$$

## 4. Polynomial Grading Profile

To prevent numerical reflections caused by sudden discretization jumps, the material properties within the PML are graded polynomially. Let $d$ be the distance into the PML and $L$ be the total thickness of the PML:

$$\rho = \frac{d}{L}$$

The parameters vary as:

$$\sigma(\rho) = \sigma_{max} \rho^m$$
$$\kappa(\rho) = 1 + (\kappa_{max} - 1) \rho^m$$
$$\alpha(\rho) = \alpha_{max} (1 - \rho)$$

Typically, $m$ (the polynomial order) is 3 or 4.

### 4.1 Optimal $\sigma_{max}$

The theoretical optimal maximum conductivity is based on the desired theoretical reflection error ($R(0)$) at normal incidence:

$$\sigma_{max} = -\frac{(m+1)\ln(R(0))}{2 \eta_0 L}$$

For NeuroWave, we implement this as a `sigma_factor` multiplier on a baseline optimal value. Based on extensive benchmarking, we found `order=4` and `sigma_factor=2.0` yields $R < -130\text{ dB}$ for a 10-cell thick PML.

## 5. Update Coefficients

The recursive convolution coefficients $b$ and $c$ are computed strictly once during initialization:

$$b = \exp\left(-\left(\frac{\sigma}{\kappa} + \alpha\right) \frac{\Delta t}{\varepsilon_0}\right)$$

$$c = \frac{\sigma}{\sigma \kappa + \kappa^2 \alpha} \left(b - 1\right)$$

In the FDTD loop, the field is updated as usual, but modified by the auxiliary variable. For instance, the $E_z$ update near the X-boundary:

$$E_z^{n+1}(i, j) = C_a E_z^{n}(i, j) + C_b \left( \frac{H_y^{n+½}(i, j) - H_y^{n+½}(i-1, j)}{\Delta x} + \psi_{ezx}^{n+½}(i) - \dots \right)$$

This effectively isolates the PML memory requirements to only the boundary regions (thickness $\times$ perimeter), rather than the entire computational domain.

## 6. References

1. J. A. Roden and S. D. Gedney, "Convolution PML (CPML): An efficient FDTD implementation of the CFS-PML for arbitrary media," *Microwave Opt. Technol. Lett.*, vol. 27, pp. 334-339, 2000.
2. A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House, 2005. (Chapter 7)

---

*Part of the NeuroWave Theory Documentation series.*
