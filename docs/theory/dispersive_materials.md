# Dispersive Material Models in FDTD

In the time-domain, frequency-dependent relative permittivity $\epsilon_r(\omega)$ cannot be implemented as a simple multiplication because the time-domain equivalent of a frequency-domain product is a convolution. 

The **Auxiliary Differential Equation (ADE)** method is used in NeuroWave to bypass the expensive convolution integral. Instead of convolving the electric field history, the ADE method relates the electric polarization $\mathbf{P}$ to the electric field $\mathbf{E}$ via an Ordinary Differential Equation (ODE) in time.

## General Polarization Formulation

Ampère's law with polarization current is:
$$ \nabla \times \mathbf{H} = \epsilon_0 \epsilon_{\infty} \frac{\partial \mathbf{E}}{\partial t} + \sigma \mathbf{E} + \mathbf{J}_p $$
where the polarization current is $\mathbf{J}_p = \frac{\partial \mathbf{P}}{\partial t}$.

For a multi-pole dispersive material, $\mathbf{J}_p = \sum_{m} \mathbf{J}_{p,m}$. We update each pole $m$ individually.

---

## 1. Debye Model

The Debye model is used extensively for modeling biological tissues and water at microwave frequencies.

**Frequency Domain:**
$$ \epsilon_r(\omega) = \epsilon_{\infty} + \frac{\Delta\epsilon}{1 + j\omega \tau} $$

**Time-Domain ADE (1st Order):**
$$ \tau \frac{\partial \mathbf{J}_p}{\partial t} + \mathbf{J}_p = \epsilon_0 \Delta\epsilon \frac{\partial \mathbf{E}}{\partial t} $$

**FDTD Discretization (Central Difference):**
Using a semi-implicit average for $\mathbf{J}_p(t)$ at $n+1/2$:
$$ \mathbf{J}_p^{n+1} = k_1 \mathbf{J}_p^n + k_2 \frac{\mathbf{E}^{n+1} - \mathbf{E}^n}{\Delta t} $$
where:
- $k_1 = \frac{2\tau - \Delta t}{2\tau + \Delta t}$
- $k_2 = \frac{2 \epsilon_0 \Delta\epsilon \Delta t}{2\tau + \Delta t}$

---

## 2. Drude Model

The Drude model describes free-electron gas behaviors, commonly used for metals (e.g., gold, silver) and cold plasmas.

**Frequency Domain:**
$$ \epsilon_r(\omega) = \epsilon_{\infty} - \frac{\omega_p^2}{\omega^2 - j\omega \gamma} $$
where $\omega_p$ is the plasma frequency and $\gamma$ is the collision frequency.

**Time-Domain ADE (1st Order in J):**
$$ \frac{\partial \mathbf{J}_p}{\partial t} + \gamma \mathbf{J}_p = \epsilon_0 \omega_p^2 \mathbf{E} $$

**FDTD Discretization:**
$$ \mathbf{J}_p^{n+1} = k_1 \mathbf{J}_p^n + k_2 \frac{\mathbf{E}^{n+1} + \mathbf{E}^n}{2} $$
where:
- $k_1 = \frac{2 - \gamma \Delta t}{2 + \gamma \Delta t}$
- $k_2 = \frac{2 \epsilon_0 \omega_p^2 \Delta t}{2 + \gamma \Delta t}$

*(Note: NeuroWave implements a modified unified form coupling the Drude current to the explicit field updates).*

---

## 3. Lorentz Model

The Lorentz model describes resonant bound-electron behaviors, used for optical resonances and metamaterials.

**Frequency Domain:**
$$ \epsilon_r(\omega) = \epsilon_{\infty} + \frac{\Delta\epsilon \, \omega_0^2}{\omega_0^2 + 2j\omega\delta - \omega^2} $$
where $\omega_0$ is the resonant frequency and $\delta$ is the damping coefficient.

**Time-Domain ADE (2nd Order):**
$$ \frac{\partial^2 \mathbf{P}}{\partial t^2} + 2\delta \frac{\partial \mathbf{P}}{\partial t} + \omega_0^2 \mathbf{P} = \epsilon_0 \Delta\epsilon \omega_0^2 \mathbf{E} $$

Taking the derivative to formulate in terms of $\mathbf{J}_p$:
$$ \frac{\partial^2 \mathbf{J}_p}{\partial t^2} + 2\delta \frac{\partial \mathbf{J}_p}{\partial t} + \omega_0^2 \mathbf{J}_p = \epsilon_0 \Delta\epsilon \omega_0^2 \frac{\partial \mathbf{E}}{\partial t} $$

**FDTD Discretization (2nd Order):**
This requires three time levels ($n+1$, $n$, $n-1$) for $\mathbf{J}_p$:
$$ \mathbf{J}_p^{n+1} = k_1 \mathbf{J}_p^n + k_3 \mathbf{J}_p^{n-1} + k_2 \frac{\mathbf{E}^{n+1} - \mathbf{E}^{n-1}}{2\Delta t} $$
where:
- Denominator $D = 4 + 2\delta \Delta t + \omega_0^2 \Delta t^2$
- $k_1 = \frac{8 - 2\omega_0^2 \Delta t^2}{D}$
- $k_3 = -\frac{4 - 2\delta \Delta t + \omega_0^2 \Delta t^2}{D}$
- $k_2 = \frac{4 \epsilon_0 \Delta\epsilon \omega_0^2 \Delta t}{D}$

## E-field Update Integration
To maintain a completely explicit update without needing implicit matrix inversions, $\mathbf{J}_p^{n+1/2}$ is expressed in terms of $\mathbf{E}^{n+1}$ algebraically. The $\mathbf{E}^{n+1}$ terms are collected and added to the effective permittivity, updating the standard $C_b$ coefficients across the grid seamlessly.
