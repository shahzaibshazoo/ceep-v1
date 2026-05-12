"""
neurowave.sources — Electromagnetic Source Implementations
=========================================================

This module provides excitation sources for FDTD simulation:

- **Gaussian**: Gaussian pulse (broadband excitation)
- **Sinusoidal**: Continuous wave (CW) source
- **Modulated**: Modulated Gaussian (band-limited)
- **Custom**: User-defined temporal waveforms
- **Plane Wave**: Total-field/scattered-field (TF/SF) sources
- **Dipole**: Point dipole sources

Source Injection
----------------
Sources can be injected as:
- Hard sources: direct field assignment (creates reflections)
- Soft sources: additive field injection (transparent)
- TF/SF: total-field/scattered-field decomposition

Gaussian Pulse
--------------
    J(t) = exp(−((t − t₀) / τ)²)

where t₀ is the time delay and τ controls the pulse width.

The frequency content spans from DC to approximately f_max = 1/(π·τ).

Sinusoidal Source
-----------------
    J(t) = sin(2πf₀t) · envelope(t)

where f₀ is the center frequency and envelope provides smooth turn-on.
"""
