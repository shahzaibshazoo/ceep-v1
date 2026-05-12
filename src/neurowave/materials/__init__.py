"""
neurowave.materials — Dielectric Material Models
================================================

This module provides material definitions for electromagnetic simulation:

- **Simple**: Constant ε, μ, σ materials
- **Dispersive**: Frequency-dependent models (Debye, Drude, Lorentz)
- **Anisotropic**: Tensor permittivity/permeability
- **Tissue**: Biomedical tissue dielectric library (Gabriel model)
- **Database**: Pre-built material database

Physical Models
---------------
Debye model:
    ε(ω) = ε∞ + (εs - ε∞) / (1 + jωτ)

Drude model:
    ε(ω) = ε∞ - ωp² / (ω² + jωγ)

Cole-Cole model (biological tissues):
    ε(ω) = ε∞ + Σ (Δεn / (1 + (jωτn)^(1-αn))) + σs/(jωε₀)

References
----------
.. [1] S. Gabriel, R. W. Lau, C. Gabriel, "The dielectric properties of
       biological tissues," Phys. Med. Biol., vol. 41, pp. 2231-2293, 1996.
"""
