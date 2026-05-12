"""
neurowave.boundaries — Boundary Condition Implementations
========================================================

This module provides absorbing and periodic boundary conditions:

- **ABC**: Mur first/second order absorbing boundary conditions
- **PML**: Perfectly Matched Layer (split-field and CPML)
- **Periodic**: Periodic boundary conditions
- **PEC/PMC**: Perfect electric/magnetic conductor

Theory
------
The Perfectly Matched Layer (PML) is the gold standard for absorbing
boundary conditions in FDTD. The Convolutional PML (CPML) formulation
is preferred for its:
- Improved absorption at grazing incidence
- Simpler implementation (no split fields)
- Better numerical stability

CPML uses auxiliary differential equations with recursive convolution
to implement frequency-dependent conductivity profiles.

References
----------
.. [1] J.-P. Berenger, "A perfectly matched layer for the absorption of
       electromagnetic waves," J. Comput. Phys., vol. 114, pp. 185-200, 1994.
.. [2] J. A. Roden and S. D. Gedney, "Convolutional PML (CPML),"
       Microwave Opt. Technol. Lett., vol. 27, pp. 334-339, 2000.
"""
