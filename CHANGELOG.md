# CHANGELOG

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] — 2026-05-14

First public release. Complete FDTD electromagnetic simulation framework with GPU acceleration.

### Features

- **2D FDTD Solver** — TMz and TEz modes with vectorized NumPy/CuPy updates
- **3D FDTD Solver** — Full Yee grid with 6 field components
- **Batched GPU Solver** — Run N simulations simultaneously for antenna arrays
- **CPML Boundaries** — Convolutional PML with polynomial grading (3rd order)
- **Dispersive Materials** — Debye, Drude, Lorentz, Cole-Cole via ADE method
- **Tissue Database** — Gabriel et al. 4-term Cole-Cole for 50+ biological tissues
- **Antenna Arrays** — Circular, planar, ULA, URA, L-shaped, random, conformal
- **Head Phantoms** — Multilayer models with hemorrhage simulation
- **DAS Beamforming** — Image reconstruction from multistatic S-parameters
- **Fused CUDA Kernels** — RawKernel implementations for 2D, 3D, and batched solvers
- **Backend Abstraction** — Seamless NumPy/CuPy switching with `set_backend()`

### Performance

- Batched 16-antenna simulation: 4.3x GPU speedup on T4
- Single 1000x1000 simulation: 2.1x GPU speedup
- Fused kernels eliminate intermediate allocations

### Validated

- 36 unit tests passing
- CPU-GPU numerical agreement to machine precision
- Wave propagation verified against analytical solutions

---

## [0.1.0-dev] — 2026-05-12

Initial project scaffold and core engine.
