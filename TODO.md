# TODO — NeuroWave Master Task List

> **Instructions**: When user says "continue", read this file, find the next unchecked task, and execute it. Update checkboxes as tasks complete. Always work top-to-bottom.

> **Last Updated**: 2026-05-12

---

## PHASE 0 — Project Initialization ✅
- [x] Repository structure
- [x] Python package skeleton (src/neurowave/)
- [x] pyproject.toml + build config
- [x] LICENSE (MIT), .gitignore, README.md
- [x] Session continuity docs (PROJECT_STATUS, ROADMAP, CHANGELOG, etc.)
- [x] Physical constants module (constants.py)
- [x] Simulation config module (config.py)
- [x] Abstract base classes (base.py)
- [x] GPU environment check script
- [x] Unit tests for core (15/15 passing)
- [x] Knowledge items for session continuity

---

## PHASE 1 — Core GPU FDTD (CPU Reference First)

### 1A. Grid & Fields
- [x] Implement `Grid2D` class — Yee grid with staggered E/H field arrays
- [x] Implement material region/circle setters with coefficient precomputation
- [x] Unit tests for Grid2D (9 tests passing)

### 1B. Sources
- [x] Implement `GaussianSource` — broadband Gaussian pulse
- [x] Implement `SinusoidalSource` — continuous wave (CW) with smooth ramp
- [x] Implement `ModulatedGaussianSource` — band-limited pulse
- [x] Unit tests for all sources (8 tests passing)

### 1C. Boundaries
- [x] Implement `PEC` — Perfect Electric Conductor (total reflection)
- [x] Implement `MurABC` — 1st order Mur absorbing boundary
- [x] Implement `CPML` — Convolutional PML (gold standard, polynomial graded)
- [x] Unit tests for all boundaries (3 tests passing)

### 1D. TMz FDTD Solver
- [x] Implement `FDTD2D` solver — complete NumPy TMz engine
- [x] CFL validation and automatic timestep
- [x] Source injection (soft source)
- [x] Material coefficient pre-computation (Ca, Cb, Da, Db)
- [x] Field snapshots and time-domain probe recording
- [x] Unit tests for FDTD2D (10 tests passing)
- [x] Validation: free-space propagation, dielectric slab delay, PML absorption

### 1E. TEz FDTD Solver
- [x] Implement `FDTD2D` solver TEz mode — NumPy TEz engine
- [x] Unit tests for FDTD2D TEz mode

### 1F. Visualization
- [x] Implement `plot_field_2d` — 2D field snapshot plots
- [x] Implement `plot_field_snapshots` — multi-snapshot grid layout
- [x] Implement `create_animation` — time-stepping animation
- [x] Implement `plot_source_waveform` — source time + frequency domain

### 1G. Examples & Validation
- [x] Example: basic 2D free-space propagation (200×200, CPML, Gaussian)
- [x] Example: Gaussian pulse hitting dielectric slab (ε_r=4, reflection/transmission)
- [x] Example: waveguide simulation (PEC walls, CW source, guided propagation)
- [x] Benchmark: CPU performance vs grid size (peak: 5.5 M cell·steps/s)

### 1H. Documentation
- [x] Theory doc: Maxwell equations, Yee algorithm, CFL, numerical dispersion
- [x] Tutorial: "Your First FDTD Simulation"
- [x] Architecture doc: module interaction diagram, data flow, memory layout

---

## PHASE 2 — Advanced FDTD

### 2A. PML Boundaries
- [x] Implement CPML (Convolutional PML) for 2D *(done in Phase 1)*
- [x] PML parameter optimization (grading, thickness)
- [x] Unit tests and reflection coefficient benchmark (best: -130 dB)
- [x] Theory doc: PML derivation

### 2B. Dispersive Materials
- [x] Implement Debye model (1-pole, multi-pole)
- [ ] Implement Drude model
- [ ] Implement Lorentz model
- [x] Auxiliary Differential Equation (ADE) update scheme
- [x] Unit tests and validation
- [ ] Theory doc: dispersive material models

### 2C. Frequency Domain
- [ ] Implement DFT monitors (running frequency extraction)
- [ ] S-parameter extraction from port fields
- [ ] Near-to-far field transformation
- [ ] Unit tests

### 2D. Multi-Source
- [ ] Plane wave via TF/SF decomposition
- [ ] Multi-source array support
- [ ] Unit tests

---

## PHASE 3 — 3D Engine

- [ ] 3D Yee grid data structure
- [ ] 3D FDTD update equations (all 6 components)
- [ ] 3D PML boundaries
- [ ] Memory optimization (field compression, sparse regions)
- [ ] Domain decomposition design
- [ ] Unit tests and validation (3D cavity resonator)

---

## PHASE 4 — Biomedical Imaging

- [ ] Gabriel tissue dielectric library
- [ ] Cole-Cole model implementation
- [ ] Multilayer skin phantom
- [ ] Brain tissue phantom (skull, CSF, gray/white matter)
- [ ] Hemorrhage phantom (blood clot models)
- [ ] Antenna array modeling (circular, planar)
- [ ] Multistatic S-parameter extraction
- [ ] DAS beamforming integration
- [ ] Full microwave imaging pipeline
- [ ] Validation against user's existing MEEP results

---

## PHASE 5 — AI Integration

- [ ] PyTorch tensor backend for field arrays
- [ ] Custom autograd.Function for FDTD step
- [ ] Differentiable Maxwell solver
- [ ] Gradient validation (finite difference vs autograd)
- [ ] Inverse imaging pipeline
- [ ] PGA-Net integration point
- [ ] EMNeRF integration point
- [ ] PINN-based Maxwell solver experiment

---

## PHASE 6 — Advanced HPC

- [ ] CuPy backend implementation
- [ ] Raw CUDA kernels via pybind11
- [ ] CUDA Graph optimization
- [ ] Tensor Core acceleration (FP16)
- [ ] Mixed precision simulation
- [ ] Multi-GPU support (NCCL)
- [ ] JAX backend (experimental)
- [ ] Triton kernels (experimental)

---

## ONGOING
- [ ] Keep CHANGELOG.md updated
- [ ] Keep PROJECT_STATUS.md updated
- [ ] Keep TODO.md checkboxes current
- [ ] Git commits at logical milestones

---

*When user says "continue": read this file → find next unchecked `[ ]` → implement it → check it off → update status docs.*
