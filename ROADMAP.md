# ROADMAP — NeuroWave

> GPU-Accelerated Electromagnetic Simulation Framework

---

## Phase 0 — Project Initialization ← **CURRENT**

**Status:** 🟡 In Progress  
**Target:** Foundation & environment setup

- [x] Repository structure
- [x] Python package skeleton
- [x] Build system (pyproject.toml)
- [x] Documentation framework
- [ ] Environment verification (CUDA, PyTorch)
- [ ] Git initialization
- [ ] CI/CD skeleton
- [ ] Core abstract base classes

---

## Phase 1 — Core GPU FDTD

**Status:** ⬜ Not Started  
**Target:** Working 2D FDTD engine with GPU acceleration

### Deliverables

- [ ] 2D Yee grid data structure
- [ ] TMz mode solver
- [ ] TEz mode solver
- [ ] CUDA kernel: E-field update
- [ ] CUDA kernel: H-field update
- [ ] Gaussian pulse source
- [ ] Sinusoidal (CW) source
- [ ] Simple absorbing boundary conditions (Mur ABC)
- [ ] Dielectric material support
- [ ] Field visualization (matplotlib)
- [ ] CFL stability condition enforcement
- [ ] CPU reference implementation
- [ ] CPU vs GPU benchmark
- [ ] Unit tests for all components
- [ ] Theory documentation (Maxwell equations, Yee algorithm)
- [ ] Tutorial: "Your First FDTD Simulation"

### Key Equations

```
∂E/∂t = (1/ε) ∇×H − σE/ε
∂H/∂t = −(1/μ) ∇×E
```

### CFL Condition (2D)

```
Δt ≤ 1/(c · √(1/Δx² + 1/Δy²))
```

---

## Phase 2 — Advanced FDTD

**Status:** ⬜ Not Started  
**Target:** Production-quality boundary conditions and material models

- [ ] Convolutional PML (CPML) boundaries
- [ ] Dispersive materials (Debye, Drude, Lorentz)
- [ ] Anisotropic materials
- [ ] Frequency-domain extraction (DFT monitors)
- [ ] S-parameter extraction
- [ ] Near-to-far field transformation
- [ ] Multi-source systems
- [ ] Waveguide mode sources
- [ ] Benchmark: PML reflection coefficient
- [ ] Benchmark: Dispersive material accuracy

---

## Phase 3 — 3D Engine

**Status:** ⬜ Not Started  
**Target:** Full 3D FDTD with memory optimization

- [ ] 3D Yee grid
- [ ] 3D CUDA kernels (E-field, H-field)
- [ ] Memory optimization (sparse regions)
- [ ] Domain decomposition
- [ ] Multi-GPU support (NCCL)
- [ ] Subgridding
- [ ] Benchmark: 3D cavity resonator
- [ ] Benchmark: Multi-GPU scaling

---

## Phase 4 — Biomedical Imaging

**Status:** ⬜ Not Started  
**Target:** Microwave/mmWave biomedical simulation capabilities

- [ ] Tissue dielectric library (Gabriel model)
- [ ] Multilayer skin models
- [ ] Breast tissue phantoms
- [ ] Brain hemorrhage phantoms
- [ ] Tumor detection phantoms
- [ ] Antenna array modeling
- [ ] Multistatic S-parameter extraction
- [ ] Delay-and-Sum (DAS) beamforming
- [ ] Realistic microwave imaging pipeline
- [ ] Benchmark: Tissue model validation

---

## Phase 5 — AI Integration

**Status:** ⬜ Not Started  
**Target:** Differentiable simulation and AI-driven inverse problems

- [ ] PyTorch custom autograd functions
- [ ] Differentiable Maxwell solver
- [ ] Gradient-based inverse imaging
- [ ] Neural reconstruction networks
- [ ] PGA-Net integration
- [ ] EMNeRF integration
- [ ] Physics-Informed Neural Networks (PINNs)
- [ ] AI-guided antenna optimization
- [ ] Transformer-based architectures
- [ ] Benchmark: Differentiable sim gradient accuracy

---

## Phase 6 — Advanced HPC

**Status:** ⬜ Not Started  
**Target:** Maximum performance and portability

- [ ] CUDA Graph optimization
- [ ] Tensor Core acceleration (FP16/BF16)
- [ ] Mixed precision simulation
- [ ] Distributed multi-node simulation (MPI)
- [ ] JAX backend
- [ ] Triton kernels
- [ ] Vulkan/OpenCL backend
- [ ] Benchmark: Scaling analysis

---

## Long-Term Vision

- Conference papers (IEEE AP-S, ACES, etc.)
- Journal publications
- PyPI package release
- Community contributions
- Integration with existing EM tools
- Commercial-grade reliability

---

*This roadmap is a living document updated as development progresses.*
