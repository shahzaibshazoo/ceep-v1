# NEXT STEPS — NeuroWave

> Last Updated: 2026-05-12

---

## Immediate (This Session / Next Session)

### 1. Environment Verification
- [ ] Check CUDA toolkit version (`nvcc --version`)
- [ ] Check GPU details (`nvidia-smi`)
- [ ] Verify PyTorch CUDA support (`torch.cuda.is_available()`)
- [ ] Check Python version
- [ ] Install package in dev mode (`pip install -e ".[dev]"`)

### 2. Core Data Structures
- [ ] Implement `Grid2D` class (Yee grid with staggered field positions)
- [ ] Implement `SimulationConfig` dataclass (grid size, resolution, timestep)
- [ ] Implement `MaterialGrid` (spatially-varying ε, μ, σ)
- [ ] Implement CFL condition calculator

### 3. CPU Reference FDTD
- [ ] Implement 2D TMz update equations (NumPy)
- [ ] Implement Gaussian pulse source
- [ ] Implement simple Mur ABC
- [ ] Create visualization routine
- [ ] Validate against analytical solutions

---

## Short-Term (Phase 1 Completion)

### 4. CUDA Kernels
- [ ] Port E-field update to CUDA (CuPy or raw CUDA)
- [ ] Port H-field update to CUDA
- [ ] Memory layout optimization (structure of arrays)
- [ ] Benchmark CPU vs GPU

### 5. Testing & Validation
- [ ] Unit tests for grid creation
- [ ] Unit tests for field updates
- [ ] Numerical stability tests
- [ ] Benchmark suite

### 6. Documentation
- [ ] Theory: Maxwell equations and Yee algorithm
- [ ] Theory: CFL condition derivation
- [ ] Tutorial: Basic 2D simulation
- [ ] Architecture: Module interaction diagram

---

## Medium-Term (Phase 2)

- [ ] Convolutional PML implementation
- [ ] Debye dispersive materials
- [ ] DFT frequency monitors
- [ ] S-parameter extraction

---

## Research Opportunities

1. **Novel GPU FDTD kernel designs** — memory coalescing strategies
2. **Mixed-precision FDTD** — FP16 for field storage with FP32 accumulation
3. **Differentiable FDTD** — adjoint method vs direct autograd
4. **AI-accelerated PML** — learned boundary conditions
5. **Connection to user's existing work** — PGA-Net and EMNeRF integration

---

*This file is a prioritized queue of upcoming work.*
