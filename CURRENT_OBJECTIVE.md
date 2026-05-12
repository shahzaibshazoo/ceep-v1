# CURRENT OBJECTIVE — NeuroWave

> Last Updated: 2026-05-12

---

## Objective: Phase 0 — Project Initialization & Environment Setup

### Goal

Establish the complete project foundation so that Phase 1 (Core GPU FDTD) development can begin immediately in the next session.

### Specific Tasks

1. **Repository scaffold** ✅
   - Directory structure matching specification
   - Python package layout (src layout)
   - Build configuration

2. **Environment verification** 🟡
   - Verify CUDA toolkit version and GPU availability
   - Verify PyTorch installation with CUDA support
   - Verify NumPy, CuPy, Numba availability
   - Document system capabilities

3. **Core abstractions design** ⬜
   - Define `Grid` base class (2D/3D Yee grids)
   - Define `Field` data structure (E, H component arrays)
   - Define `Material` interface
   - Define `Source` interface
   - Define `Boundary` interface
   - Define `Solver` interface

4. **Git initialization** ⬜
   - Initialize repository
   - Create initial commit
   - Set up branch strategy

### Success Criteria

- [ ] All directories exist with placeholder files
- [ ] `pip install -e .` works without errors
- [ ] CUDA availability confirmed
- [ ] Core interfaces documented
- [ ] Project is in a resumable state

### Priority

**HIGH** — This blocks all subsequent development.

---

*This file tracks the immediate, active objective.*
