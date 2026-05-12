# CHANGELOG — NeuroWave

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Core FDTD Engine (Phase 1):**
  - `neurowave.core.grid`: `Grid2D` class implementing the standard 2D Yee staggered grid with automatic material coefficient precomputation.
  - `neurowave.sources`: Soft source injection including `GaussianSource`, `SinusoidalSource`, and `ModulatedGaussianSource`.
  - `neurowave.boundaries`: Absorbing and reflecting boundaries including `PEC`, `MurABC`, and a highly effective `CPML` (Convolutional Perfectly Matched Layer) with polynomial grading.
  - `neurowave.solvers`: `FDTD2D` solver with fully vectorized NumPy updates for both TMz and TEz polarizations.
  - `neurowave.visualization`: Matplotlib-based plotting suite (`plot_field_2d`, `plot_field_snapshots`, `create_animation`, `plot_source_waveform`).
- **Tests & Validation:**
  - 47 comprehensive Pytest unit tests achieving 100% pass rate.
  - Examples: Basic 2D free-space propagation, Dielectric slab reflection/transmission, Parallel-plate waveguide.
  - Benchmarks: CPU scaling benchmark establishing a baseline throughput of ~5.5M cell-steps/s.
- **Documentation:**
  - Theory documentation (`maxwell_fdtd.md`) covering Maxwell's equations, Yee algorithm, CFL, and dispersion.
  - Architecture overview (`overview.md`).
  - Tutorial (`first_simulation.md`).

---

## [0.1.0-dev] — 2026-05-12

### Added
- Initial project structure and repository scaffold
- Python package skeleton (`src/neurowave/`)
- `pyproject.toml` with full dependency specification
- MIT License
- Comprehensive `.gitignore`
- Project management documents:
  - `PROJECT_STATUS.md`
  - `ROADMAP.md`
  - `CHANGELOG.md` (this file)
  - `CURRENT_OBJECTIVE.md`
  - `NEXT_STEPS.md`
- Documentation directory structure (`docs/theory`, `docs/architecture`, etc.)
- Module skeleton with `__init__.py` files for all subpackages
- `neurowave.core.constants` and `neurowave.core.config`
- `neurowave.core.base` abstract base classes

### Infrastructure
- setuptools-based build system
- Development tools: black, ruff, mypy, pytest
- Test markers: slow, gpu, benchmark

---

*Format: [version] — YYYY-MM-DD*
