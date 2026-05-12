# NeuroWave 🧠⚡

> **GPU-Accelerated Electromagnetic Simulation Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-downloads)

---

## Overview

**NeuroWave** is a modern, GPU-native electromagnetic simulation framework designed as a high-performance alternative to [Meep/PyMeep](https://meep.readthedocs.io/). It provides GPU-accelerated Maxwell equation solvers with a Python-first API, targeting research-grade computational electromagnetics, biomedical microwave imaging, and AI-integrated physics simulation.

### Why NeuroWave?

| Feature | Meep | NeuroWave |
|---------|------|-----------|
| **Compute Backend** | CPU (MPI parallel) | GPU-native (CUDA) |
| **API** | C++ / Python wrapper | Python-first |
| **AI Integration** | None | PyTorch native |
| **Differentiable** | No | Yes (planned) |
| **Biomedical Focus** | General purpose | Specialized modules |

## Core Goals

1. 🚀 **GPU-Accelerated Maxwell Solver** — CUDA-native FDTD engine
2. 🐍 **Python-First API** — Intuitive, NumPy-style interface
3. 🔥 **CUDA Backend** — Memory-coalesced, warp-optimized kernels
4. 🧠 **AI-Native Architecture** — PyTorch autograd compatible
5. 🏥 **Biomedical Imaging** — Microwave/mmWave tissue imaging
6. 📐 **Differentiable Simulation** — Gradient-based inverse design
7. 🧩 **Modular Architecture** — Pluggable solvers, materials, boundaries
8. 📚 **Complete Documentation** — Theory, tutorials, benchmarks
9. 🔬 **Reproducible Experiments** — Deterministic, logged, versioned
10. 📄 **Publication-Ready** — Benchmarked, validated, citable

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/neurowave.git
cd neurowave

# Install in development mode
pip install -e ".[dev]"

# Run a basic simulation (coming soon)
python examples/basic_2d_fdtd.py
```

## Architecture

```
neurowave/
├── core/          # Grid, field arrays, simulation engine
├── cuda/          # CUDA kernels, GPU memory management
├── materials/     # Dielectric models, tissue libraries
├── boundaries/    # ABC, PML, periodic boundaries
├── sources/       # Gaussian, sinusoidal, custom sources
├── solvers/       # FDTD, frequency-domain extractors
├── visualization/ # Real-time & post-processing visualization
└── ai/            # PyTorch integration, differentiable physics
```

## Development Phases

- **Phase 1** — Core GPU FDTD (2D Yee grid, TMz/TEz, CUDA kernels) ← *Current*
- **Phase 2** — Advanced FDTD (PML, dispersive materials, S-parameters)
- **Phase 3** — 3D Engine (full 3D Yee grid, multi-GPU)
- **Phase 4** — Biomedical Imaging (tissue models, tumor phantoms)
- **Phase 5** — AI Integration (differentiable sim, inverse imaging)
- **Phase 6** — Advanced HPC (Tensor Cores, distributed, JAX/Triton)

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Core** | Python, NumPy, PyTorch |
| **GPU** | CUDA, CuPy, Numba |
| **Bindings** | pybind11, C++ |
| **Data** | HDF5, NumPy arrays |
| **Visualization** | Matplotlib |
| **Optional** | JAX, Triton, OpenCL, Vulkan, MPI |

## Philosophy

> Correctness → Reproducibility → Performance → Elegance

Every feature is documented, benchmarked, tested, modular, and designed for future scaling.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use NeuroWave in your research, please cite:

```bibtex
@software{neurowave2026,
  title={NeuroWave: GPU-Accelerated Electromagnetic Simulation Framework},
  year={2026},
  url={https://github.com/your-org/neurowave}
}
```

---

*Built with ⚡ for the computational electromagnetics community.*
