"""
neurowave.cuda — CUDA Kernels and GPU Memory Management
=======================================================

This module contains all GPU-accelerated implementations:

- **Kernels**: CUDA kernels for E-field and H-field updates
- **Memory**: GPU memory allocation and transfer utilities
- **Launch**: Kernel launch configuration (grid/block dimensions)
- **Profiling**: GPU performance measurement tools

GPU Engineering Considerations
------------------------------
All kernels are designed with attention to:
- Memory coalescing (row-major access patterns)
- Shared memory utilization for halo regions
- Occupancy optimization
- Minimal warp divergence
- Efficient synchronization

Backend Support (Planned)
-------------------------
- CuPy kernels (primary, fast prototyping)
- Raw CUDA via pybind11 (performance-critical paths)
- Numba CUDA (experimental)
- Triton kernels (future)
"""
