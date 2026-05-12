"""
NeuroWave Test Suite
====================

Tests are organized by module:
- test_core/      : Grid, config, constants
- test_solvers/   : FDTD solver correctness
- test_sources/   : Source waveforms
- test_boundaries/: Boundary condition accuracy
- test_materials/ : Material model validation
- test_cuda/      : GPU kernel correctness (requires GPU)

Run tests:
    pytest                          # All CPU tests
    pytest -m gpu                   # GPU tests only
    pytest -m "not gpu"             # Skip GPU tests
    pytest -m benchmark             # Benchmark tests only
    pytest --cov=neurowave          # With coverage
"""
