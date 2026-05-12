# Examples

This directory contains example scripts demonstrating NeuroWave capabilities.

## Planned Examples

| # | Example | Phase | Status |
|---|---------|-------|--------|
| 1 | `basic_2d_fdtd.py` — Simple 2D free-space propagation | Phase 1 | ⬜ |
| 2 | `gaussian_pulse.py` — Gaussian pulse in dielectric slab | Phase 1 | ⬜ |
| 3 | `pml_demo.py` — PML boundary condition demonstration | Phase 2 | ⬜ |
| 4 | `dispersive_material.py` — Debye material simulation | Phase 2 | ⬜ |
| 5 | `microwave_imaging.py` — Basic microwave imaging setup | Phase 4 | ⬜ |
| 6 | `differentiable_fdtd.py` — Gradient through simulation | Phase 5 | ⬜ |

## Running Examples

```bash
# After installing neurowave
pip install -e ".[all]"

# Run an example
python examples/basic_2d_fdtd.py
```
