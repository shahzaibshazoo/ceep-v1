# Benchmarks Documentation

This directory contains performance benchmarks and validation results.

## Planned Benchmarks

| Benchmark | Description | Status |
|-----------|-------------|--------|
| CPU vs GPU FDTD | Compare NumPy vs CuPy vs CUDA kernel performance | ⬜ |
| Grid scaling | Performance vs grid size (memory throughput) | ⬜ |
| PML accuracy | Reflection coefficient vs PML parameters | ⬜ |
| Material accuracy | Dispersive model vs analytical solutions | ⬜ |
| Multi-GPU scaling | Scaling efficiency with multiple GPUs | ⬜ |
| Memory usage | Peak memory vs grid size | ⬜ |

## Benchmark Methodology

All benchmarks will:
1. Report median of N=5 runs (after warmup)
2. Report memory usage (peak and average)
3. Compare against Meep where applicable
4. Include reproducibility information (hardware, software versions)
