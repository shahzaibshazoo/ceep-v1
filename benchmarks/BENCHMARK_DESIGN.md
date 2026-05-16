# Benchmark Suite Design Document

## Overview

This comprehensive benchmark suite validates the performance of the batched 2D FDTD solver by comparing sequential vs parallel execution across multiple configurations.

## Architecture

### Component Structure

```
benchmarks/
├── batched_2d_benchmark.py      # Main benchmark runner
├── analyze_results.py            # Results analysis & report generation
├── generate_plots.py             # Visualization generation
├── quick_validation.py           # Lightweight functional test
├── run_full_benchmark.sh         # Automated pipeline orchestrator
├── BENCHMARK_README.md           # User guide & interpretation
├── BENCHMARK_DESIGN.md           # This file
└── plots/                        # Output visualizations (generated)
```

### Data Flow

```
1. batched_2d_benchmark.py
   │
   ├── Runs 14-30 configurations
   ├── Measures: time, throughput, accuracy
   └── Outputs: benchmark_raw_data.json
        │
        ├──→ analyze_results.py
        │    ├── Computes speedups
        │    ├── Validates accuracy
        │    ├── Analyzes scaling
        │    └── Generates: batched_2d_results.md
        │
        └──→ generate_plots.py
             ├── Speedup vs Batch
             ├── Throughput vs Grid
             ├── Time Comparison
             ├── Scaling Efficiency
             └── Accuracy Validation (PNG files)
```

## Measurement Strategy

### Sequential Execution
- Create N independent solvers, each with one source position
- Run them back-to-back on GPU
- Total time = sum of individual run times
- Reflects current (unoptimized) workflow

### Batched Execution
- Create one solver with N source positions
- Stack all N grids in batch dimension
- Run all N sources in parallel via single kernel launch
- Total time reflects true parallel processing

### Speedup Calculation
```
Speedup = Time(Sequential) / Time(Batched)
```

**Interpretation**:
- 1.0× = No improvement (CPU-bound or overhead dominates)
- 5-8× = Good improvement (typical for smaller problems)
- 10-15× = Target: Strong GPU utilization
- 15×+ = Excellent: GPU core utilization approaching theoretical max

### GPU vs CPU Behavior

**On GPU (Expected)**:
- Sequential: N kernel launches with ~10% GPU utilization
- Batched: 1 kernel launch per timestep with 80%+ GPU utilization
- Result: Dramatic speedup (10-15×)

**On CPU (Observed)**:
- Sequential: N sequential simulations (fully utilized, no parallelism)
- Batched: Larger arrays, same sequential execution
- Result: Batched slightly slower due to overhead
- **This is expected and normal!** Speedup only manifests with GPU parallelism

## Benchmark Configurations

### Full Suite (≈30-45 min on T4 GPU)

```
Grid Sizes:   300×300, 600×600, 1000×1000
Batch Sizes:  1, 4, 8, 16
Timesteps:    100, 200, 400
Total Runs:   30-40 configurations × 2 runs = 60-80 simulations
```

Reasoning:
- 300×300: Small grid, tests launch overhead sensitivity
- 600×600: Medium grid, typical brain imaging resolution
- 1000×1000: Large grid, tests memory scaling
- Batch 1-16: Full range from sequential to highly parallel
- Steps 100-400: Tests timestep scaling behavior

### Quick Mode (≈5 min)

```
Grid Sizes:   300×300, 600×600
Batch Sizes:  1, 4, 8
Timesteps:    100, 50
Total Runs:   ~8 configurations × 1-2 runs = ~10 simulations
```

Used for:
- Development/debugging
- Quick validation before full run
- CI/CD pipelines

## Measurement Accuracy

### Timing Resolution

```python
# Use high-resolution timer with GPU synchronization
if gpu_available:
    cp.cuda.Device().synchronize()  # Flush GPU queue
    t0 = time.perf_counter()        # High-res CPU timer
    result = solver.run()            # GPU work
    cp.cuda.Device().synchronize()  # Wait for GPU completion
    elapsed = time.perf_counter() - t0
```

**Why synchronize?**
- GPU operations are asynchronous
- Must wait for completion before measuring
- Without sync: times are meaningless (measure queue time, not work time)

### Multiple Runs
- Each configuration runs 2×
- Report mean ± std
- Detects jitter (variance in timing)
- Improves statistical confidence

### Error Metrics
- Wall-clock time ±10% acceptable
- Accuracy error < 1e-12 (machine precision)
- Speedup ±5% normal variance

## Accuracy Validation

### Why Validate?
Batching stacks multiple simulations and processes them together. Must verify:
1. Results match sequential baseline
2. No cross-talk between batch elements
3. Numerical errors don't accumulate

### Validation Method

```python
# Run same source position sequentially
result_seq = solver_seq.run()

# Run same source in batch of 1
result_batch = solver_batch.run()

# Compute error
error = max(abs(result_seq - result_batch))
assert error < 1e-12  # Machine precision
```

**Why 1e-12?**
- IEEE float64 machine epsilon ≈ 2.2e-16
- Round-trip accumulation ≈ 1e-12 for 1M operations
- Acceptable tolerance for numerical computing
- Tighter tolerance catches real bugs

### Pass Criteria
- Error < 1e-12 for all configurations: PASS ✓
- Any error > 1e-12: FAIL ✗ (indicates algorithm issue)

## Output Interpretation

### Report Sections

#### Executive Summary
- Target met? (Speedup ≥ 10×)
- Accuracy? (Error < 1e-12)
- Status summary (ready for production?)

#### Raw Data Tables
- Sequential execution: grid, batch, steps, time, throughput
- Batched execution: same measurements
- Speedup comparison: side-by-side timing

#### Performance Analysis
- Average/min/max speedup
- Accuracy validation results
- Scaling efficiency (throughput per batch element)
- Key findings (where speedup comes from)

#### Recommendations
- When to use batched solver
- Optimal configuration (batch size, grid size, timesteps)
- Memory requirements
- Production deployment advice

### Visualizations

1. **speedup_vs_batch.png**
   - X-axis: Batch size (1, 4, 8, 16)
   - Y-axis: Speedup factor
   - Multiple lines: different grid sizes
   - Green/orange lines: target range (10-15×)
   - Interpretation: How speedup improves with batch size

2. **throughput_vs_grid.png**
   - X-axis: Grid size (300×300, 600×600, 1000×1000)
   - Y-axis: Throughput (GCell-steps/s)
   - Grouped bars: batch sizes
   - Interpretation: GPU utilization scales with problem size

3. **time_comparison.png**
   - X-axis: Configurations
   - Y-axis: Wall-clock time (log scale)
   - Bars: Sequential vs Batched
   - Interpretation: Absolute timing differences

4. **scaling_efficiency.png**
   - X-axis: Batch size (log scale)
   - Y-axis: Efficiency (%)
   - Horizontal dashed line: Ideal (100%)
   - Interpretation: How well speedup scales vs theory

5. **accuracy_validation.png**
   - X-axis: Configurations
   - Y-axis: Error (log scale)
   - Horizontal dashed line: Tolerance (1e-12)
   - Colors: Green (pass) vs Red (fail)
   - Interpretation: Numerical stability

## Expected Results (GPU)

### Typical Tesla T4 (2560 CUDA cores)

| Config | Sequential (s) | Batched (s) | Speedup |
|--------|---|---|---|
| 300×300, B=4, S=100 | 2.8 | 0.35 | 8× |
| 300×300, B=8, S=100 | 5.6 | 0.55 | 10× |
| 300×300, B=16, S=100 | 11.2 | 1.0 | 11× |
| 600×600, B=8, S=100 | 11.5 | 1.2 | 10× |
| 1000×1000, B=8, S=50 | 25.0 | 2.5 | 10× |

**Average Speedup: 10-12×** ✓ Within target range

### Why GPU shows speedup

```
Sequential:                     Batched:
┌─────────────────┐           ┌──────────────┐
│ GPU Core 0-255  │           │ GPU Core 0-  │
│ (1% active)     │ ────────→ │    2560      │
│ Cores 256-2559: │ (95%      │  (90%+       │
│ IDLE            │  active)  │   active)    │
└─────────────────┘           └──────────────┘

Sequential:       Batched:
16 kernel calls   1 kernel call per timestep
High launch       Low overhead
overhead          Sustained utilization
```

### Why CPU shows no speedup

```
Sequential:       Batched:
─────────────     ──────────────────
Process B1        Process B1,B2,B3,B4
Process B2        (single thread)
Process B3        Overhead > benefit
Process B4        (same CPU cores)

Result: Batched slightly slower
```

## Quality Assurance

### Pre-Benchmark Checks

```bash
# 1. Functional test
python benchmarks/quick_validation.py

# 2. Small smoke test
PYTHONPATH=./src python benchmarks/batched_2d_benchmark.py --quick
```

### Post-Benchmark Checks

```bash
# 1. Data completeness
jq '.sequential | length' benchmarks/benchmark_raw_data.json
# Should match .batched | length

# 2. Report generation
wc -l benchmarks/batched_2d_results.md
# Should be > 50 lines

# 3. Visualization generation
ls -1 benchmarks/plots/ | wc -l
# Should be 5 PNG files
```

## Troubleshooting

### "GPU not available"
```
Expected on CPU-only systems
- Benchmark runs on CPU (slower results)
- Speedup will be < 2× (just overhead)
- Normal and expected behavior
```

### Low speedup (< 8×) on GPU
```
Check:
1. CPML overhead (complex boundary condition)
2. Grid size (too small = launch limited)
3. Batch size (too small = parallelism limited)
4. GPU utilization (nvidia-smi)
```

### Accuracy failure (error > 1e-12)
```
Critical issue - investigate:
1. CPML implementation correctness
2. Floating point accumulation
3. Array indexing in batched version
4. Compare sequential vs batched source code
```

### Memory errors
```
Reduce:
1. Grid size (300×300 → 256×256)
2. Batch size (16 → 8)
3. Timesteps (200 → 100)
Check: nvidia-smi (GPU memory usage)
```

## Maintenance

### Update Checklist

When modifying benchmarks:

- [ ] Run quick validation: `python benchmarks/quick_validation.py`
- [ ] Run quick benchmark: `./benchmarks/run_full_benchmark.sh --quick`
- [ ] Check report generates: `python benchmarks/analyze_results.py`
- [ ] Verify plots: `python benchmarks/generate_plots.py`
- [ ] Update documentation if algorithm changes
- [ ] Commit all: `git add benchmarks/ && git commit -m "..."`

### Versioning

Current version: 1.0 (as of 2026-05-16)

Future enhancements:
- [ ] 3D batching benchmark
- [ ] Memory profiling
- [ ] GPU utilization metrics (via NVIDIA profiler)
- [ ] Real-time results dashboard
- [ ] Automated performance regression detection

## References

### FDTD Theory
- Taflove & Hagness (2005): Computational Electrodynamics
- Roden & Gedney (2000): Convolution PML

### GPU Programming
- NVIDIA CUDA Programming Guide
- CuPy Documentation: https://docs.cupy.dev/

### Benchmarking Best Practices
- Gorelick & Ozsvald (2020): High Performance Python
- NVIDIA Benchmarking Guidelines

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-16  
**Author**: NeuroWave Development Team
