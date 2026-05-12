# Benchmarks

Performance benchmarks for NeuroWave.

## Running Benchmarks

```bash
# Run all benchmarks
pytest -m benchmark benchmarks/

# Run with detailed timing
pytest -m benchmark --benchmark-json=results.json benchmarks/
```

## Results

Results will be stored in `benchmarks/results/` directory.
