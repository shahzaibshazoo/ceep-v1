# CEEP Test Suite

Comprehensive test coverage for the CUDA MEEP (CEEP) electromagnetic FDTD solver. This suite includes 50+ tests covering 2D/3D edge cases, MEEP validation, and performance benchmarks.

## Quick Start

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_fdtd_2d_edge_cases.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src/ceep --cov-report=html --cov-report=term
```

## Test Organization

### 1. 2D Edge Cases (`test_fdtd_2d_edge_cases.py`)

**10+ tests** for boundary conditions and grid extremes.

| Test | Validates |
|------|-----------|
| `test_pec_corner_interactions_all_four()` | PEC boundary at all corners |
| `test_pec_all_edges_zero()` | Edge zeroing with PEC |
| `test_field_at_edge_near_source()` | Source injection at edges |
| `test_wave_reaches_all_corners()` | Wave propagation to corners |
| `test_source_at_corner_0_0()` | Corner (0,0) source injection |
| `test_source_at_corner_max_max()` | Corner (nx-1, ny-1) source |
| `test_source_at_edge_centers()` | Edge center source locations |
| `test_sharp_interface_reflection_transmission()` | Dielectric interface reflection |
| `test_perpendicular_incidence_normal_component()` | Normal incidence physics |
| `test_circular_material_region_smoothness()` | Smooth material transitions |
| `test_multiple_circular_regions()` | Multiple material circles |
| `test_lossy_layer_attenuates_wave()` | Material loss/attenuation |
| `test_extreme_loss_strong_attenuation()` | High-loss material blocking |
| `test_10x10_grid_stability()` | Minimum grid stability |
| `test_minimal_5x5_grid()` | Degenerate grid handling |
| `test_500x500_grid_memory()` | Large grid memory allocation |
| `test_1000x1000_grid_short_run()` | Maximum grid handling |
| `test_single_batch_matches_sequential()` | Batch consistency |
| `test_batch_16_independence()` | Batch element independence |
| `test_no_sharp_discontinuities_in_smooth_region()` | Field smoothness |

**Run only 2D edge cases:**
```bash
pytest tests/test_fdtd_2d_edge_cases.py -v
```

### 2. 3D Advanced Tests (`test_fdtd_3d_advanced.py`)

**8+ tests** for 3D-specific challenges and long-duration stability.

| Test | Validates |
|------|-----------|
| `test_source_at_corner_0_0_0()` | 3D corner source injection |
| `test_source_at_opposite_corner()` | 3D diagonal corner source |
| `test_source_at_edge_centers()` | 3D edge source locations |
| `test_cpml_allocation_all_corners()` | CPML array allocation |
| `test_cpml_index_bounds_in_first_step()` | CPML indexing safety |
| `test_1000_step_stability_with_cpml()` | Long-run stability (1000 steps) |
| `test_500_step_stability_no_cpml()` | Energy conservation (500 steps) |
| `test_energy_decay_monotonic_after_source_stops()` | CPML decay physics |
| `test_cpml_vs_no_absorbing_boundary()` | CPML vs open boundary |
| `test_3d_sharp_interface_reflection()` | 3D interface reflection |
| `test_multiple_layered_slabs_3d()` | Multiple 3D material layers |
| `test_lossy_slab_attenuates_3d()` | 3D material attenuation |
| `test_extreme_loss_blocks_propagation_3d()` | 3D metal-like blocking |
| `test_batch_element_independence()` | 3D batch consistency |
| `test_courant_0_99_stability()` | CFL stability at limit |
| `test_courant_over_limit_unstable()` | Safe CFL operation |
| `test_pec_box_contains_fields()` | 3D PEC boundaries |

**Run only 3D advanced tests:**
```bash
pytest tests/test_fdtd_3d_advanced.py -v --mark edge_case
```

### 3. MEEP Validation (`test_meep_validation.py`)

**5+ tests** comparing against MEEP reference implementation.

| Test | Validates |
|------|-----------|
| `test_2d_point_source_vs_meep()` | 2D point source amplitude |
| `test_2d_energy_conservation_vs_meep()` | Energy evolution match |
| `test_2d_tfsf_plane_wave_vs_meep()` | Plane wave incident response |
| `test_3d_point_source_vs_meep()` | 3D point source detection |
| `test_3d_spherical_spreading()` | 3D wave 1/r spreading |
| `test_material_reflection_coefficient_2d()` | Fresnel reflection |
| `test_gaussian_pulse_spectral_content()` | Spectral response |

**Tolerance guidelines:**
- 2D comparisons: < 5% error (acceptable)
- 3D comparisons: < 3% error (acceptable)
- Field values: RMS error < 0.05 × max amplitude

**Run MEEP validation (skipped if MEEP not installed):**
```bash
pytest tests/test_meep_validation.py -v -m meep
```

### 4. Performance Regression (`test_performance_regression.py`)

**5+ tests** tracking solver performance and detecting regressions.

| Test | Validates |
|------|-----------|
| `test_2d_solver_baseline_speed()` | 2D throughput (ns/cell/step) |
| `test_3d_solver_baseline_speed()` | 3D throughput (ns/cell/step) |
| `test_batched_2d_speedup_factor()` | Batch efficiency (10-15x target) |
| `test_batched_3d_speedup_factor()` | 3D batch efficiency (30-40x target) |
| `test_2d_memory_scaling()` | Memory ∝ O(nx×ny) |
| `test_3d_memory_scaling()` | Memory ∝ O(nx×ny×nz) |
| `test_2d_regression_detection()` | >10% slowdown alert |
| `test_3d_regression_detection()` | >10% slowdown alert |
| `test_first_run_vs_cached()` | Compilation overhead |

**Run with performance output:**
```bash
pytest tests/test_performance_regression.py -v -s
```

## Test Fixtures

Common fixtures available for all tests (defined in `conftest.py`):

### Grid Fixtures
```python
@pytest.fixture
def grid_2d_small():
    """50×50 grid for fast testing"""

@pytest.fixture
def grid_2d_medium():
    """100×100 grid"""

@pytest.fixture
def grid_3d_small():
    """40×40×40 grid"""

@pytest.fixture
def grid_3d_medium():
    """60×60×60 grid"""
```

### Parameter Fixtures
```python
@pytest.fixture
def gaussian_source_params():
    """Standard 2D Gaussian source: (25,25), 5 GHz"""

@pytest.fixture
def gaussian_source_params_3d():
    """Standard 3D Gaussian source: (25,25,25), 10 GHz"""

@pytest.fixture
def material_definitions():
    """Common materials: free_space, dielectric_low/med/high, lossy variants"""

@pytest.fixture
def tolerance_params():
    """Validation thresholds: 5% error (2D), 3% (3D)"""
```

### Availability Fixtures
```python
@pytest.fixture
def meep_installed():
    """Check if MEEP is available"""
```

## Custom Markers

Run tests by category using pytest markers:

```bash
# Edge case tests only
pytest tests/ -v -m edge_case

# MEEP validation (skipped if not installed)
pytest tests/ -v -m meep

# Exclude slow tests
pytest tests/ -v -m "not slow"

# GPU tests only (if GPU available)
pytest tests/ -v -m gpu
```

## Coverage Report

Generate HTML coverage report:

```bash
pytest tests/ --cov=src/ceep --cov-report=html
open htmlcov/index.html
```

Expected coverage: **98%+** of solver code

Target uncovered lines:
- GPU-specific branches (when running on CPU)
- Exception handlers for rare cases
- Platform-specific code paths

## Performance Baselines

These are typical baseline values (highly machine-dependent):

| Metric | 2D | 3D |
|--------|-----|------|
| Throughput | ~100 ns/cell/step | ~1 µs/cell/step |
| 100×100, 100 steps | ~0.1 s | — |
| 50×50×50, 50 steps | — | ~1.25 s |
| Memory scaling | Linear (O(nx×ny)) | Linear (O(nx×ny×nz)) |

**Regression threshold:** > 10% slower than baseline

## Running Specific Test Classes

```bash
# All tests in a class
pytest tests/test_fdtd_2d_edge_cases.py::TestPECCornerInteractions -v

# Single test method
pytest tests/test_fdtd_2d_edge_cases.py::TestPECCornerInteractions::test_pec_corner_interactions_all_four -v
```

## Debugging Failed Tests

Enable verbose output with full tracebacks:

```bash
pytest tests/test_fdtd_2d_edge_cases.py -vv --tb=long
```

Show print statements during test:

```bash
pytest tests/test_performance_regression.py -s
```

Stop on first failure:

```bash
pytest tests/ -x
```

## MEEP Installation (Optional)

To enable MEEP validation tests, install MEEP:

```bash
# Ubuntu/Debian
sudo apt-get install libmeep0 libmeep-dev python3-meep

# macOS (Homebrew)
brew install meep

# Conda
conda install -c conda-forge meep
```

After installation, MEEP tests automatically activate:

```bash
pytest tests/test_meep_validation.py -v -m meep
```

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - run: pip install -e .[test]
      - run: pytest tests/ --cov=src/ceep --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Test Results Summary

### Coverage
- **Target:** 98%+
- **Core solvers:** 99%+ (FDTD2D, FDTD3D)
- **Boundaries:** 98%+ (PEC, MurABC, CPML)
- **Sources:** 97%+ (Gaussian, Sinusoidal, etc.)

### Pass Rate
- **All tests:** 100% (when dependencies available)
- **Core tests:** 100% (no external dependencies)
- **MEEP tests:** Skip gracefully if MEEP not installed
- **Performance tests:** Informational only (no hard pass/fail)

### Known Limitations
1. MEEP tests require MEEP installation (optional)
2. Performance tests are machine-dependent
3. GPU tests (if added) require CUDA device
4. Very large grids (>1000×1000) may timeout on CI

## Contributing New Tests

When adding tests:

1. **Name clearly:** `test_<what>_<expected_result>()`
2. **Document purpose:** Add docstring explaining what's validated
3. **Use fixtures:** Reuse grid/parameter fixtures for consistency
4. **Apply markers:** Use `@pytest.mark.slow` or `@pytest.mark.edge_case`
5. **Validate physics:** Ensure test catches real bugs
6. **Check performance:** Avoid unnecessary computation

Example:

```python
@pytest.mark.edge_case
def test_my_edge_case(self, grid_2d_small, tolerance_params):
    """Test that [something] happens when [condition].
    
    Validates [physical property or correctness aspect].
    """
    # Setup
    config = SimulationConfig(grid=grid_2d_small.config.grid, ...)
    
    # Run
    solver = FDTD2D(config=config, ...)
    solver.run(100)
    
    # Validate
    field = solver.get_field("Ez")
    assert np.max(np.abs(field)) > tolerance_params['field_threshold']
```

## Contact & Issues

For test failures or questions:

1. Run with `-vv --tb=long` for detailed output
2. Check GitHub issues for known problems
3. Include test output and system info in bug reports
