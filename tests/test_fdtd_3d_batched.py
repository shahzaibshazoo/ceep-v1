"""
Comprehensive tests for 3D batched FDTD solver.

Tests cover:
- Initialization and field setup
- Single-batch (batch=1) equivalence to sequential
- Multi-batch consistency
- CPML stability (H-field CPML with simple E-boundary)
- Material handling
- Edge cases (corners, boundaries, extreme grids)
- Performance scaling
- Numerical accuracy
"""

import numpy as np
import pytest
from ceep.solvers.fdtd_3d_batched import BatchedFDTD3D
from ceep.core.constants import C_0


# ============================================================================
# Phase 1: Initialization and Field Setup Tests
# ============================================================================


def test_batched_3d_creation():
    """Test basic BatchedFDTD3D instantiation."""
    sources = [(50, 50, 50), (60, 50, 50)]
    probes = [(55, 50, 50), (65, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=100,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    assert solver.batch == 2
    assert solver.nx == 100
    assert solver.ny == 100
    assert solver.nz == 100
    assert len(solver.source_positions) == 2
    assert len(solver.probe_positions) == 2
    assert solver.dt > 0


def test_batched_3d_field_shapes():
    """Test that field arrays have correct shapes after build."""
    sources = [(50, 50, 50), (60, 50, 50), (70, 50, 50)]
    probes = [(55, 50, 50)]

    solver = BatchedFDTD3D(
        nx=80, ny=80, nz=80, dx=1e-3, total_steps=50,
        cpml_thickness=8,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    assert not solver._built
    solver._build()

    # Check field shapes: (batch, nx, ny, nz)
    assert solver.ex.shape == (3, 80, 80, 80)
    assert solver.ey.shape == (3, 80, 80, 80)
    assert solver.ez.shape == (3, 80, 80, 80)
    assert solver.hx.shape == (3, 80, 80, 80)
    assert solver.hy.shape == (3, 80, 80, 80)
    assert solver.hz.shape == (3, 80, 80, 80)

    # Check probe recording buffer: (batch, num_probes, total_steps)
    assert solver.probe_data.shape == (3, 1, 50)


def test_batched_3d_material_region():
    """Test setting material properties in a rectangular region."""
    sources = [(50, 50, 50)]
    probes = [(60, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=100,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    solver.set_material_region(
        x_start=30, x_end=70,
        y_start=30, y_end=70,
        z_start=30, z_end=70,
        eps_r=4.0,
        sigma_e=0.1,
    )

    assert np.allclose(solver._eps_r[50, 50, 50], 4.0)
    assert np.allclose(solver._sigma_e[50, 50, 50], 0.1)
    assert np.allclose(solver._eps_r[10, 10, 10], 1.0)
    assert np.allclose(solver._sigma_e[10, 10, 10], 0.0)


def test_batched_3d_material_sphere():
    """Test setting material properties in a spherical region."""
    sources = [(50, 50, 50)]
    probes = [(70, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=100,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    solver.set_material_sphere(
        center_x=50, center_y=50, center_z=50,
        radius=10,
        eps_r=3.0,
        sigma_e=0.05,
    )

    assert np.allclose(solver._eps_r[50, 50, 50], 3.0)
    assert np.allclose(solver._eps_r[90, 90, 90], 1.0)


# ============================================================================
# Phase 2: Single-Batch Tests
# ============================================================================


def test_batched_3d_single_batch_with_source():
    """Test that batch=1 solver runs with source injection."""
    sources = [(40, 40, 40)]
    probes = [(50, 40, 40)]

    solver = BatchedFDTD3D(
        nx=80, ny=80, nz=80, dx=1e-3, total_steps=100,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    probe_dict = solver.run()

    assert 0 in probe_dict
    assert 0 in probe_dict[0]
    signal = probe_dict[0][0]

    assert len(signal) == 100
    assert np.max(np.abs(signal)) > 1e-6


# ============================================================================
# Phase 3: Multi-Batch Consistency Tests
# ============================================================================


def test_batched_3d_multi_batch_consistency():
    """Test that all batch elements receive source at their own location."""
    sources = [(40, 50, 50), (60, 50, 50)]
    probes = [(50, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=150,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    probe_dict = solver.run()

    signal_0 = probe_dict[0][0]
    signal_1 = probe_dict[1][0]

    assert not np.allclose(signal_0, signal_1, rtol=0.1)
    assert np.max(np.abs(signal_0)) > 1e-6
    assert np.max(np.abs(signal_1)) > 1e-6


def test_batched_3d_multi_batch_scaling():
    """Test that batch size doesn't significantly affect accuracy."""
    probes = [(60, 50, 50)]

    sources_1 = [(50, 50, 50)]
    solver_1 = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=100,
        cpml_thickness=10,
        source_positions=sources_1,
        probe_positions=probes,
        frequency=2e9,
    )
    result_1 = solver_1.run()
    signal_1 = result_1[0][0]

    sources_4 = [
        (50, 50, 50),
        (45, 55, 50),
        (55, 45, 50),
        (50, 50, 45),
    ]
    solver_4 = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=100,
        cpml_thickness=10,
        source_positions=sources_4,
        probe_positions=probes,
        frequency=2e9,
    )
    result_4 = solver_4.run()
    signal_4 = result_4[0][0]

    relative_error = np.linalg.norm(signal_1 - signal_4) / (
        np.linalg.norm(signal_1) + 1e-10
    )
    assert relative_error < 0.01, f"Batch scaling error: {relative_error}"


# ============================================================================
# Phase 4: CPML Stability Tests
# ============================================================================


def test_batched_3d_cpml_stability():
    """Test that CPML remains stable over many steps."""
    sources = [(50, 50, 50)]
    probes = [(65, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=500,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    probe_dict = solver.run()

    for tx_idx in probe_dict:
        for rx_idx in probe_dict[tx_idx]:
            signal = probe_dict[tx_idx][rx_idx]
            assert np.all(np.isfinite(signal))
            assert np.max(np.abs(signal)) < 1e10


# ============================================================================
# Phase 5: Material Handling Tests
# ============================================================================


def test_batched_3d_materials_in_simulation():
    """Test that material properties are used in simulation."""
    sources = [(50, 50, 50)]
    probes = [(65, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=200,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    solver.set_material_sphere(
        center_x=60, center_y=50, center_z=50,
        radius=8,
        eps_r=4.0,
        sigma_e=0.01,
    )

    result = solver.run()
    signal = result[0][0]

    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 1e-6


def test_batched_3d_conductivity():
    """Test that conductivity (losses) reduces signal amplitude."""
    sources = [(50, 50, 50)]
    probes = [(70, 50, 50)]

    solver_no_loss = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=200,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )
    result_no_loss = solver_no_loss.run()
    signal_no_loss = result_no_loss[0][0]

    solver_loss = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=200,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )
    solver_loss.set_material_region(
        x_start=0, x_end=100,
        y_start=0, y_end=100,
        z_start=0, z_end=100,
        eps_r=1.0,
        sigma_e=0.5,
    )
    result_loss = solver_loss.run()
    signal_loss = result_loss[0][0]

    amp_no_loss = np.max(np.abs(signal_no_loss))
    amp_loss = np.max(np.abs(signal_loss))

    assert amp_loss < amp_no_loss


# ============================================================================
# Phase 6: Edge Cases Tests
# ============================================================================


def test_batched_3d_source_at_boundary():
    """Test source placement near boundaries."""
    sources = [(5, 50, 50)]
    probes = [(50, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=150,
        cpml_thickness=5,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    result = solver.run()
    signal = result[0][0]

    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 1e-6


def test_batched_3d_small_grid():
    """Test with minimal grid size."""
    sources = [(20, 20, 20)]
    probes = [(25, 20, 20)]

    solver = BatchedFDTD3D(
        nx=30, ny=30, nz=30, dx=1e-3, total_steps=100,
        cpml_thickness=3,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    result = solver.run()
    signal = result[0][0]

    assert np.all(np.isfinite(signal))
    assert len(signal) == 100


def test_batched_3d_large_batch():
    """Test with large batch size."""
    batch_size = 32
    sources = [(50 + i % 8, 50 + (i // 8) % 4, 50) for i in range(batch_size)]
    probes = [(70, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=50,
        cpml_thickness=8,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    result = solver.run()

    assert len(result) == batch_size
    for b in range(batch_size):
        assert 0 in result[b]
        signal = result[b][0]
        assert np.all(np.isfinite(signal))


def test_batched_3d_many_probes():
    """Test with many probe locations."""
    sources = [(50, 50, 50)]
    probes = [
        (x, y, z)
        for x in range(40, 80, 5)
        for y in range(40, 80, 5)
        for z in range(40, 80, 5)
    ]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=100,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    result = solver.run()

    assert len(result[0]) == len(probes)
    for rx_idx in range(len(probes)):
        signal = result[0][rx_idx]
        assert np.all(np.isfinite(signal))


# ============================================================================
# Phase 7: Numerical Accuracy Tests
# ============================================================================


def test_batched_3d_wave_propagation_speed():
    """Test that wave propagates at correct speed."""
    dx = 1e-3
    sources = [(30, 50, 50)]
    probes = [(50, 50, 50), (70, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=dx, total_steps=200,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    result = solver.run()
    signal_near = result[0][0]
    signal_far = result[0][1]

    threshold = np.max(np.abs(signal_far)) * 0.01
    first_near = np.argmax(np.abs(signal_near) > threshold)
    first_far = np.argmax(np.abs(signal_far) > threshold)

    delay_steps = first_far - first_near
    distance = 20 * dx

    expected_delay_steps = distance / (C_0 * solver.dt)
    relative_error = abs(delay_steps - expected_delay_steps) / expected_delay_steps

    assert relative_error < 0.1


def test_batched_3d_no_spurious_oscillations():
    """Test that fields don't show spurious high-frequency content."""
    sources = [(50, 50, 50)]
    probes = [(65, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=300,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    result = solver.run()
    signal = result[0][0]

    diffs = np.abs(np.diff(signal))
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)

    assert max_diff < mean_diff * 50


def test_batched_3d_all_fields_bounded():
    """Test that all 6 field components remain bounded."""
    sources = [(50, 50, 50), (60, 50, 50)]
    probes = [(70, 50, 50)]

    solver = BatchedFDTD3D(
        nx=100, ny=100, nz=100, dx=1e-3, total_steps=200,
        cpml_thickness=10,
        source_positions=sources,
        probe_positions=probes,
        frequency=2e9,
    )

    probe_dict = solver.run()

    import ceep.core.backend as xpb

    assert np.all(np.isfinite(xpb.to_numpy(solver.ex)))
    assert np.all(np.isfinite(xpb.to_numpy(solver.ey)))
    assert np.all(np.isfinite(xpb.to_numpy(solver.ez)))
    assert np.all(np.isfinite(xpb.to_numpy(solver.hx)))
    assert np.all(np.isfinite(xpb.to_numpy(solver.hy)))
    assert np.all(np.isfinite(xpb.to_numpy(solver.hz)))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
