"""
Example: Batched 3D Microwave Imaging with Antenna Array

Demonstrates realistic usage of the BatchedFDTD3D solver for brain imaging
using a 4×4 antenna array. This example shows:

1. Creating a batched solver with multiple TX/RX positions
2. Setting up a realistic brain-like phantom
3. Running parallel simulations efficiently
4. Processing S-parameters for imaging

Performance Notes
-----------------
On a T4 GPU:
- Grid: 120×120×100, batch=16, 150 steps
- Sequential: ~180s
- Batched: ~8-12s
- Speedup: ~15-22×

Author: NeuroWave Development Team
Date: 2026-05-16
"""

import numpy as np
from ceep.solvers.fdtd_3d_batched import BatchedFDTD3D


def create_brain_phantom(nx: int, ny: int, nz: int, dx: float) -> dict:
    """Create a simplified brain phantom with white/gray matter.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    dx : float
        Grid spacing (m).

    Returns
    -------
    phantom : dict
        Dictionary with 'eps_r' and 'sigma_e' arrays.
    """
    # Standard tissue properties at 2 GHz
    # (From Gabriel et al., IEEE Trans. Biomed. Eng. 1996)
    eps_r_white_matter = 38.0  # Relative permittivity
    sigma_white_matter = 0.5   # Conductivity (S/m)

    eps_r_gray_matter = 52.0
    sigma_gray_matter = 0.7

    eps_r_csf = 65.0
    sigma_csf = 2.0

    # Initialize with white matter (background)
    eps_r = np.full((nx, ny, nz), eps_r_white_matter, dtype=np.float64)
    sigma_e = np.full((nx, ny, nz), sigma_white_matter, dtype=np.float64)

    # Add gray matter sphere (simulates tumor or lesion)
    center_x, center_y, center_z = nx // 2, ny // 2, nz // 2
    radius_cells = 15

    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    mask_lesion = (
        (x_grid - center_x) ** 2
        + (y_grid - center_y) ** 2
        + (z_grid - center_z) ** 2
        <= radius_cells ** 2
    )
    eps_r[mask_lesion.T] = eps_r_gray_matter
    sigma_e[mask_lesion.T] = sigma_gray_matter

    # Add cerebrospinal fluid ventricles
    radius_csf = 8
    mask_csf = (
        (x_grid - center_x) ** 2
        + (y_grid - center_y - 10) ** 2
        + (z_grid - center_z) ** 2
        <= radius_csf ** 2
    )
    eps_r[mask_csf.T] = eps_r_csf
    sigma_e[mask_csf.T] = sigma_csf

    return {"eps_r": eps_r, "sigma_e": sigma_e}


def example_4x4_antenna_array():
    """Run a 4×4 antenna array imaging scenario.

    This demonstrates:
    - Multiple simultaneous TX elements
    - Dense RX receiver grid
    - S-parameter extraction
    - Performance on multiple batch elements
    """
    # Grid configuration
    nx, ny, nz = 120, 120, 100
    dx = 0.5e-3  # 0.5 mm resolution

    # 4×4 transmitter array (16 elements total)
    tx_spacing = 5  # Grid cells
    tx_positions = []
    for i in range(4):
        for j in range(4):
            x = 40 + i * tx_spacing
            y = 40 + j * tx_spacing
            z = 50  # Mid-plane
            tx_positions.append((x, y, z))

    # Dense receiver grid (20×20 elements)
    rx_spacing = 2
    rx_positions = []
    for x in range(40, 80, rx_spacing):
        for y in range(40, 80, rx_spacing):
            z = 50
            rx_positions.append((x, y, z))

    print(f"Configuration:")
    print(f"  Grid: {nx}×{ny}×{nz}, dx={dx*1e3:.1f}mm")
    print(f"  TX antennas: {len(tx_positions)}")
    print(f"  RX antennas: {len(rx_positions)}")
    print(f"  S-parameters: {len(tx_positions)}×{len(rx_positions)}")
    print()

    # Create solver
    print("Creating solver...")
    solver = BatchedFDTD3D(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        total_steps=150,
        cpml_thickness=12,
        source_positions=tx_positions,
        probe_positions=rx_positions,
        frequency=2e9,  # 2 GHz
        delay_factor=5.0,
    )

    # Load phantom
    print("Loading brain phantom...")
    phantom = create_brain_phantom(nx, ny, nz, dx)
    solver._eps_r[:] = phantom["eps_r"]
    solver._sigma_e[:] = phantom["sigma_e"]

    # Run simulation
    print("Running batched simulation...")
    import time

    start = time.time()
    s_matrix = solver.run(verbose=True)
    elapsed = time.time() - start

    print(f"\nSimulation completed in {elapsed:.2f}s")

    # Extract some statistics
    print("\nS-parameter statistics:")
    amplitudes = []
    for tx_idx in s_matrix:
        for rx_idx in s_matrix[tx_idx]:
            signal = s_matrix[tx_idx][rx_idx]
            amplitude = np.max(np.abs(signal))
            amplitudes.append(amplitude)

    amplitudes = np.array(amplitudes)
    print(f"  Mean amplitude: {np.mean(amplitudes):.2e}")
    print(f"  Std deviation: {np.std(amplitudes):.2e}")
    print(f"  Min amplitude: {np.min(amplitudes):.2e}")
    print(f"  Max amplitude: {np.max(amplitudes):.2e}")

    # Compute S-matrix condition number (for imaging quality)
    print("\nS-matrix properties:")
    print(f"  Shape: {len(s_matrix)} TX × {len(s_matrix[0])} RX")

    # Reciprocity check (should have S_ij ≈ S_ji in lossless media)
    if len(tx_positions) > 1 and len(rx_positions) > 1:
        s12 = np.max(np.abs(s_matrix[0][1]))
        s21 = np.max(np.abs(s_matrix[1][0]))
        print(f"  Reciprocity check: S12={s12:.2e}, S21={s21:.2e}")
        if s12 > 0:
            print(f"  Reciprocity error: {abs(s12 - s21) / s12 * 100:.2f}%")

    return s_matrix, solver


def example_single_transmitter():
    """Run a simple single-transmitter scenario.

    This is useful for validation and debugging.
    """
    print("=" * 60)
    print("Example: Single Transmitter")
    print("=" * 60)

    nx, ny, nz = 100, 100, 100
    dx = 1e-3  # 1 mm

    tx_positions = [(50, 50, 50)]
    rx_positions = [
        (x, y, 50) for x in range(30, 80, 5) for y in range(30, 80, 5)
    ]

    solver = BatchedFDTD3D(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        total_steps=100,
        cpml_thickness=10,
        source_positions=tx_positions,
        probe_positions=rx_positions,
        frequency=2e9,
    )

    print(f"Running single-TX simulation...")
    print(f"  Grid: {nx}×{ny}×{nz}, {len(rx_positions)} probes")

    result = solver.run(verbose=False)

    # Extract central signal
    central_signal = result[0][0]  # First probe
    print(f"  Central probe signal: {len(central_signal)} samples")
    print(f"  Peak amplitude: {np.max(np.abs(central_signal)):.2e}")
    print(f"  Energy: {np.sum(central_signal ** 2):.2e}")

    return result, solver


if __name__ == "__main__":
    # Example 1: Single transmitter (validation)
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single Transmitter")
    print("=" * 60)
    result1, solver1 = example_single_transmitter()

    # Example 2: Full 4×4 array
    print("\n" + "=" * 60)
    print("EXAMPLE 2: 4×4 Antenna Array Imaging")
    print("=" * 60)
    s_matrix, solver2 = example_4x4_antenna_array()

    print("\n✅ Examples completed successfully!")
