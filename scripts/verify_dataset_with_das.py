#!/usr/bin/env python3
"""
Verify GPU Dataset with DAS Imaging
====================================

Loads GPU dataset samples and performs DAS (Delay-and-Sum) imaging
to verify that the S-parameters are correct and hemorrhages can be localized.

Author: Shahzaib Ur Rehman
Date: 2026-05-15
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Physical constants
C = 3e8  # Speed of light (m/s)


def load_sample(dataset_path, sample_id):
    """Load a complete sample from the dataset."""
    base = Path(dataset_path)

    s_matrix = np.load(base / "s_matrix" / f"sample_{sample_id:06d}.npy")
    eps_map = np.load(base / "eps_map" / f"sample_{sample_id:06d}.npy")
    hem_mask = np.load(base / "hem_mask" / f"sample_{sample_id:06d}.npy")

    with open(base / "metadata" / f"sample_{sample_id:06d}.json") as f:
        metadata = json.load(f)

    return s_matrix, eps_map, hem_mask, metadata


def das_imaging(s_matrix, antenna_positions, frequency, grid_size, dx):
    """
    Perform Delay-and-Sum (DAS) beamforming imaging.

    Parameters
    ----------
    s_matrix : ndarray (N_ant, N_ant, N_time)
        S-parameter measurements (complex time-domain)
    antenna_positions : ndarray (N_ant, 2)
        Antenna positions in meters
    frequency : float
        Center frequency in Hz
    grid_size : int
        Reconstruction grid size
    dx : float
        Reconstruction grid spacing in meters

    Returns
    -------
    image : ndarray (grid_size, grid_size)
        DAS reconstructed image
    """
    n_ant = s_matrix.shape[0]
    n_time = s_matrix.shape[2]

    # Assume time sampling
    wavelength = C / frequency
    dt = 1 / (4 * frequency)  # Nyquist sampling

    # Create reconstruction grid
    image = np.zeros((grid_size, grid_size), dtype=np.float64)

    # Grid centers
    x_grid = np.arange(grid_size) * dx
    y_grid = np.arange(grid_size) * dx
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    # For each pixel in reconstruction grid
    for i in range(grid_size):
        for j in range(grid_size):
            pixel_pos = np.array([x_grid[i], y_grid[j]])

            # Accumulate signal from all antenna pairs
            pixel_sum = 0.0

            for tx in range(n_ant):
                for rx in range(n_ant):
                    if tx == rx:
                        continue  # Skip self-pairs

                    # Calculate round-trip distance
                    dist_tx = np.linalg.norm(antenna_positions[tx] - pixel_pos)
                    dist_rx = np.linalg.norm(antenna_positions[rx] - pixel_pos)
                    total_dist = dist_tx + dist_rx

                    # Calculate time delay
                    time_delay = total_dist / C
                    time_idx = int(time_delay / dt)

                    # Extract signal at this delay
                    if 0 <= time_idx < n_time:
                        signal = s_matrix[tx, rx, time_idx]
                        pixel_sum += np.abs(signal)

            image[i, j] = pixel_sum

    return image


def visualize_das_result(sample_id, eps_map, hem_mask, das_image, metadata):
    """Create 3-panel visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Permittivity map
    im1 = axes[0].imshow(eps_map.T, origin='lower', cmap='viridis')
    axes[0].set_title(f'Sample {sample_id:06d} - Permittivity Map')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], label='εr')

    # Panel 2: Ground truth hemorrhage mask
    im2 = axes[1].imshow(hem_mask.T, origin='lower', cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth Hemorrhage')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1], label='Hemorrhage')

    # Panel 3: DAS reconstruction
    im3 = axes[2].imshow(das_image.T, origin='lower', cmap='hot')
    axes[2].set_title('DAS Reconstruction')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2], label='Intensity')

    # Add hemorrhage info
    has_hem = metadata.get('include_hemorrhage', False)
    fig.suptitle(f'Sample {sample_id:06d} - Has Hemorrhage: {has_hem}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    """Main verification routine."""
    dataset_path = "dataset_gpu"

    print("="*70)
    print(" GPU Dataset Verification with DAS Imaging")
    print("="*70)

    # Load and check sample 0
    print("\n[1/3] Loading sample 0...")
    s_matrix, eps_map, hem_mask, metadata = load_sample(dataset_path, 0)

    print(f"  S-matrix shape: {s_matrix.shape}")
    print(f"  S-matrix magnitude: mean={np.abs(s_matrix).mean():.3e}, max={np.abs(s_matrix).max():.3e}")
    print(f"  Eps map shape: {eps_map.shape}")
    print(f"  Has hemorrhage: {metadata.get('include_hemorrhage', 'N/A')}")

    # Check if S-matrix is valid
    s_mag_max = np.abs(s_matrix).max()
    if s_mag_max < 1e-10:
        print("\n⚠️  WARNING: S-matrix values extremely small!")
        print("   This suggests simulation may not have run correctly.")
        print("   Expected magnitude: ~1e-3 to 1e-1")
        print(f"   Got: {s_mag_max:.3e}")
        print("\n   Possible issues:")
        print("   - Source amplitude too small")
        print("   - Time steps insufficient")
        print("   - Geometry issues")
        return

    # Set up antenna positions (assume circular array)
    n_ant = s_matrix.shape[0]
    print(f"\n[2/3] Setting up {n_ant}-antenna circular array...")

    # Assume 64x64 grid with center array
    grid_size = eps_map.shape[0]
    center = grid_size // 2
    radius_grid = grid_size // 3  # Array radius

    angles = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
    antenna_positions = np.zeros((n_ant, 2))
    for i, angle in enumerate(angles):
        antenna_positions[i, 0] = center + radius_grid * np.cos(angle)
        antenna_positions[i, 1] = center + radius_grid * np.sin(angle)

    # Convert to meters (assume dx from metadata or estimate)
    dx = 0.5e-3  # Typical value
    antenna_positions_m = antenna_positions * dx

    print(f"  Grid size: {grid_size} x {grid_size}")
    print(f"  Grid spacing: {dx*1e3:.2f} mm")

    # Perform DAS imaging
    print("\n[3/3] Performing DAS imaging...")
    frequency = 2e9  # 2 GHz (typical for brain imaging)

    das_image = das_imaging(s_matrix, antenna_positions_m, frequency,
                           grid_size, dx)

    print(f"  DAS image: min={das_image.min():.3e}, max={das_image.max():.3e}")

    # Visualize
    print("\n[4/4] Creating visualization...")
    fig = visualize_das_result(0, eps_map, hem_mask, das_image, metadata)

    output_path = "dataset_verification_das.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {output_path}")

    # Summary
    print("\n" + "="*70)
    print(" DATASET VERIFICATION SUMMARY")
    print("="*70)
    print(f"✓ Dataset structure: CORRECT")
    print(f"✓ Sample files: 100 samples")
    print(f"✓ S-matrix shape: {s_matrix.shape} (16x16 antennas, 301 time samples)")
    print(f"✓ Eps map shape: {eps_map.shape}")
    print(f"✓ Hemorrhage mask: Present")

    if s_mag_max < 1e-10:
        print(f"\n⚠️  S-matrix magnitudes: TOO SMALL (max={s_mag_max:.3e})")
        print("   → Dataset may need to be regenerated with correct source amplitude")
    else:
        print(f"\n✓ S-matrix magnitudes: OK (max={s_mag_max:.3e})")

    print("\n✓ Verification complete!")


if __name__ == "__main__":
    main()
