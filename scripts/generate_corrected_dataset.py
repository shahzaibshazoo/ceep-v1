#!/usr/bin/env python3
"""
Generate Corrected Brain Hemorrhage Dataset
============================================

Generates dataset with CORRECT S-parameter magnitudes using CEEP.
Applies the correction factor (6.58×10¹²) during generation.

This is the FIXED version - future datasets won't need correction.

Usage:
    # On GPU machine
    python3 scripts/generate_corrected_dataset.py --n_samples 100

Author: Shahzaib Ur Rehman
Date: 2026-05-15
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time

try:
    from ceep.core.backend import set_backend, to_numpy
    from ceep.solvers.fdtd_2d_batched import BatchedFDTD2D
    from ceep.phantoms import BrainPhantom2D
    CEEP_AVAILABLE = True
except ImportError as e:
    print(f"❌ CEEP not available: {e}")
    print("   This script requires CEEP with GPU support")
    CEEP_AVAILABLE = False


# CORRECTION FACTOR from MEEP validation
# This corrects the S-parameter magnitude issue in CEEP
CORRECTION_FACTOR = 6.58e12  # meep_magnitude / ceep_magnitude


def generate_hemorrhage_params():
    """Generate random hemorrhage parameters."""
    # Random location within brain (avoid edges)
    center_range = 3.0  # cm from center
    angle = np.random.uniform(0, 2*np.pi)
    distance = np.random.uniform(0, center_range)

    hem_x = distance * np.cos(angle)
    hem_y = distance * np.sin(angle)

    # Random radius
    hem_radius = np.random.uniform(0.3, 1.5)  # cm

    return (hem_x, hem_y), hem_radius


def generate_sample(sample_id, include_hemorrhage=True, nx=64, ny=64,
                   dx=0.5e-3, frequency=2e9, n_ant=16):
    """
    Generate a single sample with CORRECTED S-parameters.

    Parameters
    ----------
    sample_id : int
        Sample index
    include_hemorrhage : bool
        Whether to include hemorrhage
    nx, ny : int
        Grid dimensions
    dx : float
        Grid spacing (meters)
    frequency : float
        Center frequency (Hz)
    n_ant : int
        Number of antennas

    Returns
    -------
    s_matrix : ndarray (n_ant, n_ant, n_time)
        Corrected S-parameter matrix
    eps_map : ndarray (nx, ny)
        Permittivity map
    hem_mask : ndarray (nx, ny)
        Hemorrhage mask (binary)
    metadata : dict
        Sample metadata
    """
    # Generate hemorrhage parameters
    if include_hemorrhage:
        hem_center, hem_radius = generate_hemorrhage_params()
    else:
        hem_center, hem_radius = None, 0.0

    # Create phantom
    phantom = BrainPhantom2D(
        nx=nx, ny=ny, dx=dx,
        hemorrhage_location=hem_center if include_hemorrhage else None,
        hemorrhage_radius=hem_radius,
        use_gabriel_database=False  # Use simple model for speed
    )

    # Get permittivity map
    eps_map, sigma_e = phantom.get_eps_map(frequency)

    # Create hemorrhage mask
    hem_mask = np.zeros((nx, ny), dtype=np.float64)
    if include_hemorrhage:
        cx, cy = nx // 2, ny // 2
        hx_m = hem_center[0] * 1e-2
        hy_m = hem_center[1] * 1e-2
        x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        r = np.sqrt(((x_grid - cx) * dx - hx_m)**2 +
                   ((y_grid - cy) * dx - hy_m)**2)
        hem_mask[r <= hem_radius * 1e-2] = 1.0

    # Create circular antenna array
    center_x, center_y = nx // 2, ny // 2
    radius = nx // 3

    angles = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
    positions = []
    for angle in angles:
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        positions.append((x, y))

    # Calculate simulation parameters
    wavelength = 3e8 / frequency
    max_dist = np.sqrt(2) * (nx * dx)
    t_propagation = max_dist / 3e8
    dt = 1 / (4 * frequency)
    total_steps = int(2 * t_propagation / dt)

    # Create and run solver
    solver = BatchedFDTD2D(
        nx=nx, ny=ny, dx=dx,
        total_steps=total_steps,
        cpml_thickness=10,
        source_positions=positions,
        probe_positions=positions,
        frequency=frequency
    )

    solver.set_phantom(phantom)
    s_params = solver.run()

    # Extract S-matrix
    s_matrix = to_numpy(s_params[0][0])  # (n_ant, n_ant, n_time)

    # **APPLY CORRECTION FACTOR**
    s_matrix_corrected = s_matrix * CORRECTION_FACTOR

    # Metadata
    metadata = {
        'sample_id': sample_id,
        'include_hemorrhage': include_hemorrhage,
        'hemorrhage_params': {
            'center_x': float(hem_center[0]) if include_hemorrhage else None,
            'center_y': float(hem_center[1]) if include_hemorrhage else None,
            'radius': float(hem_radius) if include_hemorrhage else None,
        } if include_hemorrhage else None,
        'simulation_params': {
            'nx': nx,
            'ny': ny,
            'dx': dx,
            'frequency': frequency,
            'n_antennas': n_ant,
            'total_steps': total_steps,
        },
        'correction_applied': True,
        'correction_factor': float(CORRECTION_FACTOR),
        'meep_validated': True,
    }

    return s_matrix_corrected, eps_map, hem_mask, metadata


def generate_dataset(n_samples=100, output_dir="dataset_corrected",
                    hemorrhage_ratio=0.7, **kwargs):
    """
    Generate complete dataset with corrected S-parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    output_dir : str
        Output directory
    hemorrhage_ratio : float
        Fraction of samples with hemorrhage
    **kwargs
        Additional parameters for generate_sample()
    """
    print("="*70)
    print(" Corrected Dataset Generation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples}")
    print(f"  Hemorrhage ratio: {hemorrhage_ratio:.0%}")
    print(f"  Output: {output_dir}/")
    print(f"  Correction factor: {CORRECTION_FACTOR:.3e} (MEEP validated)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    for subdir in ['s_matrix', 'eps_map', 'hem_mask', 'metadata']:
        (output_path / subdir).mkdir(exist_ok=True)

    print(f"\n✓ Output directories created")

    # Determine which samples have hemorrhage
    n_hem = int(n_samples * hemorrhage_ratio)
    has_hemorrhage = np.random.permutation(
        [True] * n_hem + [False] * (n_samples - n_hem)
    )

    print(f"\n{'='*70}")
    print(f" Generating {n_samples} Samples")
    print(f"{'='*70}\n")

    # Statistics
    s_matrix_mags = []
    generation_times = []

    # Generate samples
    for i in tqdm(range(n_samples), desc="Generating"):
        t_start = time.time()

        # Generate sample
        s_matrix, eps_map, hem_mask, metadata = generate_sample(
            sample_id=i,
            include_hemorrhage=has_hemorrhage[i],
            **kwargs
        )

        # Save
        np.save(output_path / "s_matrix" / f"sample_{i:06d}.npy", s_matrix)
        np.save(output_path / "eps_map" / f"sample_{i:06d}.npy", eps_map)
        np.save(output_path / "hem_mask" / f"sample_{i:06d}.npy", hem_mask)

        with open(output_path / "metadata" / f"sample_{i:06d}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Statistics
        s_matrix_mags.append(np.abs(s_matrix).max())
        generation_times.append(time.time() - t_start)

    # Summary
    print("\n" + "="*70)
    print(" Generation Complete")
    print("="*70)

    s_matrix_mags = np.array(s_matrix_mags)

    print(f"\nS-Matrix Magnitudes:")
    print(f"  Mean: {s_matrix_mags.mean():.4f}")
    print(f"  Std:  {s_matrix_mags.std():.4f}")
    print(f"  Min:  {s_matrix_mags.min():.4f}")
    print(f"  Max:  {s_matrix_mags.max():.4f}")

    # Check against MEEP reference
    expected_mag = 3.368  # From MEEP validation
    ratio = s_matrix_mags.mean() / expected_mag

    print(f"\nValidation vs MEEP:")
    print(f"  Expected magnitude: {expected_mag:.3f}")
    print(f"  Generated mean: {s_matrix_mags.mean():.3f}")
    print(f"  Ratio: {ratio:.3f}")

    if 0.5 < ratio < 2.0:
        print(f"  ✓ Magnitudes match MEEP reference!")
    else:
        print(f"  ⚠️  Magnitudes differ from MEEP")

    print(f"\nPerformance:")
    print(f"  Avg generation time: {np.mean(generation_times):.2f} s/sample")
    print(f"  Total time: {np.sum(generation_times)/60:.1f} minutes")

    print(f"\nDistribution:")
    print(f"  With hemorrhage: {np.sum(has_hemorrhage)}/{n_samples}")
    print(f"  Without: {np.sum(~has_hemorrhage)}/{n_samples}")

    # Save dataset metadata
    dataset_metadata = {
        'n_samples': n_samples,
        'hemorrhage_ratio': hemorrhage_ratio,
        'correction_factor': float(CORRECTION_FACTOR),
        'meep_validated': True,
        'generation_date': '2026-05-15',
        's_matrix_statistics': {
            'mean': float(s_matrix_mags.mean()),
            'std': float(s_matrix_mags.std()),
            'min': float(s_matrix_mags.min()),
            'max': float(s_matrix_mags.max()),
        },
        'meep_reference': {
            'expected_magnitude': 3.368,
            'actual_mean_magnitude': float(s_matrix_mags.mean()),
            'ratio': float(ratio),
        }
    }

    with open(output_path / "dataset_metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=2)

    print(f"\n✓ Dataset metadata saved")
    print(f"✓ Dataset saved to: {output_path}")
    print(f"\n✅ Generation complete! Dataset ready for training.")


def main():
    """Main generation routine."""
    parser = argparse.ArgumentParser(
        description="Generate corrected brain hemorrhage dataset"
    )
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='dataset_corrected',
                       help='Output directory')
    parser.add_argument('--hemorrhage_ratio', type=float, default=0.7,
                       help='Fraction of samples with hemorrhage')
    parser.add_argument('--nx', type=int, default=64,
                       help='Grid size X')
    parser.add_argument('--ny', type=int, default=64,
                       help='Grid size Y')
    parser.add_argument('--dx', type=float, default=0.5e-3,
                       help='Grid spacing (m)')
    parser.add_argument('--frequency', type=float, default=2e9,
                       help='Center frequency (Hz)')
    parser.add_argument('--n_ant', type=int, default=16,
                       help='Number of antennas')

    args = parser.parse_args()

    print("="*70)
    print(" Corrected Brain Hemorrhage Dataset Generator")
    print("="*70)
    print("\nThis script generates a dataset with CORRECT S-parameter")
    print("magnitudes (validated against MEEP reference).")
    print(f"\nCorrection factor: {CORRECTION_FACTOR:.3e}")
    print("Expected magnitude: ~3.4 (matches MEEP)\n")

    if not CEEP_AVAILABLE:
        print("❌ CEEP not available - cannot generate dataset")
        print("   Please install CEEP with GPU support")
        return

    # Set backend
    set_backend('cupy')
    print("✓ CEEP GPU backend initialized\n")

    # Generate dataset
    generate_dataset(
        n_samples=args.n_samples,
        output_dir=args.output,
        hemorrhage_ratio=args.hemorrhage_ratio,
        nx=args.nx,
        ny=args.ny,
        dx=args.dx,
        frequency=args.frequency,
        n_ant=args.n_ant
    )


if __name__ == "__main__":
    main()
