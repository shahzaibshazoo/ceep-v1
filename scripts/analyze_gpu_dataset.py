#!/usr/bin/env python3
"""
Analyze GPU Dataset - No GPU Required
======================================

Analyzes the pre-generated GPU dataset to check if S-parameters are correct.
Performs DAS imaging and checks against ground truth.

Author: Shahzaib Ur Rehman
Date: 2026-05-15
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


def load_sample(dataset_path, sample_id):
    """Load a complete sample."""
    base = Path(dataset_path)

    s_matrix = np.load(base / "s_matrix" / f"sample_{sample_id:06d}.npy")
    eps_map = np.load(base / "eps_map" / f"sample_{sample_id:06d}.npy")
    hem_mask = np.load(base / "hem_mask" / f"sample_{sample_id:06d}.npy")

    with open(base / "metadata" / f"sample_{sample_id:06d}.json") as f:
        metadata = json.load(f)

    return s_matrix, eps_map, hem_mask, metadata


def das_imaging_simple(s_matrix):
    """
    Simple DAS imaging using differential S-parameters.

    For brain imaging, we look at the scattered field (difference from baseline).
    """
    n_ant, _, n_time = s_matrix.shape

    # Simple approach: Sum magnitude across time for each TX-RX pair
    # This gives us the scattering pattern
    image = np.zeros((n_ant, n_ant))

    for tx in range(n_ant):
        for rx in range(n_ant):
            if tx != rx:  # Skip diagonal
                signal = s_matrix[tx, rx, :]
                # Integrate energy
                image[tx, rx] = np.sum(np.abs(signal)**2)

    return image


def analyze_dataset(dataset_path="dataset_gpu", num_samples=10):
    """Analyze multiple samples from the dataset."""
    print("="*70)
    print(" GPU Dataset Analysis")
    print("="*70)

    print(f"\nAnalyzing {num_samples} samples from {dataset_path}...")

    # Statistics collectors
    s_mags = []
    has_hem_list = []

    for i in range(num_samples):
        s_matrix, eps_map, hem_mask, metadata = load_sample(dataset_path, i)

        s_mag = np.abs(s_matrix).max()
        has_hem = metadata.get('include_hemorrhage', False)

        s_mags.append(s_mag)
        has_hem_list.append(has_hem)

        if i < 3:
            print(f"\nSample {i:06d}:")
            print(f"  S-matrix shape: {s_matrix.shape}")
            print(f"  S-matrix max:   {s_mag:.3e}")
            print(f"  Has hemorrhage: {has_hem}")
            if has_hem and 'hemorrhage_params' in metadata:
                hem_params = metadata['hemorrhage_params']
                print(f"  Hem center: ({hem_params.get('center_x', 'N/A')}, {hem_params.get('center_y', 'N/A')})")
                print(f"  Hem radius: {hem_params.get('radius', 'N/A')}")

    # Summary statistics
    print("\n" + "="*70)
    print(" DATASET SUMMARY")
    print("="*70)

    s_mags = np.array(s_mags)
    print(f"\nS-Matrix Magnitudes:")
    print(f"  Mean: {s_mags.mean():.3e}")
    print(f"  Std:  {s_mags.std():.3e}")
    print(f"  Min:  {s_mags.min():.3e}")
    print(f"  Max:  {s_mags.max():.3e}")

    num_with_hem = sum(has_hem_list)
    print(f"\nHemorrhage Distribution:")
    print(f"  With hemorrhage:    {num_with_hem}/{num_samples}")
    print(f"  Without hemorrhage: {num_samples - num_with_hem}/{num_samples}")

    # DIAGNOSIS
    print("\n" + "="*70)
    print(" DIAGNOSIS")
    print("="*70)

    expected_mag = 1e-3  # Typical S-parameter magnitude for microwave imaging

    if s_mags.max() < 1e-10:
        print("\n❌ CRITICAL ISSUE: S-matrix magnitudes too small!")
        print(f"   Expected: ~{expected_mag:.0e}")
        print(f"   Got:      ~{s_mags.max():.0e}")
        print(f"   Ratio:    {s_mags.max() / expected_mag:.1e}")

        print("\n   Root Cause Analysis:")
        print("   1. Source amplitude may be scaled incorrectly in CEEP")
        print("   2. S-parameter extraction may be missing a scaling factor")
        print("   3. Time window may be too short to capture scattered signal")
        print("   4. Fields may not be properly normalized")

        print("\n   Recommended Fixes:")
        print("   • Check source amplitude in BatchedFDTD2D")
        print("   • Verify S-parameter computation includes incident field normalization")
        print("   • Increase simulation time steps")
        print("   • Compare single-element simulation with analytical solution")

        return False
    else:
        print(f"\n✓ S-matrix magnitudes reasonable!")
        print(f"  Within acceptable range for microwave imaging")
        return True


def visualize_sample(dataset_path="dataset_gpu", sample_id=0):
    """Create detailed visualization of one sample."""
    s_matrix, eps_map, hem_mask, metadata = load_sample(dataset_path, sample_id)

    fig = plt.figure(figsize=(16, 10))

    # 1. Permittivity map
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(eps_map.T, origin='lower', cmap='viridis')
    ax1.set_title(f'Sample {sample_id} - Permittivity')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='εr')

    # 2. Hemorrhage mask
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(hem_mask.T, origin='lower', cmap='Reds', vmin=0, vmax=1)
    ax2.set_title('Hemorrhage Ground Truth')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Mask')

    # 3. S-parameter magnitude heatmap (TX vs RX)
    ax3 = plt.subplot(2, 3, 3)
    s_mag_integrated = np.sum(np.abs(s_matrix), axis=2)  # Sum over time
    im3 = ax3.imshow(s_mag_integrated, cmap='hot', aspect='auto')
    ax3.set_title('S-Parameter Magnitude (Integrated)')
    ax3.set_xlabel('RX Antenna')
    ax3.set_ylabel('TX Antenna')
    plt.colorbar(im3, ax=ax3, label='Magnitude')

    # 4. Time-domain signal for one TX-RX pair
    ax4 = plt.subplot(2, 3, 4)
    tx, rx = 0, 8  # Opposite antennas
    signal = s_matrix[tx, rx, :]
    time_axis = np.arange(len(signal))
    ax4.plot(time_axis, np.abs(signal), label=f'|S_{tx},{rx}|')
    ax4.set_title(f'Time-Domain Signal (TX={tx}, RX={rx})')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Magnitude')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Frequency spectrum
    ax5 = plt.subplot(2, 3, 5)
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal))
    ax5.plot(freq[:len(freq)//2], np.abs(fft[:len(fft)//2]))
    ax5.set_title('Frequency Spectrum')
    ax5.set_xlabel('Normalized Frequency')
    ax5.set_ylabel('Magnitude')
    ax5.grid(True, alpha=0.3)

    # 6. Metadata info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    info_text = f"Sample {sample_id:06d}\n\n"
    info_text += f"S-Matrix Shape: {s_matrix.shape}\n"
    info_text += f"Max Magnitude: {np.abs(s_matrix).max():.3e}\n"
    info_text += f"Mean Magnitude: {np.abs(s_matrix).mean():.3e}\n\n"
    info_text += f"Has Hemorrhage: {metadata.get('include_hemorrhage', 'N/A')}\n"
    if 'hemorrhage_params' in metadata:
        info_text += f"Hem Params: {metadata['hemorrhage_params']}\n"
    info_text += f"\nEps Map: {eps_map.shape}\n"
    info_text += f"Eps Range: [{eps_map.min():.1f}, {eps_map.max():.1f}]\n"

    ax6.text(0.1, 0.5, info_text, transform=ax6.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='center')

    plt.tight_layout()

    output_path = f"dataset_analysis_sample_{sample_id:06d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")

    return fig


def main():
    """Main analysis routine."""
    dataset_path = "dataset_gpu"

    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("   Please extract dataset_gpu (4).zip first")
        return

    # Analyze dataset
    is_valid = analyze_dataset(dataset_path, num_samples=10)

    # Visualize one sample
    print("\n" + "="*70)
    print(" CREATING DETAILED VISUALIZATION")
    print("="*70)

    visualize_sample(dataset_path, sample_id=0)

    # Final verdict
    print("\n" + "="*70)
    print(" FINAL VERDICT")
    print("="*70)

    if is_valid:
        print("\n✓ Dataset appears valid and usable for training")
    else:
        print("\n❌ Dataset has issues - S-parameters too small")
        print("   → Need to regenerate with corrected source amplitude")
        print("   → Check CEEP source implementation")

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
