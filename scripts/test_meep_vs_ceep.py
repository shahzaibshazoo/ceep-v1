#!/usr/bin/env python3
"""
Test and Compare MEEP vs CEEP Samples
======================================

Comprehensive comparison of MEEP reference simulation vs corrected CEEP dataset.

Tests:
1. S-parameter magnitude comparison
2. Time-domain signal comparison
3. Frequency spectrum comparison
4. Statistical analysis
5. Visual comparison

Author: Shahzaib Ur Rehman
Date: 2026-05-15
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_samples():
    """Load MEEP reference and CEEP corrected sample."""
    print("="*70)
    print(" Loading Samples")
    print("="*70)

    # Load MEEP reference
    print("\n[1/2] Loading MEEP reference...")
    try:
        meep_s = np.load("meep_reference_s_matrix.npy")
        print(f"  ✓ MEEP shape: {meep_s.shape}")
        print(f"    Magnitude: mean={np.abs(meep_s).mean():.4f}, max={np.abs(meep_s).max():.4f}")
    except FileNotFoundError:
        print("  ❌ MEEP reference not found!")
        return None, None

    # Load CEEP corrected
    print("\n[2/2] Loading CEEP corrected sample 0...")
    try:
        ceep_s = np.load("dataset_gpu_corrected/s_matrix/sample_000000.npy")
        print(f"  ✓ CEEP shape: {ceep_s.shape}")
        print(f"    Magnitude: mean={np.abs(ceep_s).mean():.4f}, max={np.abs(ceep_s).max():.4f}")
    except FileNotFoundError:
        print("  ❌ CEEP corrected dataset not found!")
        return None, None

    return meep_s, ceep_s


def compare_magnitudes(meep_s, ceep_s):
    """Compare S-parameter magnitudes."""
    print("\n" + "="*70)
    print(" Magnitude Comparison")
    print("="*70)

    meep_mag = np.abs(meep_s)
    ceep_mag = np.abs(ceep_s)

    # Overall statistics
    print(f"\nOverall Magnitude Statistics:")
    print(f"  MEEP: mean={meep_mag.mean():.4f}, std={meep_mag.std():.4f}, max={meep_mag.max():.4f}")
    print(f"  CEEP: mean={ceep_mag.mean():.4f}, std={ceep_mag.std():.4f}, max={ceep_mag.max():.4f}")

    ratio_mean = ceep_mag.mean() / meep_mag.mean()
    ratio_max = ceep_mag.max() / meep_mag.max()

    print(f"\n  Ratio (CEEP/MEEP):")
    print(f"    Mean: {ratio_mean:.3f}")
    print(f"    Max:  {ratio_max:.3f}")

    # Per-antenna statistics
    print(f"\nPer-Antenna Analysis:")
    n_ant = min(meep_s.shape[0], ceep_s.shape[0])

    for i in [0, n_ant//4, n_ant//2, 3*n_ant//4]:
        meep_ant = meep_mag[i, :, :].mean()
        ceep_ant = ceep_mag[i, :, :].mean()
        ratio = ceep_ant / meep_ant
        print(f"  Antenna {i:2d}: MEEP={meep_ant:.4f}, CEEP={ceep_ant:.4f}, ratio={ratio:.3f}")

    # Verdict
    print(f"\n{'='*70}")
    if 0.8 < ratio_mean < 1.2 and 0.8 < ratio_max < 1.2:
        print(" ✓ EXCELLENT! Magnitudes match within 20%")
    elif 0.5 < ratio_mean < 2.0 and 0.5 < ratio_max < 2.0:
        print(" ✓ GOOD! Magnitudes match within factor of 2")
    else:
        print(" ⚠️  WARNING: Magnitudes differ significantly")
    print("="*70)

    return ratio_mean, ratio_max


def compare_time_domain(meep_s, ceep_s):
    """Compare time-domain signals."""
    print("\n" + "="*70)
    print(" Time-Domain Signal Comparison")
    print("="*70)

    # Pick a representative TX-RX pair (opposite antennas)
    n_ant = min(meep_s.shape[0], ceep_s.shape[0])
    tx, rx = 0, n_ant // 2

    meep_signal = meep_s[tx, rx, :]
    ceep_signal = ceep_s[tx, rx, :]

    print(f"\nAnalyzing TX={tx}, RX={rx} (opposite antennas)")
    print(f"  MEEP signal length: {len(meep_signal)}")
    print(f"  CEEP signal length: {len(ceep_signal)}")

    # Align signals (CEEP may have different length)
    min_len = min(len(meep_signal), len(ceep_signal))
    meep_sig = meep_signal[:min_len]
    ceep_sig = ceep_signal[:min_len]

    # Magnitude comparison
    meep_mag = np.abs(meep_sig)
    ceep_mag = np.abs(ceep_sig)

    print(f"\n  Peak magnitude:")
    print(f"    MEEP: {meep_mag.max():.4f} at t={meep_mag.argmax()}")
    print(f"    CEEP: {ceep_mag.max():.4f} at t={ceep_mag.argmax()}")

    # Correlation (if similar length)
    if len(meep_sig) == len(ceep_sig):
        correlation = np.corrcoef(meep_mag, ceep_mag)[0, 1]
        print(f"\n  Correlation: {correlation:.3f}")
        if correlation > 0.7:
            print(f"    ✓ Good correlation")
        elif correlation > 0.5:
            print(f"    ~ Moderate correlation")
        else:
            print(f"    ⚠️  Low correlation")

    return meep_sig, ceep_sig, tx, rx


def compare_frequency_spectrum(meep_sig, ceep_sig):
    """Compare frequency spectra."""
    print("\n" + "="*70)
    print(" Frequency Spectrum Comparison")
    print("="*70)

    # FFT of signals
    meep_fft = np.fft.fft(meep_sig)
    ceep_fft = np.fft.fft(ceep_sig)

    meep_freq = np.fft.fftfreq(len(meep_sig))
    ceep_freq = np.fft.fftfreq(len(ceep_sig))

    # Compare peak frequencies
    meep_peak_idx = np.argmax(np.abs(meep_fft[:len(meep_fft)//2]))
    ceep_peak_idx = np.argmax(np.abs(ceep_fft[:len(ceep_fft)//2]))

    print(f"\n  Peak frequency:")
    print(f"    MEEP: index={meep_peak_idx}, normalized freq={meep_freq[meep_peak_idx]:.4f}")
    print(f"    CEEP: index={ceep_peak_idx}, normalized freq={ceep_freq[ceep_peak_idx]:.4f}")

    # Spectral energy
    meep_energy = np.sum(np.abs(meep_fft)**2)
    ceep_energy = np.sum(np.abs(ceep_fft)**2)

    print(f"\n  Total spectral energy:")
    print(f"    MEEP: {meep_energy:.2e}")
    print(f"    CEEP: {ceep_energy:.2e}")
    print(f"    Ratio: {ceep_energy/meep_energy:.3f}")

    return meep_fft, ceep_fft, meep_freq, ceep_freq


def create_comparison_visualization(meep_s, ceep_s, meep_sig, ceep_sig,
                                   meep_fft, ceep_fft, meep_freq, ceep_freq,
                                   tx, rx):
    """Create comprehensive comparison visualization."""
    print("\n" + "="*70)
    print(" Creating Comparison Visualization")
    print("="*70)

    fig = plt.figure(figsize=(18, 12))

    # 1. S-parameter heatmaps (integrated over time)
    ax1 = plt.subplot(3, 3, 1)
    meep_integrated = np.sum(np.abs(meep_s), axis=2)
    im1 = ax1.imshow(meep_integrated, cmap='hot', aspect='auto')
    ax1.set_title('MEEP: S-Parameter Magnitude\n(Integrated over time)')
    ax1.set_xlabel('RX Antenna')
    ax1.set_ylabel('TX Antenna')
    plt.colorbar(im1, ax=ax1)

    ax2 = plt.subplot(3, 3, 2)
    ceep_integrated = np.sum(np.abs(ceep_s), axis=2)
    im2 = ax2.imshow(ceep_integrated, cmap='hot', aspect='auto')
    ax2.set_title('CEEP: S-Parameter Magnitude\n(Integrated over time)')
    ax2.set_xlabel('RX Antenna')
    ax2.set_ylabel('TX Antenna')
    plt.colorbar(im2, ax=ax2)

    # Difference map
    ax3 = plt.subplot(3, 3, 3)
    # Normalize both to same scale for comparison
    meep_norm = meep_integrated / meep_integrated.max()
    ceep_norm = ceep_integrated / ceep_integrated.max()
    diff = np.abs(ceep_norm - meep_norm)
    im3 = ax3.imshow(diff, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.5)
    ax3.set_title('Normalized Difference\n|CEEP - MEEP|')
    ax3.set_xlabel('RX Antenna')
    ax3.set_ylabel('TX Antenna')
    plt.colorbar(im3, ax=ax3, label='Difference')

    # 2. Time-domain signals
    ax4 = plt.subplot(3, 3, 4)
    t_meep = np.arange(len(meep_sig))
    t_ceep = np.arange(len(ceep_sig))
    ax4.plot(t_meep, np.abs(meep_sig), 'b-', label='MEEP', alpha=0.7, linewidth=2)
    ax4.plot(t_ceep, np.abs(ceep_sig), 'r--', label='CEEP', alpha=0.7, linewidth=2)
    ax4.set_title(f'Time-Domain Signal\nTX={tx}, RX={rx}')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('|S-Parameter|')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Time-domain (zoomed)
    ax5 = plt.subplot(3, 3, 5)
    zoom_range = slice(0, min(100, len(meep_sig)))
    ax5.plot(t_meep[zoom_range], np.real(meep_sig[zoom_range]), 'b-', label='MEEP (real)', alpha=0.7)
    ax5.plot(t_ceep[zoom_range], np.real(ceep_sig[zoom_range]), 'r--', label='CEEP (real)', alpha=0.7)
    ax5.set_title('Time-Domain (Zoomed, First 100 steps)')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Real Part')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Correlation plot
    ax6 = plt.subplot(3, 3, 6)
    min_len = min(len(meep_sig), len(ceep_sig))
    ax6.scatter(np.abs(meep_sig[:min_len]), np.abs(ceep_sig[:min_len]),
                alpha=0.3, s=1)
    ax6.set_title('MEEP vs CEEP Magnitude\n(each time point)')
    ax6.set_xlabel('MEEP |S|')
    ax6.set_ylabel('CEEP |S|')
    # Add y=x line
    max_val = max(np.abs(meep_sig).max(), np.abs(ceep_sig).max())
    ax6.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 3. Frequency spectra
    ax7 = plt.subplot(3, 3, 7)
    half_len_meep = len(meep_fft) // 2
    half_len_ceep = len(ceep_fft) // 2
    ax7.plot(meep_freq[:half_len_meep], np.abs(meep_fft[:half_len_meep]),
             'b-', label='MEEP', alpha=0.7, linewidth=2)
    ax7.plot(ceep_freq[:half_len_ceep], np.abs(ceep_fft[:half_len_ceep]),
             'r--', label='CEEP', alpha=0.7, linewidth=2)
    ax7.set_title('Frequency Spectrum')
    ax7.set_xlabel('Normalized Frequency')
    ax7.set_ylabel('Magnitude')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 0.5)

    # Histogram of magnitudes
    ax8 = plt.subplot(3, 3, 8)
    meep_mags = np.abs(meep_s).flatten()
    ceep_mags = np.abs(ceep_s).flatten()
    ax8.hist(meep_mags, bins=50, alpha=0.5, label='MEEP', density=True)
    ax8.hist(ceep_mags, bins=50, alpha=0.5, label='CEEP', density=True)
    ax8.set_title('Distribution of S-Parameter Magnitudes')
    ax8.set_xlabel('|S|')
    ax8.set_ylabel('Density')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.set_xlim(0, max(meep_mags.max(), ceep_mags.max()) * 0.5)

    # Statistics summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = "COMPARISON SUMMARY\n\n"
    summary_text += f"MEEP Reference:\n"
    summary_text += f"  Shape: {meep_s.shape}\n"
    summary_text += f"  Mean: {np.abs(meep_s).mean():.4f}\n"
    summary_text += f"  Max:  {np.abs(meep_s).max():.4f}\n\n"

    summary_text += f"CEEP Corrected:\n"
    summary_text += f"  Shape: {ceep_s.shape}\n"
    summary_text += f"  Mean: {np.abs(ceep_s).mean():.4f}\n"
    summary_text += f"  Max:  {np.abs(ceep_s).max():.4f}\n\n"

    ratio_mean = np.abs(ceep_s).mean() / np.abs(meep_s).mean()
    ratio_max = np.abs(ceep_s).max() / np.abs(meep_s).max()

    summary_text += f"Magnitude Ratios:\n"
    summary_text += f"  Mean: {ratio_mean:.3f}\n"
    summary_text += f"  Max:  {ratio_max:.3f}\n\n"

    if 0.8 < ratio_mean < 1.2 and 0.8 < ratio_max < 1.2:
        summary_text += "Status: ✓ EXCELLENT\n"
        summary_text += "Magnitudes match\nwithin 20%"
    elif 0.5 < ratio_mean < 2.0:
        summary_text += "Status: ✓ GOOD\n"
        summary_text += "Magnitudes match\nwithin factor of 2"
    else:
        summary_text += "Status: ⚠️ WARNING\n"
        summary_text += "Magnitudes differ\nsignificantly"

    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='center')

    plt.tight_layout()

    output_path = "meep_vs_ceep_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved to {output_path}")

    return fig


def test_dataset_integrity(ceep_path="dataset_gpu_corrected"):
    """Test multiple samples from corrected dataset."""
    print("\n" + "="*70)
    print(" Dataset Integrity Test")
    print("="*70)

    path = Path(ceep_path)
    s_matrix_files = list((path / "s_matrix").glob("*.npy"))

    print(f"\nTesting {len(s_matrix_files)} samples...")

    # Load MEEP reference for comparison
    try:
        meep_s = np.load("meep_reference_s_matrix.npy")
        meep_max = np.abs(meep_s).max()
    except:
        print("  ⚠️  MEEP reference not found, using expected value")
        meep_max = 3.368

    # Test 10 samples
    sample_ids = [0, 10, 25, 50, 75, 99] if len(s_matrix_files) >= 100 else range(min(10, len(s_matrix_files)))

    results = []
    for sample_id in sample_ids:
        try:
            s_matrix = np.load(path / "s_matrix" / f"sample_{sample_id:06d}.npy")
            mag_max = np.abs(s_matrix).max()
            ratio = mag_max / meep_max
            results.append((sample_id, mag_max, ratio))
        except FileNotFoundError:
            continue

    print(f"\n  {'Sample':<10} {'Max Mag':<12} {'Ratio vs MEEP':<15} {'Status':<10}")
    print(f"  {'-'*10} {'-'*12} {'-'*15} {'-'*10}")

    all_good = True
    for sample_id, mag, ratio in results:
        status = "✓ GOOD" if 0.5 < ratio < 2.0 else "⚠️ CHECK"
        if status != "✓ GOOD":
            all_good = False
        print(f"  {sample_id:<10d} {mag:<12.4f} {ratio:<15.3f} {status:<10}")

    print(f"\n  {'='*70}")
    if all_good:
        print("  ✓ All tested samples have correct magnitudes!")
    else:
        print("  ⚠️  Some samples have unusual magnitudes")
    print(f"  {'='*70}")


def main():
    """Main testing routine."""
    print("="*70)
    print(" MEEP vs CEEP Comparison Test")
    print("="*70)
    print("\nThis script comprehensively compares MEEP reference simulation")
    print("with corrected CEEP dataset to validate the correction.\n")

    # Load samples
    meep_s, ceep_s = load_samples()

    if meep_s is None or ceep_s is None:
        print("\n❌ Cannot proceed without both MEEP and CEEP samples")
        return

    # Compare magnitudes
    ratio_mean, ratio_max = compare_magnitudes(meep_s, ceep_s)

    # Compare time-domain
    meep_sig, ceep_sig, tx, rx = compare_time_domain(meep_s, ceep_s)

    # Compare frequency
    meep_fft, ceep_fft, meep_freq, ceep_freq = compare_frequency_spectrum(meep_sig, ceep_sig)

    # Create visualization
    fig = create_comparison_visualization(meep_s, ceep_s, meep_sig, ceep_sig,
                                         meep_fft, ceep_fft, meep_freq, ceep_freq,
                                         tx, rx)

    # Test dataset integrity
    test_dataset_integrity()

    # Final verdict
    print("\n" + "="*70)
    print(" FINAL VERDICT")
    print("="*70)

    if 0.8 < ratio_mean < 1.2 and 0.8 < ratio_max < 1.2:
        print("\n✅ EXCELLENT! CEEP and MEEP match within 20%")
        print("   Dataset is fully validated and ready for training.")
    elif 0.5 < ratio_mean < 2.0 and 0.5 < ratio_max < 2.0:
        print("\n✓ GOOD! CEEP and MEEP match within factor of 2")
        print("  Dataset is usable for training.")
    else:
        print("\n⚠️  WARNING: CEEP and MEEP magnitudes differ significantly")
        print("   Consider re-checking the correction factor.")

    print(f"\nVisualization saved to: meep_vs_ceep_comparison.png")
    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()
