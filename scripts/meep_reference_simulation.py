#!/usr/bin/env python3
"""
MEEP Reference Simulation - Brain Hemorrhage Detection
=======================================================

Generates a reference simulation using classical MEEP to compare with CEEP.
Run with: conda activate pymeep && python3 scripts/meep_reference_simulation.py

Author: Shahzaib Ur Rehman
Date: 2026-05-15
"""

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import time
import json


def create_brain_geometry(sim_size=10.0, head_radius=4.0, hem_center=(1.0, 0.5), hem_radius=0.5):
    """
    Create 2D brain cross-section geometry for MEEP.

    Parameters
    ----------
    sim_size : float
        Simulation domain size (in wavelengths at center frequency)
    head_radius : float
        Head radius (in wavelengths)
    hem_center : tuple
        Hemorrhage center (x, y) in wavelengths
    hem_radius : float
        Hemorrhage radius in wavelengths

    Returns
    -------
    geometry : list
        MEEP geometry objects
    """
    # Tissue permittivities at 2 GHz (simplified)
    eps_air = 1.0
    eps_skin = 40.0
    eps_skull = 12.0
    eps_csf = 68.0
    eps_gray = 52.0
    eps_white = 38.0
    eps_blood = 61.0  # Hemorrhage

    geometry = []

    # Layer thicknesses (normalized)
    skin_thick = 0.15
    skull_thick = 0.35
    csf_thick = 0.10
    gray_thick = 0.50

    # Build concentric circles (outside to inside)
    # 1. Skin layer
    r_skin = head_radius
    geometry.append(
        mp.Cylinder(radius=r_skin, height=mp.inf,
                   center=mp.Vector3(0, 0, 0),
                   material=mp.Medium(epsilon=eps_skin))
    )

    # 2. Skull layer
    r_skull = head_radius - skin_thick
    geometry.append(
        mp.Cylinder(radius=r_skull, height=mp.inf,
                   center=mp.Vector3(0, 0, 0),
                   material=mp.Medium(epsilon=eps_skull))
    )

    # 3. CSF layer
    r_csf = r_skull - skull_thick
    geometry.append(
        mp.Cylinder(radius=r_csf, height=mp.inf,
                   center=mp.Vector3(0, 0, 0),
                   material=mp.Medium(epsilon=eps_csf))
    )

    # 4. Gray matter
    r_gray = r_csf - csf_thick
    geometry.append(
        mp.Cylinder(radius=r_gray, height=mp.inf,
                   center=mp.Vector3(0, 0, 0),
                   material=mp.Medium(epsilon=eps_gray))
    )

    # 5. White matter (core)
    r_white = r_gray - gray_thick
    geometry.append(
        mp.Cylinder(radius=r_white, height=mp.inf,
                   center=mp.Vector3(0, 0, 0),
                   material=mp.Medium(epsilon=eps_white))
    )

    # 6. Hemorrhage (if specified)
    if hem_center is not None:
        hx, hy = hem_center
        geometry.append(
            mp.Cylinder(radius=hem_radius, height=mp.inf,
                       center=mp.Vector3(hx, hy, 0),
                       material=mp.Medium(epsilon=eps_blood))
        )

    return geometry


def run_meep_simulation(
    n_ant=16,
    sim_size=10.0,
    resolution=20,
    frequency=1.0,
    hem_center=(1.0, 0.5),
    hem_radius=0.5,
    run_time=50
):
    """
    Run MEEP simulation for brain hemorrhage detection.

    Parameters
    ----------
    n_ant : int
        Number of antennas in circular array
    sim_size : float
        Simulation domain size (normalized units)
    resolution : int
        Grid points per wavelength
    frequency : float
        Center frequency (normalized, typically 1.0)
    hem_center : tuple or None
        Hemorrhage center (x, y), None for no hemorrhage
    hem_radius : float
        Hemorrhage radius
    run_time : float
        Simulation run time

    Returns
    -------
    s_matrix : ndarray (n_ant, n_ant, n_time)
        Scattering parameter matrix (time-domain)
    metadata : dict
        Simulation metadata
    """
    print("="*70)
    print(" MEEP Simulation - Brain Hemorrhage Detection")
    print("="*70)

    # Create cell
    cell_size = mp.Vector3(sim_size, sim_size, 0)

    # Antenna positions (circular array)
    array_radius = sim_size * 0.4  # 40% of domain
    angles = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
    ant_positions = []

    for angle in angles:
        x = array_radius * np.cos(angle)
        y = array_radius * np.sin(angle)
        ant_positions.append(mp.Vector3(x, y, 0))

    print(f"\nConfiguration:")
    print(f"  Cell size: {sim_size} x {sim_size} (normalized units)")
    print(f"  Resolution: {resolution} points/wavelength")
    print(f"  Frequency: {frequency} (normalized)")
    print(f"  Antennas: {n_ant} (circular array, radius={array_radius:.2f})")
    print(f"  Hemorrhage: {hem_center if hem_center else 'None'}")

    # Create geometry
    head_radius = sim_size * 0.35
    geometry = create_brain_geometry(sim_size, head_radius, hem_center, hem_radius)

    print(f"  Head radius: {head_radius:.2f}")
    print(f"  Geometry layers: {len(geometry)}")

    # S-parameter matrix storage
    s_matrix_data = {}  # Will store time-series for each TX-RX pair

    # Run simulation for each transmitter
    print(f"\n{'='*70}")
    print(f" Running {n_ant} Transmissions")
    print(f"{'='*70}\n")

    t_start_total = time.time()

    for tx_idx in range(n_ant):
        print(f"[TX {tx_idx+1}/{n_ant}] Simulating...")

        # Source at this antenna
        sources = [
            mp.Source(
                mp.GaussianSource(frequency=frequency, fwidth=0.2),
                component=mp.Ez,
                center=ant_positions[tx_idx]
            )
        ]

        # Create simulation
        sim = mp.Simulation(
            cell_size=cell_size,
            geometry=geometry,
            sources=sources,
            resolution=resolution,
            boundary_layers=[mp.PML(1.0)]
        )

        # Monitors at all RX antennas
        rx_monitors = []
        for rx_idx in range(n_ant):
            rx_monitors.append(ant_positions[rx_idx])

        # Storage for this TX
        fields_vs_time = {rx: [] for rx in range(n_ant)}

        def record_fields(sim_obj):
            """Callback to record fields at each time step."""
            for rx_idx, rx_pos in enumerate(rx_monitors):
                ez_val = sim_obj.get_field_point(mp.Ez, rx_pos)
                fields_vs_time[rx_idx].append(ez_val)

        # Run simulation with field recording
        t_start = time.time()

        sim.run(mp.at_every(0.1, record_fields), until=run_time)

        t_elapsed = time.time() - t_start

        # Store S-parameters for this TX
        for rx_idx in range(n_ant):
            s_matrix_data[(tx_idx, rx_idx)] = np.array(fields_vs_time[rx_idx])

        print(f"    ✓ TX {tx_idx+1} complete ({t_elapsed:.2f} s)")

        # Clean up
        sim = None

    t_total = time.time() - t_start_total

    print(f"\n✓ All transmissions complete!")
    print(f"  Total runtime: {t_total:.2f} s ({t_total/n_ant:.2f} s/TX)")

    # Convert to S-matrix array
    n_time = len(s_matrix_data[(0, 0)])
    s_matrix = np.zeros((n_ant, n_ant, n_time), dtype=np.complex128)

    for tx in range(n_ant):
        for rx in range(n_ant):
            s_matrix[tx, rx, :] = s_matrix_data[(tx, rx)]

    print(f"\n  S-matrix shape: {s_matrix.shape}")
    print(f"  S-matrix magnitude: mean={np.abs(s_matrix).mean():.3e}, max={np.abs(s_matrix).max():.3e}")

    # Metadata
    metadata = {
        'n_antennas': n_ant,
        'sim_size': sim_size,
        'resolution': resolution,
        'frequency': frequency,
        'has_hemorrhage': hem_center is not None,
        'hem_center': hem_center,
        'hem_radius': hem_radius,
        'runtime_seconds': t_total,
        'n_timesteps': n_time
    }

    return s_matrix, metadata, ant_positions


def visualize_results(s_matrix, metadata, output_path="meep_reference_result.png"):
    """Visualize MEEP simulation results."""
    n_ant = s_matrix.shape[0]

    fig = plt.figure(figsize=(16, 10))

    # 1. S-parameter magnitude heatmap
    ax1 = plt.subplot(2, 3, 1)
    s_mag = np.sum(np.abs(s_matrix), axis=2)  # Integrate over time
    im1 = ax1.imshow(s_mag, cmap='hot', aspect='auto')
    ax1.set_title('S-Parameter Magnitude (Integrated)')
    ax1.set_xlabel('RX Antenna')
    ax1.set_ylabel('TX Antenna')
    plt.colorbar(im1, ax=ax1, label='Magnitude')

    # 2. Diagonal S-parameters (reflection)
    ax2 = plt.subplot(2, 3, 2)
    s_diag = np.array([s_mag[i, i] for i in range(n_ant)])
    ax2.plot(s_diag, 'o-')
    ax2.set_title('Reflection Coefficients (S_ii)')
    ax2.set_xlabel('Antenna Index')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True, alpha=0.3)

    # 3. Off-diagonal S-parameters (transmission)
    ax3 = plt.subplot(2, 3, 3)
    s_offdiag = []
    for i in range(n_ant):
        for j in range(n_ant):
            if i != j:
                s_offdiag.append(s_mag[i, j])
    ax3.hist(s_offdiag, bins=30, edgecolor='black')
    ax3.set_title('Transmission Coefficient Distribution')
    ax3.set_xlabel('Magnitude')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Time-domain signal (one TX-RX pair)
    ax4 = plt.subplot(2, 3, 4)
    tx, rx = 0, n_ant // 2  # Opposite antennas
    signal = s_matrix[tx, rx, :]
    time_axis = np.arange(len(signal)) * 0.1  # dt=0.1 in normalized units
    ax4.plot(time_axis, np.real(signal), label='Real', alpha=0.7)
    ax4.plot(time_axis, np.imag(signal), label='Imag', alpha=0.7)
    ax4.plot(time_axis, np.abs(signal), label='|S|', linewidth=2, color='black')
    ax4.set_title(f'Time-Domain Signal (TX={tx}, RX={rx})')
    ax4.set_xlabel('Time (normalized)')
    ax4.set_ylabel('Field Amplitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Frequency spectrum
    ax5 = plt.subplot(2, 3, 5)
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=0.1)
    ax5.plot(freq[:len(freq)//2], np.abs(fft[:len(fft)//2]))
    ax5.set_title('Frequency Spectrum')
    ax5.set_xlabel('Frequency (normalized)')
    ax5.set_ylabel('Magnitude')
    ax5.grid(True, alpha=0.3)

    # 6. Metadata
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    info_text = "MEEP Reference Simulation\n\n"
    info_text += f"Antennas: {metadata['n_antennas']}\n"
    info_text += f"Grid size: {metadata['sim_size']}\n"
    info_text += f"Resolution: {metadata['resolution']} pts/λ\n"
    info_text += f"Frequency: {metadata['frequency']}\n\n"
    info_text += f"Has hemorrhage: {metadata['has_hemorrhage']}\n"
    if metadata['has_hemorrhage']:
        info_text += f"Hem center: {metadata['hem_center']}\n"
        info_text += f"Hem radius: {metadata['hem_radius']}\n"
    info_text += f"\nS-matrix shape: {s_matrix.shape}\n"
    info_text += f"Max magnitude: {np.abs(s_matrix).max():.3e}\n"
    info_text += f"Mean magnitude: {np.abs(s_matrix).mean():.3e}\n"
    info_text += f"\nRuntime: {metadata['runtime_seconds']:.1f} s\n"

    ax6.text(0.1, 0.5, info_text, transform=ax6.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")


def main():
    """Main simulation routine."""
    print("="*70)
    print(" MEEP Reference Simulation")
    print("="*70)
    print("\nThis will generate a reference simulation using classical MEEP")
    print("for comparison with the GPU dataset from CEEP.\n")

    # Parameters
    n_ant = 16
    sim_size = 10.0  # Normalized units
    resolution = 20  # Points per wavelength
    frequency = 1.0  # Normalized
    hem_center = (1.0, 0.5)  # Normalized
    hem_radius = 0.5
    run_time = 50  # Normalized time

    # Run simulation
    s_matrix, metadata, ant_pos = run_meep_simulation(
        n_ant=n_ant,
        sim_size=sim_size,
        resolution=resolution,
        frequency=frequency,
        hem_center=hem_center,
        hem_radius=hem_radius,
        run_time=run_time
    )

    # Save results
    print("\n" + "="*70)
    print(" Saving Results")
    print("="*70)

    np.save("meep_reference_s_matrix.npy", s_matrix)
    print("  ✓ Saved S-matrix to meep_reference_s_matrix.npy")

    with open("meep_reference_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ Saved metadata to meep_reference_metadata.json")

    # Visualize
    visualize_results(s_matrix, metadata)

    # Comparison with GPU dataset
    print("\n" + "="*70)
    print(" Comparison with GPU Dataset")
    print("="*70)

    try:
        ceep_s = np.load("dataset_gpu/s_matrix/sample_000000.npy")
        ceep_mag = np.abs(ceep_s).max()
        meep_mag = np.abs(s_matrix).max()

        print(f"\nS-Matrix Maximum Magnitude:")
        print(f"  MEEP (reference): {meep_mag:.3e}")
        print(f"  CEEP (GPU):       {ceep_mag:.3e}")
        print(f"  Ratio (CEEP/MEEP): {ceep_mag/meep_mag:.2e}")

        if ceep_mag / meep_mag < 1e-6:
            print(f"\n❌ CEEP values are {meep_mag/ceep_mag:.0e}x too small!")
            print("   → CEEP dataset needs to be regenerated")
        else:
            print("\n✓ CEEP and MEEP magnitudes are comparable")

    except FileNotFoundError:
        print("\n⚠️  GPU dataset not found - cannot compare")

    print("\n✓ Reference simulation complete!")


if __name__ == "__main__":
    main()
