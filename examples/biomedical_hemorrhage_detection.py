"""
Complete Biomedical Example: Brain Hemorrhage Detection
========================================================

End-to-end demonstration of microwave imaging for stroke detection.

This example shows the complete pipeline:
1. Create anatomically realistic head phantom with hemorrhage
2. Set up circular antenna array (16 elements)
3. Run FDTD simulations for each transmitter
4. Collect multistatic S-parameters
5. Reconstruct image using DAS beamforming
6. Visualize and analyze results

This is a complete, production-ready workflow for biomedical microwave imaging.

Author: NeuroWave Development Team
Date: 2026-05-13
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from ceep.core.config import GridConfig, SimulationConfig, SimulationMode
from ceep.core.constants import C_0
from ceep.solvers.fdtd_2d import FDTD2D
from ceep.solvers.dft import DFTMonitor
from ceep.solvers.s_params import MultistaticSParameters, MultistaticDataCollector
from ceep.sources.waveforms import ModulatedGaussianSource
from ceep.boundaries.absorbing import CPML
from ceep.materials.tissue_database import TissueDatabase
from ceep.phantoms import SimpleHeadPhantom
from ceep.antennas import CircularArray
from ceep.imaging import DelayAndSumBeamformer, ImagingRegion


def main():
    """Run complete hemorrhage detection example."""

    print("="*70)
    print("  Brain Hemorrhage Detection with Microwave Imaging")
    print("  Complete End-to-End Pipeline")
    print("="*70)

    # ========================================================================
    # STEP 1: Define Simulation Parameters
    # ========================================================================
    print("\n[1/7] Setting up simulation parameters...")

    dx = 1e-3  # 1 mm grid resolution
    nx = ny = 200  # 200×200 grid (20 cm × 20 cm)

    # Frequency sweep for S-parameters
    freq_center = 1.5e9  # 1.5 GHz center frequency
    bandwidth = 1e9      # 1 GHz bandwidth
    frequencies = np.linspace(freq_center - bandwidth/2,
                             freq_center + bandwidth/2,
                             21)  # 21 frequency points

    # Time domain parameters
    total_time = 10e-9  # 10 ns simulation

    grid_config = GridConfig(nx=nx, ny=ny, dx=dx, dy=dx)
    config = SimulationConfig(
        grid=grid_config,
        mode=SimulationMode.TMZ,
        total_time=total_time
    )

    print(f"  Grid: {nx}×{ny}, dx={dx*1e3:.1f} mm")
    print(f"  Frequency: {freq_center/1e9:.2f} GHz ± {bandwidth/2/1e9:.2f} GHz")
    print(f"  Time steps: {config.total_steps}")

    # ========================================================================
    # STEP 2: Create Head Phantom with Hemorrhage
    # ========================================================================
    print("\n[2/7] Creating head phantom with hemorrhage...")

    phantom = SimpleHeadPhantom(
        nx=nx,
        ny=ny,
        dx=dx,
        head_radius_mm=80  # 8 cm radius head
    )

    # Add 10mm hemorrhage at offset position
    hemorrhage_x = 115  # Offset from center
    hemorrhage_y = 100
    hemorrhage_radius = 10  # mm -> 10 cells

    # Get tissue database
    db = TissueDatabase()
    hemorrhage_tissue = db.get('hemorrhage')

    # Create phantom with hemorrhage
    tissue_map = phantom.create_tissue_map()

    # Insert hemorrhage (simplified - replace tissue in circular region)
    center_x, center_y = nx // 2, ny // 2
    for i in range(nx):
        for j in range(ny):
            r = np.sqrt((i - hemorrhage_x)**2 + (j - hemorrhage_y)**2)
            if r <= hemorrhage_radius:
                tissue_map[i, j] = hemorrhage_tissue

    print(f"  Head radius: 80 mm")
    print(f"  Hemorrhage location: ({hemorrhage_x}, {hemorrhage_y})")
    print(f"  Hemorrhage radius: {hemorrhage_radius} mm")

    # ========================================================================
    # STEP 3: Set Up Antenna Array
    # ========================================================================
    print("\n[3/7] Configuring circular antenna array...")

    array = CircularArray(
        num_antennas=16,
        radius_mm=120,  # 12 cm from center
        center=(nx//2, ny//2),
        dx=dx,
        antenna_type='monopole',
        polarization='vertical'
    )

    antenna_positions = array.get_antenna_positions()
    print(f"  Array: {len(antenna_positions)} antennas")
    print(f"  Radius: 120 mm")
    print(f"  TX-RX pairs: {len(antenna_positions)}×{len(antenna_positions)} = {len(antenna_positions)**2}")

    # ========================================================================
    # STEP 4: Run FDTD for Each Transmitter (Multistatic Measurement)
    # ========================================================================
    print("\n[4/7] Running FDTD simulations...")
    print("  (This may take several minutes...)")

    # Initialize S-parameter collector
    collector = MultistaticDataCollector(
        antenna_array=array,
        frequencies=frequencies,
        dx=dx
    )

    t_start = time.time()

    for tx_idx in range(len(antenna_positions)):
        print(f"\n  Transmitter {tx_idx+1}/{len(antenna_positions)}...")

        # Create source at transmitter position
        tx_pos = antenna_positions[tx_idx]
        source = ModulatedGaussianSource(
            x=tx_pos[0],
            y=tx_pos[1],
            frequency=freq_center,
            bandwidth=bandwidth,
            field_component='Ez',
            delay_factor=5.0
        )

        # Set up FDTD solver
        solver = FDTD2D(
            config=config,
            sources=[source],
            boundaries=[CPML(thickness=10)],
            record_field='Ez'
        )

        # Apply tissue properties to grid
        eps_real, eps_imag = phantom.get_permittivity_map(freq_center)

        # Set permittivity (simplified - uses center frequency)
        for i in range(nx):
            for j in range(ny):
                if tissue_map[i, j] is not None:
                    solver.grid.set_material_region(
                        i, i+1, j, j+1,
                        eps_r=eps_real[i, j]
                    )

        # Add DFT monitors at all receiver positions
        dft_monitors = []
        for rx_pos in antenna_positions:
            monitor = DFTMonitor(
                frequencies=frequencies,
                region=(rx_pos[0], rx_pos[1]),
                component='Ez'
            )
            dft_monitors.append(monitor)
            solver.dft_monitors.append(monitor)

        # Run simulation
        solver.run(verbose=False)

        # Collect DFT data
        dft_data = {}
        for rx_idx, (monitor, rx_pos) in enumerate(zip(dft_monitors, antenna_positions)):
            # Get DFT data - monitor stores results in dft_data attribute
            dft_data[rx_pos] = monitor.dft_data[0, 0, :]  # Point monitor shape is (1, 1, n_freq)

        # Record in collector
        collector.record_transmission(tx_idx, dft_data)

        progress = (tx_idx + 1) / len(antenna_positions) * 100
        print(f"    Progress: {progress:.1f}%")

    t_elapsed = time.time() - t_start
    print(f"\n  ✓ All simulations complete in {t_elapsed:.1f} seconds")
    print(f"  Average time per TX: {t_elapsed/len(antenna_positions):.2f} s")

    # ========================================================================
    # STEP 5: Extract S-Parameters
    # ========================================================================
    print("\n[5/7] Extracting S-parameters...")

    s_params = collector.get_s_parameters()
    print(f"  S-matrix shape: {s_params.s_matrix.shape}")
    print(f"  Frequencies: {len(s_params.frequencies)}")
    print(f"  Antennas: {s_params.num_antennas}")

    # ========================================================================
    # STEP 6: Reconstruct Image with DAS Beamforming
    # ========================================================================
    print("\n[6/7] Reconstructing image with DAS beamforming...")

    # Define imaging region (inside the head)
    imaging_region = ImagingRegion(
        x_range=(50, 150),  # 100×100 pixel region
        y_range=(50, 150),
        dx=dx,
        background_permittivity=45.0  # Approximate brain permittivity
    )

    # Create beamformer
    beamformer = DelayAndSumBeamformer(
        s_parameters=s_params,
        antenna_positions=antenna_positions,
        imaging_region=imaging_region
    )

    # Reconstruct
    print("  Running DAS algorithm...")
    image = beamformer.reconstruct(method='das', normalize=True)

    print(f"  ✓ Image reconstructed: {image.shape}")

    # ========================================================================
    # STEP 7: Visualize Results
    # ========================================================================
    print("\n[7/7] Visualizing results...")

    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Phantom geometry
    ax1 = plt.subplot(131)
    phantom_image = np.ones((nx, ny)) * eps_real
    ax1.imshow(phantom_image.T, origin='lower', cmap='viridis', aspect='equal')
    ax1.set_title('Ground Truth\n(Permittivity Distribution)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (grid cells)')
    ax1.set_ylabel('Y (grid cells)')

    # Mark hemorrhage
    circle = plt.Circle((hemorrhage_x, hemorrhage_y), hemorrhage_radius,
                       color='red', fill=False, linewidth=3, linestyle='--',
                       label='Hemorrhage')
    ax1.add_patch(circle)

    # Mark antennas
    ant_x = [pos[0] for pos in antenna_positions]
    ant_y = [pos[1] for pos in antenna_positions]
    ax1.scatter(ant_x, ant_y, s=80, c='cyan', marker='^',
               edgecolors='blue', linewidth=2, label='Antennas', zorder=10)
    ax1.legend()

    # Plot 2: S-parameter magnitude
    ax2 = plt.subplot(132)
    s_mag = 20 * np.log10(np.abs(s_params.s_matrix[10]) + 1e-20)  # Mid frequency
    im2 = ax2.imshow(s_mag, cmap='viridis', aspect='equal', origin='lower')
    plt.colorbar(im2, ax=ax2, label='|S| (dB)')
    ax2.set_title(f'S-Parameter Matrix\nat {s_params.frequencies[10]/1e9:.2f} GHz',
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('TX Antenna')
    ax2.set_ylabel('RX Antenna')

    # Plot 3: Reconstructed image
    ax3 = plt.subplot(133)
    extent = [imaging_region.x_range[0], imaging_region.x_range[1],
             imaging_region.y_range[0], imaging_region.y_range[1]]
    im3 = ax3.imshow(image.T, origin='lower', cmap='hot', extent=extent,
                    aspect='equal', interpolation='bilinear')
    plt.colorbar(im3, ax=ax3, label='Normalized Intensity')
    ax3.set_title('DAS Reconstruction\n(Hemorrhage Detection)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (grid cells)')
    ax3.set_ylabel('Y (grid cells)')

    # Mark expected hemorrhage location
    circle2 = plt.Circle((hemorrhage_x, hemorrhage_y), hemorrhage_radius,
                        color='lime', fill=False, linewidth=3, linestyle='--',
                        label='True Location')
    ax3.add_patch(circle2)

    # Mark antennas
    ax3.scatter(ant_x, ant_y, s=80, c='cyan', marker='^',
               edgecolors='blue', linewidth=2, label='Antennas', zorder=10)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('hemorrhage_detection_results.png', dpi=150)
    print("  ✓ Figure saved: hemorrhage_detection_results.png")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"  Phantom: {nx}×{ny} head with {hemorrhage_radius}mm hemorrhage")
    print(f"  Array: {len(antenna_positions)} antennas, circular configuration")
    print(f"  Simulations: {len(antenna_positions)} FDTD runs")
    print(f"  S-matrix: {s_params.num_antennas}×{s_params.num_antennas} at {len(frequencies)} frequencies")
    print(f"  Image: {image.shape[0]}×{image.shape[1]} pixels")
    print(f"  Total time: {t_elapsed:.1f} seconds")
    print(f"\n  ✓ Complete pipeline executed successfully!")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()
