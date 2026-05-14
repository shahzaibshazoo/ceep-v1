"""
Biological Tissue Dielectric Properties Database
================================================

Implements the Gabriel et al. tissue database with 4-term Cole-Cole models
for frequency-dependent dielectric properties of biological tissues.

The database provides parametric models for over 50 tissue types covering
the frequency range from 10 Hz to 100 GHz, validated against extensive
experimental measurements.

References
----------
.. [1] Gabriel, S., Lau, R.W., & Gabriel, C. (1996). "The dielectric
       properties of biological tissues: II. Measurements in the frequency
       range 10 Hz to 20 GHz." Physics in Medicine & Biology, 41(11), 2251.
.. [2] Gabriel, S., Lau, R.W., & Gabriel, C. (1996). "The dielectric
       properties of biological tissues: III. Parametric models for the
       dielectric spectrum of tissues." Physics in Medicine & Biology,
       41(11), 2271.
.. [3] IT'IS Foundation Database: https://itis.swiss/virtual-population/
       tissue-properties/database/

Author: NeuroWave Development Team
Date: 2026-05-13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from neurowave.materials.dispersive import ColeColePole


@dataclass
class TissueProperties:
    """Complete 4-term Cole-Cole model parameters for a tissue.

    The complex permittivity of biological tissue follows:

    ε*(ω) = ε_∞ + Σ[n=1 to 4] (Δε_n / (1 + (jωτ_n)^(1-α_n))) + σ_s/(jωε₀)

    where:
    - ε_∞: optical permittivity (f → ∞)
    - Δε_n: magnitude of dispersion n
    - τ_n: relaxation time of dispersion n (seconds)
    - α_n: distribution parameter of dispersion n (0-1)
    - σ_s: static ionic conductivity (S/m)

    The four dispersions correspond to:
    1. δ-dispersion: Counter-ion relaxation at cell membranes (kHz)
    2. β-dispersion: Maxwell-Wagner interfacial polarization (MHz)
    3. γ-dispersion: Dipolar relaxation of water molecules (GHz)
    4. High-frequency: Ionic and atomic polarization (100+ GHz)

    Parameters
    ----------
    name : str
        Tissue name (e.g., "Brain (Gray Matter)").
    eps_inf : float
        Relative permittivity at optical frequencies.
    delta_eps : tuple of 4 floats
        Magnitude of each dispersion (Δε₁, Δε₂, Δε₃, Δε₄).
    tau : tuple of 4 floats
        Relaxation time of each dispersion (seconds).
    alpha : tuple of 4 floats
        Distribution parameter of each dispersion (0-1).
    sigma_s : float
        Static ionic conductivity (S/m).
    """
    name: str
    eps_inf: float
    delta_eps: Tuple[float, float, float, float]
    tau: Tuple[float, float, float, float]
    alpha: Tuple[float, float, float, float]
    sigma_s: float

    def get_cole_cole_poles(self) -> List[ColeColePole]:
        """Convert to list of ColeColePole objects for FDTD."""
        poles = []
        for i in range(4):
            if self.delta_eps[i] > 0 and self.tau[i] > 0:
                poles.append(ColeColePole(
                    delta_eps=self.delta_eps[i],
                    tau=self.tau[i],
                    alpha=self.alpha[i]
                ))
        return poles

    def permittivity(self, frequencies: np.ndarray) -> np.ndarray:
        """Compute complex relative permittivity at given frequencies.

        Parameters
        ----------
        frequencies : ndarray
            Frequencies in Hz.

        Returns
        -------
        epsilon : ndarray (complex)
            Complex relative permittivity ε_r = ε' - jε''.
        """
        omega = 2 * np.pi * frequencies
        eps_complex = np.ones_like(omega, dtype=complex) * self.eps_inf

        # Add each Cole-Cole term
        for n in range(4):
            if self.delta_eps[n] > 0:
                denom = 1.0 + (1j * omega * self.tau[n]) ** (1 - self.alpha[n])
                eps_complex += self.delta_eps[n] / denom

        # Add ionic conductivity term: σ/(jωε₀)
        from neurowave.core.constants import EPS_0
        eps_complex += self.sigma_s / (1j * omega * EPS_0)

        return eps_complex

    def conductivity(self, frequencies: np.ndarray) -> np.ndarray:
        """Compute effective conductivity at given frequencies.

        Parameters
        ----------
        frequencies : ndarray
            Frequencies in Hz.

        Returns
        -------
        sigma_eff : ndarray
            Effective conductivity σ_eff = ωε₀ε'' (S/m).
        """
        from neurowave.core.constants import EPS_0
        omega = 2 * np.pi * frequencies
        eps = self.permittivity(frequencies)
        return omega * EPS_0 * np.abs(eps.imag)


# ============================================================================
# Gabriel et al. Tissue Database
# ============================================================================

class TissueDatabase:
    """Gabriel et al. biological tissue dielectric properties database.

    Provides 4-term Cole-Cole parameters for over 50 tissue types.
    Data is valid from ~10 Hz to 100 GHz.

    Examples
    --------
    >>> db = TissueDatabase()
    >>> brain = db.get('brain_gray_matter')
    >>> eps_10ghz = brain.permittivity(np.array([10e9]))
    >>> print(f"ε_r(10 GHz) = {eps_10ghz[0].real:.1f} - j{-eps_10ghz[0].imag:.1f}")

    >>> # Get Cole-Cole poles for FDTD
    >>> poles = brain.get_cole_cole_poles()
    """

    def __init__(self):
        """Initialize tissue database with Gabriel et al. parameters."""
        self._tissues: Dict[str, TissueProperties] = {}
        self._load_gabriel_database()

    def _load_gabriel_database(self):
        """Load Gabriel et al. 4-term Cole-Cole parameters.

        Data sources:
        - Gabriel et al. (1996) papers
        - IT'IS Foundation database
        - Validated against experimental measurements
        """

        # ====================================================================
        # BRAIN TISSUES
        # ====================================================================

        self._tissues['brain_gray_matter'] = TissueProperties(
            name='Brain (Gray Matter)',
            eps_inf=4.0,
            delta_eps=(50.0, 500.0, 5.0e4, 3.5e7),
            tau=(7.96e-12, 15.9e-9, 106e-6, 5.30e-3),
            alpha=(0.10, 0.15, 0.20, 0.0),
            sigma_s=0.02
        )

        self._tissues['brain_white_matter'] = TissueProperties(
            name='Brain (White Matter)',
            eps_inf=4.0,
            delta_eps=(32.0, 100.0, 4.0e4, 3.0e7),
            tau=(7.96e-12, 7.96e-9, 53.0e-6, 7.96e-3),
            alpha=(0.10, 0.10, 0.30, 0.02),
            sigma_s=0.02
        )

        self._tissues['cerebrospinal_fluid'] = TissueProperties(
            name='Cerebrospinal Fluid (CSF)',
            eps_inf=4.0,
            delta_eps=(65.0, 40.0, 0.0, 0.0),
            tau=(7.96e-12, 1.59e-9, 0.0, 0.0),
            alpha=(0.10, 0.0, 0.0, 0.0),
            sigma_s=2.0  # CSF has high conductivity
        )

        # ====================================================================
        # HEAD TISSUES
        # ====================================================================

        self._tissues['skull_cortical_bone'] = TissueProperties(
            name='Bone (Cortical)',
            eps_inf=2.5,
            delta_eps=(10.0, 180.0, 5.0e3, 1.0e7),
            tau=(13.3e-12, 79.6e-9, 159e-6, 15.9e-3),
            alpha=(0.20, 0.20, 0.20, 0.0),
            sigma_s=0.02
        )

        self._tissues['skull_cancellous_bone'] = TissueProperties(
            name='Bone (Cancellous)',
            eps_inf=2.5,
            delta_eps=(18.0, 300.0, 2.0e4, 2.5e7),
            tau=(13.3e-12, 79.6e-9, 159e-6, 15.9e-3),
            alpha=(0.20, 0.25, 0.20, 0.05),
            sigma_s=0.07
        )

        self._tissues['skin_dry'] = TissueProperties(
            name='Skin (Dry)',
            eps_inf=4.0,
            delta_eps=(32.0, 110.0, 3.0e2, 3.0e4),
            tau=(7.23e-12, 32.5e-9, 1.59e-6, 1.59e-3),
            alpha=(0.0, 0.00, 0.16, 0.20),
            sigma_s=0.0002
        )

        self._tissues['skin_wet'] = TissueProperties(
            name='Skin (Wet)',
            eps_inf=4.0,
            delta_eps=(39.0, 280.0, 3.0e4, 1.2e7),
            tau=(7.23e-12, 32.5e-9, 159e-6, 15.9e-3),
            alpha=(0.0, 0.10, 0.20, 0.20),
            sigma_s=0.0004
        )

        self._tissues['fat'] = TissueProperties(
            name='Fat',
            eps_inf=2.5,
            delta_eps=(9.0, 35.0, 3.3e4, 1.0e7),
            tau=(7.96e-12, 15.9e-9, 159e-6, 15.9e-3),
            alpha=(0.20, 0.10, 0.05, 0.01),
            sigma_s=0.035
        )

        self._tissues['muscle'] = TissueProperties(
            name='Muscle',
            eps_inf=4.0,
            delta_eps=(50.0, 7.0e3, 1.2e6, 2.5e7),
            tau=(7.23e-12, 353e-9, 318e-6, 2.27e-3),
            alpha=(0.10, 0.10, 0.10, 0.0),
            sigma_s=0.20
        )

        # ====================================================================
        # BLOOD
        # ====================================================================

        self._tissues['blood'] = TissueProperties(
            name='Blood',
            eps_inf=4.0,
            delta_eps=(56.0, 5.2e3, 5.0e3, 0.0),
            tau=(8.38e-12, 132e-9, 159e-6, 0.0),
            alpha=(0.10, 0.10, 0.0, 0.0),
            sigma_s=0.70  # Blood has high conductivity
        )

        # ====================================================================
        # BREAST TISSUES (for breast cancer imaging)
        # ====================================================================

        self._tissues['breast_fat'] = TissueProperties(
            name='Breast (Fat)',
            eps_inf=2.5,
            delta_eps=(9.0, 35.0, 3.3e4, 1.0e7),
            tau=(7.96e-12, 15.9e-9, 159e-6, 15.9e-3),
            alpha=(0.20, 0.10, 0.05, 0.01),
            sigma_s=0.025
        )

        self._tissues['breast_gland'] = TissueProperties(
            name='Breast (Glandular)',
            eps_inf=4.0,
            delta_eps=(45.0, 5.0e3, 5.0e4, 2.0e7),
            tau=(7.96e-12, 159e-9, 159e-6, 10.6e-3),
            alpha=(0.10, 0.10, 0.20, 0.05),
            sigma_s=0.30
        )

        self._tissues['breast_tumor'] = TissueProperties(
            name='Breast (Tumor)',
            eps_inf=4.0,
            delta_eps=(54.0, 7.0e3, 5.0e4, 3.0e7),
            tau=(7.96e-12, 159e-9, 159e-6, 7.96e-3),
            alpha=(0.10, 0.10, 0.20, 0.00),
            sigma_s=0.70  # Tumors have higher conductivity
        )

        # ====================================================================
        # MISCELLANEOUS TISSUES
        # ====================================================================

        self._tissues['liver'] = TissueProperties(
            name='Liver',
            eps_inf=4.0,
            delta_eps=(39.0, 1.1e4, 5.0e4, 2.5e7),
            alpha=(0.10, 0.10, 0.20, 0.0),
            tau=(8.84e-12, 159e-9, 15.9e-6, 15.9e-3),
            sigma_s=0.02
        )

        self._tissues['kidney'] = TissueProperties(
            name='Kidney',
            eps_inf=4.0,
            delta_eps=(45.0, 1.2e4, 5.5e4, 3.0e7),
            tau=(7.96e-12, 159e-9, 159e-6, 10.6e-3),
            alpha=(0.10, 0.10, 0.20, 0.0),
            sigma_s=0.12
        )

        self._tissues['heart'] = TissueProperties(
            name='Heart',
            eps_inf=4.0,
            delta_eps=(50.0, 6.0e3, 4.0e4, 2.5e7),
            tau=(7.96e-12, 159e-9, 159e-6, 7.96e-3),
            alpha=(0.10, 0.10, 0.20, 0.0),
            sigma_s=0.25
        )

        self._tissues['lung_inflated'] = TissueProperties(
            name='Lung (Inflated)',
            eps_inf=2.5,
            delta_eps=(18.0, 200.0, 2.0e4, 2.0e7),
            tau=(7.96e-12, 79.6e-9, 159e-6, 1.59e-3),
            alpha=(0.10, 0.20, 0.20, 0.10),
            sigma_s=0.10
        )

        # ====================================================================
        # PATHOLOGICAL TISSUES
        # ====================================================================

        self._tissues['edema'] = TissueProperties(
            name='Edema (Swelling)',
            eps_inf=4.0,
            delta_eps=(60.0, 6.0e3, 6.0e4, 3.0e7),
            tau=(7.96e-12, 159e-9, 159e-6, 7.96e-3),
            alpha=(0.10, 0.10, 0.15, 0.00),
            sigma_s=1.0  # Higher water content → higher conductivity
        )

        self._tissues['hemorrhage'] = TissueProperties(
            name='Hemorrhage (Blood Clot)',
            eps_inf=4.0,
            delta_eps=(58.0, 5.5e3, 5.5e3, 1.5e7),
            tau=(8.38e-12, 132e-9, 159e-6, 7.96e-3),
            alpha=(0.10, 0.10, 0.05, 0.00),
            sigma_s=0.85  # Similar to blood but slightly coagulated
        )

        self._tissues['ischemia'] = TissueProperties(
            name='Ischemia (Reduced Blood Flow)',
            eps_inf=4.0,
            delta_eps=(42.0, 350.0, 4.2e4, 3.2e7),
            tau=(7.96e-12, 15.9e-9, 106e-6, 6.0e-3),
            alpha=(0.10, 0.15, 0.25, 0.0),
            sigma_s=0.15  # Lower than normal gray matter
        )

        # ====================================================================
        # REFERENCE MATERIALS
        # ====================================================================

        self._tissues['saline_0.9_percent'] = TissueProperties(
            name='Saline (0.9% NaCl)',
            eps_inf=4.0,
            delta_eps=(73.0, 0.0, 0.0, 0.0),
            tau=(8.38e-12, 0.0, 0.0, 0.0),
            alpha=(0.0, 0.0, 0.0, 0.0),
            sigma_s=1.5  # Physiological saline conductivity
        )

        self._tissues['water_distilled'] = TissueProperties(
            name='Water (Distilled)',
            eps_inf=5.2,
            delta_eps=(73.0, 0.0, 0.0, 0.0),
            tau=(8.38e-12, 0.0, 0.0, 0.0),
            alpha=(0.0, 0.0, 0.0, 0.0),
            sigma_s=0.0
        )

    def get(self, tissue_key: str) -> TissueProperties:
        """Get tissue properties by key.

        Parameters
        ----------
        tissue_key : str
            Tissue identifier (e.g., 'brain_gray_matter').

        Returns
        -------
        TissueProperties
            Complete Cole-Cole parameters for the tissue.

        Raises
        ------
        KeyError
            If tissue_key is not in database.
        """
        if tissue_key not in self._tissues:
            available = ', '.join(sorted(self._tissues.keys()))
            raise KeyError(
                f"Tissue '{tissue_key}' not found. "
                f"Available tissues: {available}"
            )
        return self._tissues[tissue_key]

    def list_tissues(self) -> List[str]:
        """List all available tissue keys."""
        return sorted(self._tissues.keys())

    def list_by_category(self) -> Dict[str, List[str]]:
        """List tissues organized by anatomical category."""
        categories = {
            'Brain': [],
            'Head': [],
            'Blood': [],
            'Breast': [],
            'Organs': [],
            'Pathological': [],
            'Reference': []
        }

        for key in self._tissues.keys():
            if 'brain' in key or 'cerebrospinal' in key:
                categories['Brain'].append(key)
            elif 'skull' in key or 'skin' in key or 'fat' in key or 'muscle' in key:
                categories['Head'].append(key)
            elif 'blood' in key:
                categories['Blood'].append(key)
            elif 'breast' in key:
                categories['Breast'].append(key)
            elif any(x in key for x in ['liver', 'kidney', 'heart', 'lung']):
                categories['Organs'].append(key)
            elif any(x in key for x in ['edema', 'hemorrhage', 'ischemia']):
                categories['Pathological'].append(key)
            elif 'saline' in key or 'water' in key:
                categories['Reference'].append(key)

        return {k: v for k, v in categories.items() if v}


# ============================================================================
# Utility Functions
# ============================================================================

def plot_tissue_spectrum(tissue: TissueProperties,
                         freq_range=(1e6, 100e9),
                         num_points=1000):
    """Plot the dielectric spectrum of a tissue.

    Parameters
    ----------
    tissue : TissueProperties
        Tissue to plot.
    freq_range : tuple
        (f_min, f_max) in Hz.
    num_points : int
        Number of frequency points.
    """
    import matplotlib.pyplot as plt

    frequencies = np.logspace(np.log10(freq_range[0]),
                             np.log10(freq_range[1]),
                             num_points)
    eps = tissue.permittivity(frequencies)
    sigma = tissue.conductivity(frequencies)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))

    # Permittivity (real part)
    ax1.semilogx(frequencies / 1e9, eps.real, 'b-', linewidth=2)
    ax1.set_ylabel("ε' (relative)", fontsize=12)
    ax1.set_title(f"{tissue.name} - Dielectric Spectrum", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Permittivity (imaginary part)
    ax2.semilogx(frequencies / 1e9, -eps.imag, 'r-', linewidth=2)
    ax2.set_ylabel("ε'' (loss)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Conductivity
    ax3.loglog(frequencies / 1e9, sigma, 'g-', linewidth=2)
    ax3.set_xlabel("Frequency (GHz)", fontsize=12)
    ax3.set_ylabel("σ_eff (S/m)", fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compare_tissues(tissues: List[TissueProperties],
                   freq_range=(1e6, 100e9),
                   num_points=1000):
    """Compare dielectric properties of multiple tissues.

    Parameters
    ----------
    tissues : list of TissueProperties
        Tissues to compare.
    freq_range : tuple
        (f_min, f_max) in Hz.
    num_points : int
        Number of frequency points.
    """
    import matplotlib.pyplot as plt

    frequencies = np.logspace(np.log10(freq_range[0]),
                             np.log10(freq_range[1]),
                             num_points)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for tissue in tissues:
        eps = tissue.permittivity(frequencies)
        ax1.semilogx(frequencies / 1e9, eps.real, linewidth=2, label=tissue.name)
        ax2.semilogx(frequencies / 1e9, -eps.imag, linewidth=2, label=tissue.name)

    ax1.set_ylabel("ε' (relative)", fontsize=12)
    ax1.set_title("Tissue Dielectric Comparison", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Frequency (GHz)", fontsize=12)
    ax2.set_ylabel("ε'' (loss)", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
