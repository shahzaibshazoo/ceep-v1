"""
Tests for the Gabriel tissue dielectric database.
"""

import numpy as np
import pytest
from neurowave.materials.tissue_database import TissueDatabase, plot_tissue_spectrum


def test_database_initialization():
    """Test that database loads correctly."""
    db = TissueDatabase()
    tissues = db.list_tissues()
    assert len(tissues) > 20, "Database should contain > 20 tissues"
    assert 'brain_gray_matter' in tissues
    assert 'blood' in tissues


def test_get_tissue_properties():
    """Test retrieving tissue properties."""
    db = TissueDatabase()
    brain = db.get('brain_gray_matter')
    assert brain.name == 'Brain (Gray Matter)'
    assert brain.eps_inf > 0
    assert len(brain.delta_eps) == 4
    assert len(brain.tau) == 4
    assert len(brain.alpha) == 4


def test_brain_gray_matter_at_10ghz():
    """Test brain gray matter permittivity at 10 GHz.

    Expected values from literature:
    ε_r ≈ 48-52 (real part)
    """
    db = TissueDatabase()
    brain = db.get('brain_gray_matter')
    eps = brain.permittivity(np.array([10e9]))

    # Real part should be around 48-52
    assert 40 < eps[0].real < 60, f"Expected ε' ≈ 48-52, got {eps[0].real:.1f}"

    # Imaginary part should be positive (loss)
    assert eps[0].imag < 0, "Loss factor should be negative (ε'' positive)"


def test_blood_high_conductivity():
    """Test that blood has higher conductivity than gray matter."""
    db = TissueDatabase()
    brain = db.get('brain_gray_matter')
    blood = db.get('blood')

    # Blood should have much higher static conductivity
    assert blood.sigma_s > brain.sigma_s * 10

    # Test at 1 GHz
    freqs = np.array([1e9])
    sigma_brain = brain.conductivity(freqs)[0]
    sigma_blood = blood.conductivity(freqs)[0]
    assert sigma_blood > sigma_brain


def test_cole_cole_pole_extraction():
    """Test extraction of Cole-Cole poles for FDTD."""
    db = TissueDatabase()
    brain = db.get('brain_gray_matter')
    poles = brain.get_cole_cole_poles()

    # Should have 4 poles
    assert len(poles) == 4

    # All poles should have valid parameters
    for pole in poles:
        assert pole.delta_eps > 0
        assert pole.tau > 0
        assert 0 <= pole.alpha < 1


def test_frequency_dispersion():
    """Test that permittivity decreases with frequency."""
    db = TissueDatabase()
    brain = db.get('brain_gray_matter')

    freqs = np.array([100e6, 1e9, 10e9])  # 100 MHz, 1 GHz, 10 GHz
    eps = brain.permittivity(freqs)

    # Real part should generally decrease with frequency
    assert eps[0].real > eps[2].real


def test_categories():
    """Test tissue categorization."""
    db = TissueDatabase()
    categories = db.list_by_category()

    assert 'Brain' in categories
    assert 'blood' in categories['Blood']
    assert len(categories['Brain']) >= 3  # gray, white, CSF


def test_pathological_tissues():
    """Test that pathological tissues are available."""
    db = TissueDatabase()

    hemorrhage = db.get('hemorrhage')
    edema = db.get('edema')
    ischemia = db.get('ischemia')

    # Hemorrhage should have high conductivity (blood-like)
    assert hemorrhage.sigma_s > 0.7

    # Edema should have very high water content → high eps
    freqs = np.array([1e9])
    eps_edema = edema.permittivity(freqs)[0].real
    assert eps_edema > 55  # Very high due to water


def test_breast_tissues():
    """Test breast tissue models for cancer imaging."""
    db = TissueDatabase()

    fat = db.get('breast_fat')
    gland = db.get('breast_gland')
    tumor = db.get('breast_tumor')

    freqs = np.array([3e9])  # 3 GHz common for breast imaging

    eps_fat = fat.permittivity(freqs)[0].real
    eps_gland = gland.permittivity(freqs)[0].real
    eps_tumor = tumor.permittivity(freqs)[0].real

    # Tumor should have highest permittivity
    assert eps_tumor > eps_gland > eps_fat

    # Tumor should have highest conductivity
    assert tumor.sigma_s > gland.sigma_s > fat.sigma_s


def test_invalid_tissue_key():
    """Test error handling for invalid tissue keys."""
    db = TissueDatabase()
    with pytest.raises(KeyError):
        db.get('nonexistent_tissue')


def test_permittivity_shape():
    """Test that permittivity returns correct shape."""
    db = TissueDatabase()
    brain = db.get('brain_gray_matter')

    freqs = np.logspace(6, 10, 100)  # 1 MHz to 10 GHz
    eps = brain.permittivity(freqs)

    assert eps.shape == freqs.shape
    assert eps.dtype == complex


def test_conductivity_positive():
    """Test that effective conductivity is always positive."""
    db = TissueDatabase()

    for tissue_key in db.list_tissues():
        tissue = db.get(tissue_key)
        freqs = np.array([1e9, 10e9])
        sigma = tissue.conductivity(freqs)
        assert np.all(sigma >= 0), f"{tissue.name} has negative conductivity"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
