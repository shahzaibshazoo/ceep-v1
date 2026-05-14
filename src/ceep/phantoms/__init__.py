"""
Anatomical Phantom Models for Biomedical Imaging
=================================================

Provides realistic tissue phantoms for microwave imaging validation.
"""

from ceep.phantoms.head_models import (
    SimpleHeadPhantom,
    DetailedBrainPhantom,
    SkinLayerPhantom,
    visualize_phantom_slice
)

# Alias for convenience
BrainPhantom = DetailedBrainPhantom

__all__ = [
    'SimpleHeadPhantom',
    'DetailedBrainPhantom',
    'BrainPhantom',  # Alias
    'SkinLayerPhantom',
    'visualize_phantom_slice'
]
