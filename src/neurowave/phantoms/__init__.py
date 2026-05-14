"""
Anatomical Phantom Models for Biomedical Imaging
=================================================

Provides realistic tissue phantoms for microwave imaging validation.
"""

from neurowave.phantoms.head_models import (
    SimpleHeadPhantom,
    DetailedBrainPhantom,
    SkinLayerPhantom,
    visualize_phantom_slice
)

__all__ = [
    'SimpleHeadPhantom',
    'DetailedBrainPhantom',
    'SkinLayerPhantom',
    'visualize_phantom_slice'
]
