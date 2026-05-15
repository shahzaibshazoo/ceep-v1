"""
Anatomical Phantom Models for Biomedical Imaging
=================================================

Provides realistic tissue phantoms for microwave imaging validation.
"""

from ceep.phantoms.head_models import (
    SimpleHeadPhantom,
    DetailedBrainPhantom,
    BrainPhantom2D,
    SkinLayerPhantom,
    visualize_phantom_slice
)

# BrainPhantom = convenience 2D class that works with BatchedFDTD2D.set_phantom()
BrainPhantom = BrainPhantom2D

__all__ = [
    'SimpleHeadPhantom',
    'DetailedBrainPhantom',
    'BrainPhantom2D',
    'BrainPhantom',
    'SkinLayerPhantom',
    'visualize_phantom_slice'
]
