# factories/wavefront_factory.py

"""Factory for creating wavefront instances.

This module manages creation of different wavefront profiles
for optical simulations.

Classes:
    WavefrontFactory: Creates configured wavefront instances

Types:
    constant: Uniform amplitude profile
    gaussian: Gaussian amplitude distribution
    rectangular: Rectangular amplitude profile

Constants:
    WAVEFRONT_TYPES: Mapping of type names to classes
"""

from ..core.wavefront import (
    ConstantWavefront, 
    GaussianWavefront, 
    RectangularWavefront
)
from .base_factory import ComponentFactory


WAVEFRONT_TYPES = {
    'constant': ConstantWavefront,
    'gaussian': GaussianWavefront,
    'rectangular': RectangularWavefront
}


class WavefrontFactory(ComponentFactory):
    """Factory for wavefront interactor creation."""
    
    def __init__(self):
        super().__init__(WAVEFRONT_TYPES)
