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

from typing import Dict, Any
from ..core.wavefront import (
    Wavefront,
    ConstantWavefront, 
    GaussianWavefront, 
    RectangularWavefront
)


WAVEFRONT_TYPES = {
    'constant': ConstantWavefront,
    'gaussian': GaussianWavefront,
    'rectangular': RectangularWavefront
}


class WavefrontFactory:
    """Factory for wavefront creation."""
    
    @staticmethod
    def create_wavefront(
        id: Any, 
        type: str, 
        parameters: Dict[str, Any]
    ) -> Wavefront:
        """Create configured wavefront instance.

        Args:
            id: Unique identifier
            type: Wavefront type
                'constant': Uniform amplitude
                'gaussian': Gaussian profile
                'rectangular': Rectangular profile
            parameters: Configuration dictionary

        Returns:
            Configured wavefront instance

        Raises:
            ValueError: Unknown wavefront type
        """
        wavefront_type = type.lower()
        wavefront_class = WAVEFRONT_TYPES.get(wavefront_type)
        
        if wavefront_class is None:
            raise ValueError(
                f"Unknown wavefront type: {type}. "
                f"Supported types: {list(WAVEFRONT_TYPES.keys())}"
            )
            
        return wavefront_class(id, parameters)
