# factories/propagator_factory.py

"""Factory for creating wavefront propagator instances.

This module manages creation of propagators for different
diffraction calculation methods.

Classes:
    PropagatorFactory: Creates configured propagator instances

Types:
    fresnel: Near-field diffraction
    fraunhofer: Far-field diffraction

PROPAGATOR_TYPES:
    PROPAGATOR_TYPES: Mapping of type names to classes
"""

from typing import Dict, Any
from ..core.propagator import (
    Propagator,
    FresnelPropagator, 
    FraunhoferPropagator
)


PROPAGATOR_TYPES = {
    'fresnel': FresnelPropagator,
    'fraunhofer': FraunhoferPropagator
}


class PropagatorFactory:
    """Factory for propagator creation."""
    
    @staticmethod
    def create_propagator(
        id: Any, 
        type: str, 
        parameters: Dict[str, Any]
    ) -> Propagator:
        """Create configured propagator instance.

        Args:
            id: Unique identifier
            type: Propagator type
                'fresnel': Near-field diffraction
                'fraunhofer': Far-field diffraction
            parameters: Configuration dictionary

        Returns:
            Configured propagator instance

        Raises:
            ValueError: Unknown propagator type
        """
        propagator_type = type.lower()
        propagator_class = PROPAGATOR_TYPES.get(propagator_type)
        
        if propagator_class is None:
            raise ValueError(
                f"Unknown propagator type: {type}. "
                f"Supported types: {list(PROPAGATOR_TYPES.keys())}"
            )
            
        return propagator_class(id, parameters)
