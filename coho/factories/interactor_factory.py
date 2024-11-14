# factories/interactor_factory.py

"""Factory for creating wavefront interactor instances.

This module manages creation of interactors for simulating
wavefront-object interactions in optical systems.

Classes:
    InteractorFactory: Creates configured interactor instances

Types:
    thin_object: Simple transmission functions
    thick_object: Multi-slice beam propagation

INTERACTOR_TYPES:
    INTERACTOR_TYPES: Mapping of type names to classes
"""

from typing import Any
from ..core.interactor import (
    Interactor,
    ThinObjectInteractor,
    ThickObjectInteractor
)
from ..core.wavefront import Wavefront


INTERACTOR_TYPES = {
    'thin_object': ThinObjectInteractor,
    'thick_object': ThickObjectInteractor
}


class InteractorFactory:
    """Factory for wavefront interactor creation."""
    
    @staticmethod
    def create_interactor(
        id: Any, 
        type: str, 
        wavefront: Wavefront
    ) -> Interactor:
        """Create configured interactor instance.

        Args:
            id: Unique identifier
            type: Interactor type
                'thin_object': Simple transmission
                'thick_object': Multi-slice propagation
            wavefront: Wavefront for interactions

        Returns:
            Configured interactor instance

        Raises:
            ValueError: Unknown interactor type
        """
        interactor_type = type.lower()
        interactor_class = INTERACTOR_TYPES.get(interactor_type)
        
        if interactor_class is None:
            raise ValueError(
                f"Unknown interactor type: {type}. "
                f"Supported types: {list(INTERACTOR_TYPES.keys())}"
            )
            
        return interactor_class(id, wavefront)
