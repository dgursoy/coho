# factories/element_factory.py

"""Factory for optical element creation.

Provides centralized creation of different element types
with proper initialization and configuration.

Classes:
    ElementFactory: Creates configured element instances

Types:
    coded_aperture: Patterned transmission regions
    slit_aperture: Single slit opening
    circle_aperture: Circular opening
    custom_profile: User-defined pattern

ELEMENT_TYPES:
    ELEMENT_TYPES: Mapping of type names to classes
"""

from typing import Dict, Any
from ..core.element import (
    Element,
    CodedApertureElement,
    SlitApertureElement,
    CircleApertureElement,
    CustomProfileElement
)


ELEMENT_TYPES = {
    'coded_aperture': CodedApertureElement,
    'slit_aperture': SlitApertureElement,
    'circle_aperture': CircleApertureElement,
    'custom_profile': CustomProfileElement
}


class ElementFactory:
    """Factory for optical element creation."""
    
    @staticmethod
    def create_element(
        id: Any, 
        type: str, 
        parameters: Dict[str, Any]
    ) -> Element:
        """Create configured element instance.

        Args:
            id: Unique identifier
            type: Element type
                'coded_aperture': Patterned transmission
                'slit_aperture': Single slit
                'circle_aperture': Circular opening
                'custom_profile': User pattern
            parameters: Configuration dictionary

        Returns:
            Configured element instance

        Raises:
            ValueError: Unknown element type
        """
        element_type = type.lower()
        element_class = ELEMENT_TYPES.get(element_type)
        
        if element_class is None:
            raise ValueError(
                f"Unknown element type: {type}. "
                f"Supported types: {list(ELEMENT_TYPES.keys())}"
            )
            
        return element_class(id, parameters)
