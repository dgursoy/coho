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

from ..core.element import (
    CodedApertureElement,
    SlitApertureElement,
    CircleApertureElement,
    CustomProfileElement
)
from .base_factory import ComponentFactory


ELEMENT_TYPES = {
    'coded_aperture': CodedApertureElement,
    'slit_aperture': SlitApertureElement,
    'circle_aperture': CircleApertureElement,
    'custom_profile': CustomProfileElement
}


class ElementFactory(ComponentFactory):
    """Factory for element creation."""
    
    def __init__(self):
        super().__init__(ELEMENT_TYPES)

