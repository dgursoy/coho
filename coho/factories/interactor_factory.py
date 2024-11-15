# factories/interactor_factory.py

"""Factory for creating interactor instances.

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

from ..core.interactor import (
    ThinObjectInteractor,
    ThickObjectInteractor
)
from .base_factory import ComponentFactory


INTERACTOR_TYPES = {
    'thin_object': ThinObjectInteractor,
    'thick_object': ThickObjectInteractor
}


class InteractorFactory(ComponentFactory):
    """Factory for interactor creation."""
    
    def __init__(self):
        super().__init__(INTERACTOR_TYPES)
