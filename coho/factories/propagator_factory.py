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

from ..core.propagator import (
    FresnelPropagator, 
    FraunhoferPropagator
)
from .base_factory import ComponentFactory


PROPAGATOR_TYPES = {
    'fresnel': FresnelPropagator,
    'fraunhofer': FraunhoferPropagator
}


class PropagatorFactory(ComponentFactory):
    """Factory for propagator creation."""
    
    def __init__(self):
        super().__init__(PROPAGATOR_TYPES)
