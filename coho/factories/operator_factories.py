# factories/operator_factories.py

"""Factories for operator components.

This module provides factories for creating forward and adjoint operators
used in both simulation and optimization.

Classes:
    PropagatorFactory: Creates propagator instances
    InteractorFactory: Creates interactor instances

Types:
    Propagators:
        fresnel: Near-field diffraction
        fraunhofer: Far-field diffraction
    
    Interactors:
        thin_object: Simple transmission functions
        thick_object: Multi-slice beam propagation
"""

from ..core.operator.propagator import (
    FresnelPropagator,
    FraunhoferPropagator
)
from ..core.operator.interactor import (
    ThinObjectInteractor,
    ThickObjectInteractor
)
from .base_factory import ComponentFactory


# Component type mappings
PROPAGATOR_TYPES = {
    'fresnel': FresnelPropagator,
    'fraunhofer': FraunhoferPropagator
}

INTERACTOR_TYPES = {
    'thin_object': ThinObjectInteractor,
    'thick_object': ThickObjectInteractor
}


class PropagatorFactory(ComponentFactory):
    """Factory for propagator creation."""
    def __init__(self):
        super().__init__(PROPAGATOR_TYPES)


class InteractorFactory(ComponentFactory):
    """Factory for interactor creation."""
    def __init__(self):
        super().__init__(INTERACTOR_TYPES)
