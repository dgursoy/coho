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

from .base_factory import ComponentFactory
from ..core.operator.propagator import *
from ..core.operator.interactor import *
from ..config.models import *

__all__ = ['PropagatorFactory', 'InteractorFactory']


class PropagatorFactory(ComponentFactory[PropagatorProperties, Propagator]):
    """Factory for propagator creation."""
    def __init__(self):
        super().__init__('operator', 'propagator')


class InteractorFactory(ComponentFactory[InteractorProperties, Interactor]):
    """Factory for interactor creation."""
    def __init__(self):
        super().__init__('operator', 'interactor')
