# core/operator/__init__.py

"""Core operator components for wave propagation.

This module provides base classes for forward and adjoint operators
used in both simulation and optimization.

Components:
    Propagator: Field propagation methods
        fresnel: Near-field diffraction
        fraunhofer: Far-field diffraction

    Interactor: Wave-object interactions
        thin_object: Simple transmission functions
        thick_object: Multi-slice beam propagation
"""

from .propagator import *
from .interactor import *

__all__ = [
    'FresnelPropagator',
    'FraunhoferPropagator',
    'ThinObjectInteractor',
    'ThickObjectInteractor'
]
