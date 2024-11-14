# factories/__init__.py

"""Factories for simulation component creation.

This package provides factory classes for initializing
and configuring optical components.

Modules:
    element: Optical elements
        lenses, mirrors, apertures

    detector: Measurement devices
        intensity, photon counting

    propagator: Propagation methods
        fresnel, fraunhofer

    wavefront: Initial profiles
        constant, gaussian, rectangular

    interactor: Wave-object interactions
        thin, thick objects
"""

from .element_factory import ElementFactory
from .detector_factory import DetectorFactory
from .propagator_factory import PropagatorFactory
from .wavefront_factory import WavefrontFactory
from .interactor_factory import InteractorFactory

__all__ = [
    'ElementFactory',
    'DetectorFactory',
    'PropagatorFactory',
    'WavefrontFactory',
    'InteractorFactory'
]
