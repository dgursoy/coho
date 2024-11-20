# core/__init__.py

"""Core components for optical simulation.

This package provides base classes and implementations for simulation,
operators, optimization and experiments.

Modules:
    simulation: Physical components
        detector: Measurement devices
        element: Optical elements
        wavefront: Light field profiles

    operator: Forward & adjoint operators
        propagator: Field propagation methods
        interactor: Wave-object interactions

    optimization: Optimization tools
        solver: Optimization algorithms
        objective: Cost functions

    experiment: High-level templates
        phase_retrieval: Phase recovery
        holography: Holographic imaging
        tomography: Tomographic reconstruction
"""

from .simulation import *
from .operator import *
from .optimization import *

__all__ = [
    # Simulation
    'IntegratingDetector',
    'PhotonCountingDetector',
    'CodedApertureOptic',
    'SlitApertureOptic', 
    'CircleApertureOptic',
    'CustomProfileOptic',
    'CustomProfileSample',
    'ConstantWavefront',
    'GaussianWavefront',
    'RectangularWavefront',
    
    # Operators
    'FresnelPropagator',
    'FraunhoferPropagator',
    'ThinObjectInteractor',
    'ThickObjectInteractor',
    'Holography',
    
    # Optimization
    'GradientDescent',
    'LeastSquares',
]
