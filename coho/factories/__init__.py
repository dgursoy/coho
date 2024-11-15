"""Factory components for coho.

This package provides factories for creating simulation, operator,
optimization and experiment components.

Modules:
    base_factory: Abstract base factory implementation
    simulation_factories: Physical components
        detector: Measurement devices
        optic: Optical components
        sample: Light-sensitive materials
        wavefront: Light field profiles

    operator_factories: Forward & adjoint operators
        propagator: Field propagation methods
        interactor: Wave-object interactions

    optimization_factories: Optimization tools
        solver: Optimization algorithms
        objective: Cost functions

    experiment_factories: High-level templates
        holography: Holographic imaging
"""

from .base_factory import ComponentFactory
from .simulation_factories import (
    DetectorFactory,
    OpticFactory,
    SampleFactory,
    WavefrontFactory
)
from .operator_factories import (
    PropagatorFactory,
    InteractorFactory
)
from .optimization_factories import (
    SolverFactory,
    ObjectiveFactory
)
from .experiment_factories import ExperimentFactory

__all__ = [
    'ComponentFactory',
    'DetectorFactory', 
    'OpticFactory',
    'SampleFactory',
    'WavefrontFactory',
    'PropagatorFactory',
    'InteractorFactory',
    'GradientFactory',
    'SolverFactory',
    'ObjectiveFactory',
    'ExperimentFactory'
]