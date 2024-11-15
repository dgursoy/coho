# coho/__init__.py

"""
Toolkit for wavefront propagation simulations.

This package provides modular components for optical simulations
using factory-based architecture.

Modules:
    factories: Component creation and configuration
        wavefront: Wavefront profiles
        optic: Optical components
        sample: Light-sensitive materials
        detector: Measurement devices
        propagator: Propagation methods
        interactor: Wave-element interactions

    core: Core simulation components
        engine: Simulation orchestration
        wavefront: Wavefront definitions
        optic: Optical components
        sample: Light-sensitive materials
        detector: Detector definitions
        propagator: Propagator definitions
        interactor: Interaction definitions

    config: Configuration handling
        manager: Config loading and validation
        parser: Simulation construction
"""

# Factories
from .factories.simulation_factories import (
    WavefrontFactory,
    OpticFactory,
    SampleFactory,
    DetectorFactory
)
from .factories.operator_factories import (
    PropagatorFactory,
    InteractorFactory
)
from .factories.experiment_factories import ExperimentFactory

# Core
from .engine.simulation import Simulation

# Config
from .config.manager import load_simulation_config
from .config.parser import build_simulation_from_config

__all__ = [
    'WavefrontFactory',
    'OpticFactory',
    'SampleFactory',
    'DetectorFactory',
    'PropagatorFactory',
    'InteractorFactory',
    'ExperimentFactory',
    'Engine',
    'load_simulation_config',
    'build_simulation_from_config'
]