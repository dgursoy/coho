# coho/__init__.py

"""
Toolkit for wavefront propagation simulations.

This package provides modular components for optical simulations
using factory-based architecture.

Modules:
    config: Configuration management
        reader: Configuration file reading
        schemas: Schema registration and access
        validator: Configuration validation
        builder: Object construction from config

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
"""

# Configuration
from .config import (
    read_config,
    load_config
)

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

__all__ = [
    # Configuration
    'read_config',
    'load_config',
    # Factories
    'WavefrontFactory',
    'OpticFactory',
    'SampleFactory',
    'DetectorFactory',
    'PropagatorFactory',
    'InteractorFactory',
    'ExperimentFactory',
    # Core
    'Simulation',
]