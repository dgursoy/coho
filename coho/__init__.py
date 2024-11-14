# coho/__init__.py

"""
Toolkit for wavefront propagation simulations.

This package provides modular components for optical simulations
using factory-based architecture.

Modules:
    factories: Component creation and configuration
        wavefront: Wavefront profiles
        element: Optical elements
        detector: Measurement devices
        propagator: Propagation methods
        interactor: Wave-element interactions

    core: Core simulation components
        engine: Simulation orchestration
        wavefront: Wavefront definitions
        element: Element definitions
        detector: Detector definitions
        propagator: Propagator definitions
        interactor: Interaction definitions

    config: Configuration handling
        manager: Config loading and validation
        parser: Simulation construction
"""

# Factories
from .factories.wavefront_factory import WavefrontFactory
from .factories.element_factory import ElementFactory
from .factories.detector_factory import DetectorFactory
from .factories.propagator_factory import PropagatorFactory
from .factories.interactor_factory import InteractorFactory

# Core
from .engine import Simulation

# Config
from .config.manager import load_simulation_config
from .config.parser import build_simulation_from_config

__all__ = [
    'WavefrontFactory',
    'ElementFactory',
    'DetectorFactory',
    'PropagatorFactory',
    'InteractorFactory',
    'Simulation',
    'load_simulation_config',
    'build_simulation_from_config'
]
