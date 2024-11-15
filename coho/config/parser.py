# config/parser.py

"""Configuration parser for simulation components.

This module handles component creation and configuration through factory pattern.
Each component is initialized with unique ID and type-specific properties.

Classes:
    IdGenerator: Generates unique IDs for components

Functions:
    parse_components: Create multiple components of specified type
    parse_config: Create all simulation components from config
    build_simulation_from_config: Build simulation from YAML config file

Registry:
    FACTORIES: Mapping of component types to their factory instances
"""

from typing import Any, Dict, List, Optional, Union

from ..config.manager import load_simulation_config
from ..engine.simulation import Simulation
from ..factories.simulation_factories import (
    DetectorFactory,
    OpticFactory,
    WavefrontFactory,
    SampleFactory
)
from ..factories.operator_factories import (
    PropagatorFactory,
    InteractorFactory
)
from ..factories.optimization_factories import (
    SolverFactory,
    ObjectiveFactory
)
from ..factories.experiment_factories import ExperimentFactory


FACTORIES = {
    "wavefront": WavefrontFactory(),
    "optic": OpticFactory(),
    "sample": SampleFactory(),
    "detector": DetectorFactory(),
    "interactor": InteractorFactory(),
    "propagator": PropagatorFactory(),
    "objective": ObjectiveFactory(),
    "solver": SolverFactory(),
    "experiment": ExperimentFactory(),
}


class IdGenerator:
    """Generates unique IDs for simulation components."""

    def __init__(self):
        """Initialize with empty counter dictionary."""
        self.counters: Dict[str, int] = {}

    def generate_id(self, component_type: str, custom_id: Optional[Any] = None) -> str:
        """Generate unique ID for component."""
        if custom_id is not None:
            return str(custom_id)

        if component_type not in self.counters:
            self.counters[component_type] = 0

        self.counters[component_type] += 1
        return f"{component_type}_{self.counters[component_type]}"


def parse_components(
    config: Union[Dict[str, Any], List[Dict[str, Any]]], 
    component_type: str, 
    id_gen: IdGenerator
) -> Union[Any, List[Any]]:
    """Create one or more components of specified type."""
    factory = FACTORIES.get(component_type)
    if not factory:
        raise ValueError(f"Unknown component type: {component_type}")

    # Handle single component case
    if isinstance(config, dict):
        component_id = id_gen.generate_id(component_type, config.get("id"))
        # Default to empty dict if properties not specified
        properties = config.get("properties", {})
        if properties is None:  # Handle explicit None case
            properties = {}
        return factory.create(component_id, config["type"], properties)

    # Handle multiple components case
    components = []
    for cfg in config:
        component_id = id_gen.generate_id(component_type, cfg.get("id"))
        # Default to empty dict if properties not specified
        properties = cfg.get("properties", {})
        if properties is None:  # Handle explicit None case
            properties = {}
        component = factory.create(component_id, cfg["type"], properties)
        components.append(component)
    return components


def parse_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create all simulation components from config dictionary."""
    id_gen = IdGenerator()
    
    components = {
        "wavefront": parse_components(config["simulation"]["wavefront"], "wavefront", id_gen),
        "detector": parse_components(config["simulation"]["detector"], "detector", id_gen),
        "propagator": parse_components(config["operator"]["propagator"], "propagator", id_gen),
        "interactor": parse_components(config["operator"]["interactor"], "interactor", id_gen),
        "optic": parse_components(config["simulation"]["optic"], "optic", id_gen),
        "sample": parse_components(config["simulation"]["sample"], "sample", id_gen),
        "workflow": config["experiment"]
    }
    
    return components


def build_simulation_from_config(config_path: str) -> Simulation:
    """Build Simulation from YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configured simulation instance
    """
    config = load_simulation_config(config_path)
    components = parse_config(config)
    return Simulation(**components)
