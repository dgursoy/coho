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

from typing import Any, Dict, List, Optional

from ..config.manager import load_simulation_config
from ..engine.simulation import Simulation
from ..factories.detector_factory import DetectorFactory
from ..factories.element_factory import ElementFactory
from ..factories.interactor_factory import InteractorFactory
from ..factories.propagator_factory import PropagatorFactory
from ..factories.wavefront_factory import WavefrontFactory


FACTORIES = {
    "detector": DetectorFactory(),
    "elements": ElementFactory(),
    "interactor": InteractorFactory(),
    "propagator": PropagatorFactory(),
    "wavefront": WavefrontFactory(),
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
    configs: List[Dict[str, Any]], 
    component_type: str, 
    id_gen: IdGenerator
) -> List[Any]:
    """Create multiple components of specified type.
    
    Args:
        configs: List of component configurations
        component_type: Type of components to create
        id_gen: ID generator for component identification

    Returns:
        List of initialized components

    Raises:
        ValueError: If component type is not supported
    """
    factory = FACTORIES.get(component_type)
    print (component_type)
    if not factory:
        raise ValueError(f"Unknown component type: {component_type}")

    components = []
    for config in configs:
        component_id = id_gen.generate_id(component_type, config.get("id"))
        properties = config.get("properties", {})
        component = factory.create(component_id, config["type"], properties)
        components.append(component)
        
    return components


def parse_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create all simulation components from config dictionary."""
    id_gen = IdGenerator()
    
    # Ensure single components are in list format
    component_configs = {
        "wavefront": [config["wavefront"]],
        "propagator": [config["propagator"]],
        "elements": config["elements"],
        "detector": [config["detector"]],
        "interactor": [config["interactor"]]
    }
    
    # Create all components using the same parser
    components = {
        name: parse_components(configs, name, id_gen)[0] if name != "elements" 
        else parse_components(configs, name, id_gen)
        for name, configs in component_configs.items()
    }
    
    # Add workflow configuration
    components["workflow"] = config["workflow"]
    
    return components


def build_simulation_from_config(config_path: str) -> Simulation:
    """Build simulation from YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configured simulation instance
    """
    config = load_simulation_config(config_path)
    components = parse_config(config)
    return Simulation(**components)
