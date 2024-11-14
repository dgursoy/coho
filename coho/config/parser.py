# config/parser.py

"""Component creation and simulation assembly.

This module converts configuration files into initialized
simulation components using factory patterns.

Classes:
    IdGenerator: Create unique IDs for components

Functions:
    create_standard_component: Create basic components
    parse_elements: Create optical element set
    parse_config: Create all simulation components
    build_simulation_from_config: Create simulation from file

Dependencies:
    factories:
        WavefrontFactory: Create wavefront profiles
        DetectorFactory: Create detector instances
        ElementFactory: Create optical elements
        PropagatorFactory: Create propagation methods
        InteractorFactory: Create interaction handlers
"""

from typing import List, Dict, Any, Optional
from .manager import load_simulation_config
from ..factories import (
    WavefrontFactory,
    DetectorFactory, 
    ElementFactory,
    PropagatorFactory, 
    InteractorFactory
)
from ..engine.simulation import Simulation


class IdGenerator:
    """Manages unique ID generation for components."""
    
    def __init__(self):
        self.used_ids = set()
    
    def generate_id(self, base: str, config_id: str = None) -> str:
        """Generate unique component ID.

        Args:
            base: Base name for component type
            config_id: Optional ID from config

        Returns:
            Unique ID string
        """
        # Use config ID if provided and unique
        if config_id and config_id not in self.used_ids:
            self.used_ids.add(config_id)
            return config_id
            
        # Generate new unique ID
        counter = 1
        while True:
            new_id = f"{base}_{counter}"
            if new_id not in self.used_ids:
                self.used_ids.add(new_id)
                return new_id
            counter += 1


def parse_component(config: Dict[str, Any], component_type: str, id_gen: IdGenerator, **kwargs) -> Any:
    """Create single component from config.

    Args:
        config: Component configuration
        component_type: Type of component to create
        id_gen: ID generator instance
        **kwargs: Additional factory arguments

    Returns:
        Initialized component
    """
    # Generate unique ID
    component_id = id_gen.generate_id(component_type, config.get("id"))
    properties = config.get("properties", {})
    
    # Select factory and handle special cases
    if component_type == "interactor":
        return InteractorFactory.create_interactor(
            component_id, 
            config["type"], 
            kwargs["wavefront"]  # Direct wavefront argument
        )
    
    # Handle standard factories
    factories = {
        "wavefront": WavefrontFactory.create_wavefront,
        "propagator": PropagatorFactory.create_propagator,
        "detector": DetectorFactory.create_detector,
    }
    
    factory = factories.get(component_type)
    if not factory:
        raise ValueError(f"Unknown component type: {component_type}")

    return factory(component_id, config["type"], properties)


def parse_elements(configs: List[Dict[str, Any]], id_gen: IdGenerator) -> List[Any]:
    """Create multiple optical elements."""
    elements = []
    for config in configs:
        element_id = id_gen.generate_id("element", config.get("id"))
        properties = config.get("properties", {})
        element = ElementFactory.create_element(
            element_id, 
            config["type"],
            properties
        )
        elements.append(element)
    return elements


def parse_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create all simulation components."""
    # Initialize ID generator
    id_gen = IdGenerator()
    
    # Create components in dependency order
    wavefront = parse_component(config["wavefront"], "wavefront", id_gen)
    interactor = parse_component(
        config["interactor"], 
        "interactor", 
        id_gen,
        wavefront=wavefront
    )
    propagator = parse_component(config["propagator"], "propagator", id_gen)
    elements = parse_elements(config["elements"], id_gen)
    detector = parse_component(config["detector"], "detector", id_gen)

    return {
        "wavefront": wavefront,
        "propagator": propagator,
        "elements": elements,
        "detector": detector,
        "interactor": interactor,
        "workflow": config["workflow"]
    }

def build_simulation_from_config(config_path: str) -> Simulation:
    """Create a Simulation instance from config file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Initialized simulation with all components.
    """
    config = load_simulation_config(config_path)
    components = parse_config(config)
    return Simulation(**components)
