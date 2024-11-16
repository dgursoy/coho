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

from dataclasses import dataclass
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

@dataclass
class ValidationResult:
    """Represents the result of a configuration validation."""
    is_valid: bool
    errors: List[str]

class ConfigValidator:
    """Handles validation of configuration sections."""
    
    @staticmethod
    def validate_simulation(config: Dict[str, Any]) -> ValidationResult:
        errors = []
        if "simulation" not in config:
            return ValidationResult(False, ["Missing 'simulation' section"])
        
        sim_config = config["simulation"]
        for component in {"wavefront", "detector"}:
            if component not in sim_config:
                errors.append(f"Missing required component: {component}")
                
        return ValidationResult(len(errors) == 0, errors)

    @staticmethod
    def validate_operator(config: Dict[str, Any]) -> ValidationResult:
        errors = []
        if "operator" not in config:
            return ValidationResult(False, ["Missing 'operator' section"])
            
        op_config = config["operator"]
        for operator in {"propagator", "interactor"}:
            if operator not in op_config:
                errors.append(f"Missing required operator: {operator}")
                
        return ValidationResult(len(errors) == 0, errors)

    @staticmethod
    def validate_optimization(config: Dict[str, Any]) -> ValidationResult:
        errors = []
        if "optimization" not in config:
            return ValidationResult(True, [])  # Optional section
            
        opt_config = config["optimization"]
        if "solver" in opt_config:
            if "type" not in opt_config["solver"]:
                errors.append("Solver missing required 'type' field")
                
        if "objective" in opt_config:
            if "type" not in opt_config["objective"]:
                errors.append("Objective missing required 'type' field")
                
        return ValidationResult(len(errors) == 0, errors)

    @staticmethod
    def validate_experiment(config: Dict[str, Any]) -> ValidationResult:
        errors = []
        if "experiment" not in config:
            return ValidationResult(False, ["Missing 'experiment' section"])
            
        exp_config = config["experiment"]
        if not isinstance(exp_config, list):
            return ValidationResult(False, ["Experiment must be a list of steps"])
            
        for idx, step in enumerate(exp_config):
            if not isinstance(step, dict):
                errors.append(f"Step {idx}: Must be a dictionary")
            elif "component_id" not in step:
                errors.append(f"Step {idx}: Missing required 'component_id'")
                
        return ValidationResult(len(errors) == 0, errors)

    @classmethod
    def validate_all(cls, config: Dict[str, Any]) -> None:
        """Validate complete configuration and raise exception if invalid."""
        validators = [
            cls.validate_simulation,
            cls.validate_operator,
            cls.validate_optimization,
            cls.validate_experiment
        ]
        
        all_errors = []
        for validator in validators:
            result = validator(config)
            if not result.is_valid:
                all_errors.extend(result.errors)
        
        if all_errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(all_errors))

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

    if isinstance(config, dict):
        component_id = id_gen.generate_id(component_type, config.get("id"))
        properties = config.get("properties", {}) or {}
        return factory.create(component_id, config["type"], properties)

    components = []
    for cfg in config:
        component_id = id_gen.generate_id(component_type, cfg.get("id"))
        properties = cfg.get("properties", {}) or {}
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
    ConfigValidator.validate_all(config)
    components = parse_config(config)
    return Simulation(**components)
