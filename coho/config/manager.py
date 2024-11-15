# config/manager.py

"""Configuration validation for simulation settings.

This module validates simulation configurations against predefined schemas
for each component (workflow, wavefront, elements, etc.).

Functions:
    load_simulation_config: Load and validate simulation config file
    load_all_schemas: Load and merge all component schemas
    load_schema: Load a single schema file

Components validated:
    workflow: Simulation workflow settings
    wavefront: Initial wavefront parameters
    optic: Optical element configurations
    sample: Sample configurations
    propagator: Propagation method settings
    detector: Detector specifications
    interactor: Interaction configurations

Constants:
    SCHEMA_DIR: Directory containing schema files
    SCHEMA_PATHS: Mapping of component names to schema files
"""

from typing import Dict, Optional
from cerberus import Validator
from pathlib import Path
from .loader import load_config

# Define paths to individual schema files
SCHEMA_DIR = Path(__file__).resolve().parent.parent / 'resources/schemas'
SCHEMA_PATHS = {
    'operator': SCHEMA_DIR / 'operator.yaml',
    'optimization': SCHEMA_DIR / 'optimization.yaml',
    'simulation': SCHEMA_DIR / 'simulation.yaml',
    'experiment': SCHEMA_DIR / 'experiment.yaml'
}


def load_schema(schema_path: Path) -> Dict:
    """Load a single schema file.

    Args:
        schema_path: Path to the schema YAML file.

    Returns:
        Parsed schema definition.

    Raises:
        FileNotFoundError: If schema file is missing or empty.
        yaml.YAMLError: If schema parsing fails.
    """
    schema = load_config(str(schema_path))
    if schema is None:
        raise FileNotFoundError(f"Schema file not found or empty at {schema_path}")
    return schema

def load_all_schemas() -> Dict:
    """Load and merge all component schemas.

    Returns:
        Combined schema for all components.
        
    Raises:
        FileNotFoundError: If any schema file is missing.
        yaml.YAMLError: If any schema fails to parse.
    """
    master_schema = {}
    for key, path in SCHEMA_PATHS.items():
        schema = load_schema(path)
        master_schema[key] = schema[key]
    return master_schema

def load_simulation_config(config_path: str) -> Dict:
    """Load and validate a simulation configuration file.

    Validates configuration against schemas for all components
    (workflow, wavefront, elements, etc.).

    Args:
        config_path: Path to the simulation config file.

    Returns:
        Validated simulation configuration.

    Raises:
        FileNotFoundError: If config file is missing or empty.
        ValueError: If validation fails, with detailed errors.
        yaml.YAMLError: If config parsing fails.
    """
    config = load_config(config_path)
    
    if config is None:
        raise FileNotFoundError(f"Configuration file not found or empty at {config_path}")
    
    master_schema = load_all_schemas()
    validator = Validator(master_schema)
    if not validator.validate(config):
        raise ValueError(f"Invalid simulation configuration: {validator.errors}")
    
    return config
