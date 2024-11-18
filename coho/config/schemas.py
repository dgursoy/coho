# config/schemas.py

"""Schema registry for configuration validation.

This module manages the registration and access of schema definitions used
for validating simulation configurations.

Functions:
    register_schemas: Load and register all component schemas
    register_schema: Register a single schema file
    get_schema: Get registered schema by name

Schema Files:
    - operator.yaml: Operator component schemas
    - optimization.yaml: Optimization component schemas
    - simulation.yaml: Simulation component schemas
    - experiment.yaml: Experiment component schemas
"""

from typing import Dict, Optional
from pathlib import Path
from .types import SchemaDict, SchemaRegistry
from .reader import read_config

# Schema registry
_SCHEMAS: SchemaRegistry = {}

# Define paths to individual schema files
_SCHEMA_DIR = Path(__file__).resolve().parent / 'schemas'
_SCHEMA_PATHS: Dict[str, Path] = {
    'operator': _SCHEMA_DIR / 'operator.yaml',
    'optimization': _SCHEMA_DIR / 'optimization.yaml', 
    'simulation': _SCHEMA_DIR / 'simulation.yaml',
    'experiment': _SCHEMA_DIR / 'experiment.yaml'
}

def register_schema(name: str, schema_path: Path) -> None:
    """Register a single schema definition.
    
    Args:
        name: Name to register schema under
        schema_path: Path to schema YAML file

    Raises:
        FileNotFoundError: If schema file is missing
        ValueError: If schema is empty or invalid
        yaml.YAMLError: If schema file has invalid YAML syntax
    """
    raw_schema = read_config(str(schema_path))
    if not raw_schema or name not in raw_schema:
        raise ValueError(f"Invalid schema at {schema_path}")
    
    # Store just the schema definition, not the whole file
    _SCHEMAS[name] = raw_schema[name]['schema']

def register_schemas() -> None:
    """Register all component schemas from schema directory.
    
    This function loads all predefined schema files from the schemas
    directory and registers them in the schema registry.

    Raises:
        FileNotFoundError: If schema directory is missing
        ValueError: If any schema file is invalid
    """
    for name, path in _SCHEMA_PATHS.items():
        register_schema(name, path)

def get_schema(name: str) -> Optional[SchemaDict]:
    """Get a registered schema by name.
    
    Args:
        name: Name of schema to retrieve

    Returns:
        SchemaDict if schema exists, None otherwise
    """
    return _SCHEMAS.get(name)