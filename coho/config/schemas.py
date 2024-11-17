# config/schemas.py

"""Schema registry for configuration validation.

This module manages the registration and access of schema definitions
used for validating simulation configurations.

Functions:
    register_schemas: Load and register all component schemas
    register_schema: Register a single schema file
    get_schema: Get registered schema by name

Constants:
    SCHEMA_DIR: Directory containing schema files
    SCHEMA_PATHS: Mapping of component names to schema files

Type Aliases:
    SchemaDict: Dict[str, Dict]
        Dictionary mapping schema names to their definitions
"""

from typing import Dict, Optional, TypeAlias
from pathlib import Path
from .reader import read_config

# Type aliases
SchemaDict: TypeAlias = Dict[str, Dict]

# Schema registry
_SCHEMAS: SchemaDict = {}

# Define paths to individual schema files
SCHEMA_DIR = Path(__file__).resolve().parent / 'schemas'
SCHEMA_PATHS: Dict[str, Path] = {
    'operator': SCHEMA_DIR / 'operator.yaml',
    'optimization': SCHEMA_DIR / 'optimization.yaml', 
    'simulation': SCHEMA_DIR / 'simulation.yaml',
    'experiment': SCHEMA_DIR / 'experiment.yaml'
}

def register_schema(name: str, schema_path: Path) -> None:
    """Register a single schema definition.

    Args:
        name: Name to register schema under
        schema_path: Path to schema YAML file

    Raises:
        FileNotFoundError: If schema file is missing
        ValueError: If schema is empty or invalid
    """
    raw_schema = read_config(str(schema_path))
    if not raw_schema or name not in raw_schema:
        raise ValueError(f"Invalid schema at {schema_path}")
    
    # Store just the schema definition, not the whole file
    _SCHEMAS[name] = raw_schema[name]['schema']

def register_schemas() -> None:
    """Load and register all component schemas.

    Raises:
        FileNotFoundError: If any schema file is missing
        ValueError: If any schema is empty or invalid
    """
    for name, path in SCHEMA_PATHS.items():
        register_schema(name, path)

def get_schema(name: str) -> Optional[Dict]:
    """Get a registered schema by name.
    
    Args:
        name: Name of schema to retrieve
        
    Returns:
        Schema definition if found, None otherwise
    """
    return _SCHEMAS.get(name)