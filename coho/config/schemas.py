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

from typing import Dict
from pathlib import Path
from os import PathLike
from .reader import read_config
from .types import SchemaDict, SchemaRegistry


# Define directory containing schema files
_SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

def get_schema_paths(schema_dir: Path) -> Dict[str, Path]:
    """Generate schema paths dynamically from the given directory.

    Args:
        schema_dir: Path to the directory containing schema files

    Returns:
        A dictionary mapping schema names to file paths
    """
    return {
        path.stem: path for path in schema_dir.glob("*.yaml") if path.is_file()
    }

def get_schema(name: str, schemas: SchemaRegistry) -> SchemaDict | None:
    """Retrieve a registered schema by name.

    Args:
        name: Name of the schema to retrieve
        schemas: The dictionary of registered schemas

    Returns:
        The schema if found, otherwise None
    """
    return schemas.get(name)

def register_schema(name: str, schema_path: PathLike) -> SchemaDict:
    """Register a single schema definition.

    Args:
        name: Name to register the schema under
        schema_path: Path to schema YAML file

    Returns:
        The schema dictionary

    Raises:
        ValueError: If the schema is empty or invalid
    """
    print(f"Reading schema from {schema_path}")
    raw_schema = read_config(schema_path)
    if not raw_schema or name not in raw_schema:
        raise ValueError(f"Invalid schema at {schema_path}")
    
    return raw_schema[name]['schema']

def register_schemas() -> SchemaRegistry:
    """Register all component schemas.

    Args:
        schema_paths: A dictionary of schema names and their file paths

    Returns:
        A dictionary of registered schemas
    """
    schema_paths = get_schema_paths(_SCHEMA_DIR)
    schemas: SchemaRegistry = {}
    for name, path in schema_paths.items():
        schemas[name] = register_schema(name, path)
    return schemas