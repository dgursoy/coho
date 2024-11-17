# config/__init__.py

"""Configuration management for optical simulations.

This package handles configuration reading, validation,
and object construction.

Functions:
    read_config: Read configuration from file
    validate_config: Validate configuration against schemas
    build_config: Build configuration objects

Type Aliases:
    ConfigDict: Dictionary containing configuration data
    BuildResult: Build result with success status and model/error
    BuildResults: Dictionary of section build results

Constants:
    SCHEMA_DIR: Directory containing schema files
    SCHEMA_PATHS: Mapping of component names to schema files
    MODEL_MAPPING: Mapping of section names to Pydantic models
"""

from .reader import read_config, ConfigDict
from .schemas import register_schemas, get_schema
from .validator import validate_config, validate_section
from .builder import (
    build_config,
    build_section,
    BuildResult,
    BuildResults,
)

__all__ = [
    # Reader
    'read_config',
    'ConfigDict',
    # Schemas
    'register_schemas',
    'get_schema',
    # Validator
    'validate_config',
    'validate_section',
    # Builder
    'build_config',
    'build_section',
    'BuildResult',
    'BuildResults',
]
