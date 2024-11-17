# config/__init__.py

"""Configuration management for optical simulations.

This package provides a complete configuration system for optical simulations,
including file reading, schema validation, and object construction.

Modules:
    reader: Configuration file reading utilities
    schemas: Schema registry and management
    validator: Configuration validation
    builder: Configuration object construction
    types: Type definitions and aliases
    models: Pydantic models for configuration objects

Main Functions:
    read_config: Read configuration from supported file formats
    validate_config: Validate configuration against schemas
    build_config: Build configuration objects from validated data
"""

from .types import (
    ConfigDict,
    SchemaDict,
    SchemaRegistry,
    ValidationErrors,
    ValidationResult,
    BuildResult,
    BuildResults,
    ConfigContent,
    EncodingType,
)
from .reader import read_config
from .schemas import register_schemas, get_schema
from .validator import validate_config, validate_section
from .builder import (
    build_config,
    build_section,
)

__all__ = [
    # Types
    'ConfigDict',
    'SchemaDict',
    'SchemaRegistry',
    'ValidationErrors',
    'ValidationResult',
    'BuildResult',
    'BuildResults',
    'ConfigContent',
    'EncodingType',
    # Reader
    'read_config',
    # Schemas
    'register_schemas',
    'get_schema',
    # Validator
    'validate_config',
    'validate_section',
    # Builder
    'build_config',
    'build_section',
]
