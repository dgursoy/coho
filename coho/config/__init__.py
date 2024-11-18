# config/__init__.py

"""Configuration management.

This package provides a complete configuration system for optical simulations,
including file reading, schema validation, and object construction.

Main Functions:
    load_config: Read, validate and build configuration from source
    read_config: Read raw configuration from supported file formats

Internal Modules:
    reader: Configuration file reading utilities
    schemas: Schema registry and management
    validator: Configuration validation
    builder: Configuration object construction
    loader: Configuration loading and building
    types: Type definitions and aliases
    models: Pydantic models for configuration objects
"""

from .loader import load_config   
from .reader import read_config  

__all__ = [
    'load_config',  
    'read_config',  
]