# config/loader.py

"""Configuration loader.

This module handles loading, validating and building configuration objects
from various sources.
"""

from pathlib import Path
from typing import Union, Dict, Any

from .reader import read_config
from .schemas import register_schemas
from .validator import validate_config
from .builder import build_config

__all__ = [
    'load_config',
]

def load_config(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load configuration from source.
    
    Args:
        source: Configuration source (file path or dict)
        
    Returns:
        Dictionary of validated configuration objects
        
    Raises:
        ValueError: Invalid configuration
    """
    # Read raw config
    raw_config = read_config(source)
    
    # Register schemas
    schemas = register_schemas()

    # Validate config
    is_valid, errors = validate_config(raw_config, schemas)
    if not is_valid:
        raise ValueError(f"Invalid configuration: {errors}")

    # Build config objects
    config_objects = build_config(raw_config)
    errors = {name: obj for name, (success, obj) in config_objects.items() if not success}
    if errors:
        raise ValueError(
            "Failed to build configuration:\n" +
            "\n".join(f"Section '{name}': {error}" for name, error in errors.items())
        )

    # If no errors, return the successful objects directly
    return {name: obj for name, (_, obj) in config_objects.items()}