# config/builder.py

"""Configuration object builder using Pydantic models.

This module constructs typed configuration objects from validated
configuration data using Pydantic models.

Functions:
    build_config: Build complete configuration object
    build_section: Build configuration section object

Constants:
    MODEL_MAPPING: Dictionary mapping section names to their Pydantic models

Type Aliases:
    BuildResult: Tuple[bool, Union[Any, str]]
        Tuple containing build success and either model or error message
    BuildResults: Dict[str, BuildResult]
        Dictionary mapping section names to their build results
    ConfigDict: Dict[str, Any]
        Dictionary containing configuration data
"""

from typing import Dict, Tuple, Type, Union, TypeAlias, Any
from pydantic import BaseModel, ValidationError

# Type aliases
BuildResult: TypeAlias = Tuple[bool, Union[Any, str]]  # type: TypeAlias
BuildResults: TypeAlias = Dict[str, BuildResult]  # type: TypeAlias
ConfigDict: TypeAlias = Dict[str, Any]  # type: TypeAlias

class SimulationConfig(BaseModel):
    """Simulation configuration settings."""
    steps: int
    seed: int
    # ... other fields ...

class OptimizationConfig(BaseModel):
    """Optimization configuration settings."""
    algorithm: str
    parameters: ConfigDict
    # ... other fields ...

# Map section names to their models
MODEL_MAPPING: Dict[str, Type[BaseModel]] = {
    'simulation': SimulationConfig,
    'optimization': OptimizationConfig,
    # ... other mappings ...
}

def build_section(section_name: str, config: ConfigDict) -> BuildResult:
    """Build a configuration section object.

    Args:
        section_name: Name of section to build
        config: Validated configuration dictionary for section

    Returns:
        Tuple of (success, result):
            - If success is True, returns (True, model)
            - If success is False, returns (False, error_message)
    """
    model_class = MODEL_MAPPING.get(section_name)
    if not model_class:
        return False, f"No model found for section: {section_name}"

    try:
        return True, model_class(**config)
    except ValidationError as e:
        return False, str(e)

def build_config(config: Dict[str, ConfigDict]) -> BuildResults:
    """Build complete configuration object.

    Args:
        config: Complete validated configuration dictionary

    Returns:
        Dictionary mapping section names to their build results
    """
    return {
        section: build_section(section, section_config)
        for section, section_config in config.items()
    }