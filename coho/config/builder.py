# config/builder.py

"""Configuration object builder using Pydantic models.

This module constructs strongly-typed configuration objects from validated
configuration data using Pydantic models.

Functions:
    build_config: Build complete configuration object
    build_section: Build configuration section object

The builder maps configuration sections to corresponding Pydantic models
and handles any conversion errors during object construction.
"""

from .types import ConfigDict, BuildResult, BuildResults
from .models.simulation import SimulationConfig
from .models.operator import OperatorConfig
from .models.experiment import ExperimentConfig
from .models.optimization import OptimizationConfig

__all__ = [
    'build_config',
    'build_section',
]

def get_model_mapping() -> dict:
    """Returns the mapping of configuration sections to Pydantic models."""
    return {
        'simulation': SimulationConfig,
        'operator': OperatorConfig,
        'experiment': ExperimentConfig,
        'optimization': OptimizationConfig,
    }

def build_section(section_name: str, config: ConfigDict) -> BuildResult:
    """Build a configuration section object.
    
    Args:
        section_name: Name of the configuration section
        config: Configuration dictionary for the section

    Returns:
        BuildResult: Tuple of (success, result) where:
            - success: Boolean indicating if build succeeded
            - result: Either Pydantic model if successful, or error string if failed

    Raises:
        None: Exceptions are caught and returned as error strings
    """
    model_mapping = get_model_mapping()
    model_class = model_mapping.get(section_name)
    
    if not model_class:
        return False, f"No model found for section: {section_name}"

    try:
        return True, model_class(**config)
    except Exception as e:
        return False, str(e)

def build_config(config: ConfigDict) -> BuildResults:
    """Build complete configuration object from all sections.
    
    Args:
        config: Dictionary containing all configuration sections

    Returns:
        BuildResults: Dictionary mapping section names to BuildResult tuples

    Example:
        >>> config = {'simulation': {...}, 'operator': {...}}
        >>> results = build_config(config)
        >>> simulation_success, simulation_model = results['simulation']
    """
    return {
        section: build_section(section, section_config)
        for section, section_config in config.items()
    }