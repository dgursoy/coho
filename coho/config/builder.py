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

from typing import Dict, Tuple, Union, TypeAlias, Any
from .models.simulation import SimulationConfig
from .models.operator import OperatorConfig
from .models.experiment import ExperimentConfig
from .models.optimization import OptimizationConfig

BuildResult: TypeAlias = Tuple[bool, Union[Any, str]]
BuildResults: TypeAlias = Dict[str, BuildResult]
ConfigDict: TypeAlias = Dict[str, Any]

def build_section(section_name: str, config: ConfigDict) -> BuildResult:
    """Build a configuration section object."""
    model_mapping = {
        'simulation': SimulationConfig,
        'operator': OperatorConfig,
        'experiment': ExperimentConfig,
        'optimization': OptimizationConfig
    }
    
    model_class = model_mapping.get(section_name)
    if not model_class:
        return False, f"No model found for section: {section_name}"

    try:
        return True, model_class(**config)
    except Exception as e:
        return False, str(e)

def build_config(config: Dict[str, ConfigDict]) -> BuildResults:
    """Build complete configuration object."""
    return {
        section: build_section(section, section_config)
        for section, section_config in config.items()
    }