# config/__init__.py

"""Configuration management for optical simulations.

This package handles config loading, validation, and
component creation.

Modules:
    loader: File loading utilities
        load_config: Load YAML/JSON files

    manager: Schema validation
        load_simulation_config: Validate against schemas

    parser: Component creation
        build_simulation_from_config: Create simulation
"""

from .loader import load_config
from .manager import load_simulation_config
from .parser import build_simulation_from_config

__all__ = [
    'load_config',
    'load_simulation_config',
    'build_simulation_from_config'
]
