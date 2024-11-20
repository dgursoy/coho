# config/builder.py

"""Configuration object builder using Pydantic models.

Constructs strongly-typed configuration objects from configuration data
using Pydantic models.
"""

from .models.base import Config
from .types import ConfigDict

def build_config(config: ConfigDict) -> Config:
    """Build configuration object."""
    return Config(**config)
