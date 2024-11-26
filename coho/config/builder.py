# config/builder.py

"""Configuration object builder using Pydantic models.

Constructs strongly-typed configuration objects from configuration data
using Pydantic models.
"""

from .models import RootConfig
from typing import Dict, Any


def build_config(config: Dict[str, Any]) -> RootConfig:
    """Build configuration object."""
    return RootConfig(**config)
