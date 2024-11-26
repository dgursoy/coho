# config/loader.py

"""Configuration loader."""

from pathlib import Path
from typing import Union, Dict, Any

from .models import RootConfig

from .reader import read_config
from .builder import build_config

__all__ = [
    'load_config',
]

def load_config(source: Union[str, Path, Dict[str, Any]]) -> RootConfig:
    """Load configuration from source."""
    raw_config = read_config(source)
    config_objects = build_config(raw_config)
    return config_objects