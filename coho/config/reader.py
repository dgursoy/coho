# config/reader.py

"""Configuration file reading utilities.

This module provides a unified interface for reading configuration files in
multiple formats (YAML, JSON, TOML).

Functions:
    read_config: Read configuration from any supported format
    read_yaml: Read and parse YAML configuration files
    read_json: Read and parse JSON configuration files
    read_toml: Read and parse TOML configuration files

Supported Formats:
    - YAML (.yaml, .yml)
    - JSON (.json)
    - TOML (.toml)
"""

import yaml
import json
import tomli
import os
from .types import ConfigContent, EncodingType

__all__ = [
    'read_config',
    'read_yaml',
    'read_json',
    'read_toml',
    'SUPPORTED_FORMATS'
]

def read_yaml(file_path: str, encoding: EncodingType = 'utf-8') -> ConfigContent:
    """Read YAML configuration.

    Args:
        file_path: Path to YAML file
        encoding: File encoding (default: utf-8)

    Returns:
        Parsed content

    Raises:
        FileNotFoundError: File missing
        yaml.YAMLError: Parse error with file path info
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            content = yaml.safe_load(file)
            if content is None:
                raise yaml.YAMLError(f"Empty or invalid YAML file: {file_path}")
            return content
    except yaml.YAMLError as e:
        # Include the problematic file path in the error
        raise yaml.YAMLError(f"YAML parse error in {file_path}: {str(e)}") from e

def read_json(file_path: str, encoding: EncodingType = 'utf-8') -> ConfigContent:
    """Read JSON configuration.

    Args:
        file_path: Path to JSON file
        encoding: File encoding (default: utf-8)

    Returns:
        Parsed content

    Raises:
        FileNotFoundError: File missing
        json.JSONDecodeError: Parse error
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON parse error: {file_path}") from e

def read_toml(file_path: str) -> ConfigContent:
    """Read TOML configuration.

    Args:
        file_path: Path to TOML file

    Returns:
        Parsed content

    Raises:
        FileNotFoundError: File missing
        tomli.TOMLDecodeError: Parse error
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TOML file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as file:
            return tomli.load(file)
    except tomli.TOMLDecodeError as e:
        raise tomli.TOMLDecodeError(f"TOML parse error: {file_path}") from e

SUPPORTED_FORMATS = {
    '.yaml': read_yaml,
    '.yml': read_yaml,
    '.json': read_json,
    '.toml': read_toml
}

def read_config(file_path: str, encoding: EncodingType = 'utf-8') -> ConfigContent:
    """Read configuration from any supported format.

    Args:
        file_path: Path to configuration file
        encoding: File encoding (default: utf-8)

    Returns:
        Parsed content

    Raises:
        ValueError: Unsupported file format
        FileNotFoundError: File missing
        Various parsing errors depending on format
    """
    _, ext = os.path.splitext(file_path)
    
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
        )
    
    # Pass encoding only to functions that accept it
    if ext == '.toml':
        return SUPPORTED_FORMATS[ext](file_path)
    return SUPPORTED_FORMATS[ext](file_path, encoding)
