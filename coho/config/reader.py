"""Configuration file reading utilities.

This module provides a unified interface for reading configuration files in
multiple formats (YAML, JSON, TOML).

Public Functions:
    read_config: Read configuration from any supported format

Supported Formats:
    - YAML (.yaml, .yml)
    - JSON (.json)
    - TOML (.toml)
"""

from os import PathLike
from pathlib import Path
import yaml
import json
import tomli
import os
from typing import Literal, Any, Dict

__all__ = [
    'read_config',
]

# Supported encodings for file reading
EncodingType = Literal['utf-8', 'ascii', 'latin-1']

def read_config(file_path: PathLike, encoding: EncodingType = 'utf-8') -> Dict[str, Any]:
    """Read configuration from a file in any supported format.
    
    Args:
        file_path: Path to configuration file
        encoding: File encoding (default: utf-8)

    Returns:
        Dict[str, Any]: Parsed configuration content in a dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
        yaml.YAMLError: If YAML parsing fails
        json.JSONDecodeError: If JSON parsing fails
        tomli.TOMLDecodeError: If TOML parsing fails

    Supported Formats:
        - YAML (.yaml, .yml)
        - JSON (.json)
        - TOML (.toml)
    """
    # Convert PathLike to string if necessary
    path = Path(file_path)
    
    # Extract the file extension
    ext = path.suffix.lower()
    
    # Map extensions to reader functions
    supported_formats = {
        '.yaml': _read_yaml,
        '.yml': _read_yaml,
        '.json': _read_json,
        '.toml': _read_toml
    }
    
    # Raise error if format is not supported
    if ext not in supported_formats:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Supported formats: {list(supported_formats.keys())}"
        )
    
    # Pass encoding only to functions that accept it
    if ext == '.toml':
        return supported_formats[ext](path)
    return supported_formats[ext](path, encoding)

def _read_yaml(file_path: str, encoding: EncodingType = 'utf-8') -> Dict[str, Any]:
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

def _read_json(file_path: str, encoding: EncodingType = 'utf-8') -> Dict[str, Any]:
    """Read JSON configuration.

    Args:
        file_path: Path to JSON file
        encoding: File encoding (default: utf-8)
    Returns:
        Parsed content

    Raises:
        FileNotFoundError: File missing
        json.JSONDecodeError: Parse error with file path info
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON parse error: {file_path}") from e

def _read_toml(file_path: str) -> Dict[str, Any]:
    """Read TOML configuration.

    Args:
        file_path: Path to TOML file

    Returns:
        Parsed content

    Raises:
        FileNotFoundError: File missing
        tomli.TOMLDecodeError: Parse error with file path info
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TOML file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as file:
            return tomli.load(file)
    except tomli.TOMLDecodeError as e:
        raise tomli.TOMLDecodeError(f"TOML parse error: {file_path}") from e
