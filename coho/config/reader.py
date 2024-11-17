# config/reader.py

"""Configuration file reading utilities.

This module provides unified reading interface for
multiple configuration formats.

Functions:
    read_config: Auto-detect and read any format
    read_yaml: Read YAML files
    read_json: Read JSON files
    read_toml: Read TOML files
    read_xml: Read XML files

Constants:
    SUPPORTED_FORMATS: Dictionary mapping file extensions to their reader functions

Type Aliases:
    ConfigDict: Dict[str, Any]
        Dictionary containing configuration data
    ConfigContent: Union[Dict, list]
        Parsed configuration content (either dict or list)
"""

from typing import Any, Dict, Optional, Union, TypeAlias
import yaml
import json
import tomli
import xml.etree.ElementTree as ET
import os

__all__ = [
    'read_config',
    'read_yaml',
    'read_json',
    'read_toml',
    'read_xml',
    'SUPPORTED_FORMATS',
    'ConfigError',
]

# Type aliases
ConfigDict: TypeAlias = Dict[str, Any]  # type: TypeAlias
ConfigContent: TypeAlias = Union[Dict, list]  # type: TypeAlias

def read_yaml(file_path: str, encoding: Optional[str] = 'utf-8') -> Union[Dict, list]:
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

def read_json(file_path: str, encoding: Optional[str] = 'utf-8') -> Union[Dict, list]:
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

def read_toml(file_path: str) -> Dict[str, Any]:
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

def read_xml(file_path: str, encoding: Optional[str] = 'utf-8') -> Dict[str, Any]:
    """Read XML configuration.

    Converts to dictionary format:
    {
        "root": {
            "attributes": {},
            "text": "content",
            "children": []
        }
    }

    Args:
        file_path: Path to XML file
        encoding: File encoding (default: utf-8)

    Returns:
        Dict representation

    Raises:
        FileNotFoundError: File missing
        ET.ParseError: Parse error
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XML file not found: {file_path}")

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        def xml_to_dict(element: ET.Element) -> Dict[str, Any]:
            """Convert XML element to dictionary representation."""
            text = ''.join(element.itertext()).strip()
            result = {
                'attributes': element.attrib,
                'text': text,
                'children': [xml_to_dict(child) for child in element]
            }
            return {element.tag: result}

        return xml_to_dict(root)
    except ET.ParseError as e:
        raise ET.ParseError(f"XML parse error: {file_path}") from e

SUPPORTED_FORMATS = {
    '.yaml': read_yaml,
    '.yml': read_yaml,
    '.json': read_json,
    '.toml': read_toml,
    '.xml': read_xml
}

def read_config(file_path: str, encoding: Optional[str] = 'utf-8') -> Union[Dict, list]:
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
