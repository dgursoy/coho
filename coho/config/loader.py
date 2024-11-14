# config/loader.py

"""Configuration file loading utilities.

This module provides unified loading interface for
multiple configuration formats.

Functions:
    load_config: Auto-detect and load any format
    load_yaml: Load YAML files
    load_json: Load JSON files
    load_toml: Load TOML files
    load_xml: Load XML files
"""

from typing import Any, Dict
import yaml
import json
import tomli
import xml.etree.ElementTree as ET
import os

def load_yaml(file_path: str) -> Any:
    """Load YAML configuration.

    Args:
        file_path: YAML file path

    Returns:
        Parsed content

    Raises:
        FileNotFoundError: File missing
        yaml.YAMLError: Parse error
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML parse error: {file_path}") from e

def load_json(file_path: str) -> Any:
    """Load JSON configuration.

    Args:
        file_path: JSON file path

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

def load_toml(file_path: str) -> Any:
    """Load TOML configuration.

    Args:
        file_path: TOML file path

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

def load_xml(file_path: str) -> Dict[str, Any]:
    """Load XML configuration.

    Converts to dictionary format:
    {
        "root": {
            "attributes": {},
            "text": "content",
            "children": []
        }
    }

    Args:
        file_path: XML file path

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

        def xml_to_dict(element):
            return {
                element.tag: {
                    'attributes': element.attrib,
                    'text': element.text.strip() if element.text else '',
                    'children': [xml_to_dict(child) for child in element]
                }
            }

        return xml_to_dict(root)
    except ET.ParseError as e:
        raise ET.ParseError(f"XML parse error: {file_path}") from e

def load_config(file_path: str) -> Any:
    """Load configuration from any supported format.

    Args:
        file_path: Config file path

    Returns:
        Parsed content

    Raises:
        ValueError: Unsupported format
        FileNotFoundError: File missing
        Various format-specific parse errors
    """
    _, ext = os.path.splitext(file_path)
    
    if ext in ['.yaml', '.yml']:
        return load_yaml(file_path)
    elif ext == '.json':
        return load_json(file_path)
    elif ext == '.toml':
        return load_toml(file_path)
    elif ext == '.xml':
        return load_xml(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")