# config/validator.py

"""Configuration validation using registered schemas.

This module validates configuration files against their registered schemas
to ensure they meet the required structure and constraints.

Functions:
    validate_config: Validate complete configuration
    validate_section: Validate a specific configuration section

Types:
    ValidationResult: Tuple of (is_valid, error_messages)
    ConfigDict: Dictionary of named configuration sections
"""

from typing import Dict, List, Tuple, TypeAlias
from cerberus import Validator
from .schemas import get_schema

# Type aliases
ValidationResult: TypeAlias = Tuple[bool, List[str]]
ConfigDict: TypeAlias = Dict[str, Dict]

def validate_section(section_name: str, config: Dict) -> ValidationResult:
    """Validate a specific section of the configuration.

    Args:
        section_name: Name of section to validate
        config: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid, list of error messages)

    Raises:
        ValueError: If schema not found for section
    """
    schema = get_schema(section_name)
    if not schema:
        raise ValueError(f"No schema found for section: {section_name}")

    validator = Validator(schema)
    is_valid = validator.validate(config)
    
    return is_valid, validator.errors

def validate_config(config: ConfigDict) -> ValidationResult:
    """Validate complete configuration against all schemas.

    Args:
        config: Complete configuration dictionary with named sections

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    all_errors: List[str] = []
    
    for section_name in config:
        is_valid, errors = validate_section(section_name, config[section_name])
        if not is_valid:
            all_errors.append([
                f"{section_name}: {errors}"
            ])
    
    return not bool(all_errors), all_errors
