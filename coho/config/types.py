# config/types.py

"""Type definitions for configuration management.

This module defines type aliases and custom types used throughout the configuration
package for improved type safety and code clarity.

Type Aliases:
    ConfigContent: Raw configuration content from files
    EncodingType: File encoding specification
    ConfigDict: Basic configuration dictionary
    SchemaDict: Schema definition dictionary
    SchemaRegistry: Registry mapping names to schemas
    ValidationErrors: List of validation error messages
    ValidationResult: Validation result tuple (success, errors)
    BuildResult: Build result tuple (success, model/error)
    BuildResults: Dictionary of build results by section
"""

from typing import Dict, List, Union, TypeAlias, Any, Optional
from pydantic import BaseModel

# Reader types
ConfigContent: TypeAlias = Union[Dict[str, Any], List[Any]]

# Encoding types
EncodingType: TypeAlias = Optional[str]

# Basic configuration types
ConfigDict: TypeAlias = Dict[str, Any]
SchemaDict: TypeAlias = Dict[str, Any]
SchemaRegistry: TypeAlias = dict[str, SchemaDict]

# Validation types
ValidationErrors: TypeAlias = List[str]
ValidationResult: TypeAlias = tuple[bool, ValidationErrors]

# Build types
BuildResult: TypeAlias = tuple[bool, Union[BaseModel, str]]
BuildResults: TypeAlias = Dict[str, BuildResult]
 