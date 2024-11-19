"""Base factory for component creation.

This module provides an abstract base factory for creating components.

Classes:
    ComponentFactory: Creates component instances from type and properties.
"""

from abc import ABC
from typing import TypeVar, Generic, Type, Mapping, Optional

__all__ = ['ComponentFactory']

# Define generic types for better type hints
P = TypeVar('P')  # Properties type
T = TypeVar('T')  # Component type

class ComponentFactory(ABC, Generic[P, T]):
    """Abstract base factory for creating component instances.
    
    Attributes:
        component_types: Mapping of component names to their classes.
    """
    
    def __init__(self, component_types: Mapping[str, Type[T]]):
        """Initialize with component type mapping.
        """
        self.component_types = component_types
    
    def create(self, model: str, properties: P) -> T:
        """Create a component instance."""
        component_type = model.lower()
        
        if component_type not in self.component_types:
            valid_types = list(self.component_types.keys())
            raise ValueError(
                f"Unsupported type: {component_type}. "
                f"Valid types are: {valid_types}"
            )
            
        component_class = self.component_types[component_type]
        return component_class(properties) 