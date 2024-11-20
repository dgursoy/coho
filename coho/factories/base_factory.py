"""Base factory for component creation.

This module provides an abstract base factory for creating components.

Classes:
    ComponentFactory: Creates component instances from type and properties.
"""

from abc import ABC
from typing import TypeVar, Generic, Type
from .registry import MODEL_REGISTRY

__all__ = ['ComponentFactory']

# Define generic types for better type hints
P = TypeVar('P')  # Properties type
T = TypeVar('T')  # Component type

class ComponentFactory(ABC, Generic[P, T]):
    """Abstract base factory for creating component instances.
    
    Attributes:
        domain: The domain of the component.
        component_type: The type of the component.
    """
    
    def __init__(self, domain: str, component_type: str):
        """Initialize with domain and component type.
        """
        self.domain = domain
        self.component_type = component_type
    
    def create(self, model: str, properties: P) -> T:
        """Create a component instance."""
        component_class = MODEL_REGISTRY[self.domain][self.component_type][model.lower()]
        return component_class(properties) 