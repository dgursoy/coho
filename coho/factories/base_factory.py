"""Base factory for component creation."""

from abc import ABC
from typing import TypeVar, Generic, Type, Dict

P = TypeVar('P')  # Properties type
T = TypeVar('T')  # Component type

class ComponentFactory(ABC, Generic[P, T]):
    """Abstract base factory for creating component instances."""
    
    def __init__(self):
        """Initialize the component mapping."""
        self._components: Dict[str, Type[T]] = {}
    
    def register(self, name: str, component_class: Type[T]) -> None:
        """Register a component class."""
        self._components[name.lower()] = component_class
    
    def create(self, model: str, properties: P) -> T:
        """Create a component instance."""
        component_class = self._components.get(model.lower())
        if component_class is None:
            raise ValueError(f"Unknown model: {model}")
        return component_class(properties) 