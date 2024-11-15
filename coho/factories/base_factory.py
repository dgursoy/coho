"""Base factory for component creation.

This module provides an abstract base factory for creating components.

Classes:
    ComponentFactory: Creates component instances from type and parameters.
"""

from abc import ABC
from typing import Dict, Any, Type, Mapping


class ComponentFactory(ABC):
    """Abstract base factory for creating component instances.
    
    Attributes:
        component_types: Mapping of component names to their classes.
    """
    
    def __init__(self, component_types: Mapping[str, Type]):
        """Initialize with component type mapping.
        
        Args:
            component_types: Map of component names to classes.
        """
        self.component_types = component_types
    
    def create(self, id: Any, type: str, parameters: Dict[str, Any]) -> Any:
        """Create a component instance.
        
        Args:
            id: Component identifier.
            type: Component type name.
            parameters: Component configuration.

        Returns:
            New component instance.

        Raises:
            ValueError: If type is not supported.
        """
        component_type = type.lower()
        
        if component_type not in self.component_types:
            valid_types = list(self.component_types.keys())
            raise ValueError(
                f"Unsupported type: {type}. "
                f"Valid types are: {valid_types}"
            )
            
        component_class = self.component_types[component_type]
        return component_class(id, parameters) 