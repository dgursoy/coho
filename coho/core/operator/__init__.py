"""Operators for wave propagation and interaction."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class Operator(ABC):
    """Base operator class."""
    
    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """Forward operation."""
        pass
    
    @abstractmethod
    def adjoint(self, *args, **kwargs) -> Any:
        """Adjoint operation."""
        pass

class Pipeline(Operator):
    """Pipeline of operators."""
    
    def __init__(self, operators: list[tuple[Operator, dict]]):
        """Initialize pipeline with operators and their arguments.
        
        Args:
            operators: List of tuples containing (operator, kwargs)
        """
        self.operators = operators
    
    def apply(self, input_data: Any) -> Any:
        """Apply operators in sequence with their respective arguments."""
        result = input_data
        for op, kwargs in self.operators:
            result = op.apply(result, **kwargs)
        return result
    
    def adjoint(self, input_data: Any) -> Any:
        """Apply adjoint operators in reverse sequence."""
        result = input_data
        for op, kwargs in reversed(self.operators):
            result = op.adjoint(result, **kwargs)
        return result

# Local imports
from .propagation import *
from .interaction import *

__all__ = [
    'Interact',
    'Detect',
    'Rotate',
    'Translate'
    'FresnelPropagate',
]
