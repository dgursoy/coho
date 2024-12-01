"""Operators for wave propagation and interaction."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Callable
from functools import wraps
import numpy as np

T = TypeVar('T')  # Input type
U = TypeVar('U')  # Output type

class Operator(ABC):
    """Base operator class."""
    
    @abstractmethod
    def apply(self, input: T, **kwargs) -> U:
        """Forward operation."""
        pass
    
    @abstractmethod
    def adjoint(self, input: U, **kwargs) -> T:
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
    
    def apply(self, input: Any) -> Any:
        """Apply operators in sequence with their respective arguments."""
        result = input
        for op, kwargs in self.operators:
            result = op.apply(result, **kwargs)
        return result
    
    def adjoint(self, input: Any) -> Any:
        """Apply adjoint operators in reverse sequence."""
        result = input
        for op, kwargs in reversed(self.operators):
            result = op.adjoint(result, **kwargs)
        return result

# Local imports
from .propagation import *
from .interaction import *
from .scanning import *

__all__ = [
    'Interact',
    'Detect',
    'Rotate',
    'Translate'
    'FresnelPropagate',
    'Scan',
    'ScanManager'
]
