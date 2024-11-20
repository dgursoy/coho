# core/optimization/objective.py

"""Objective functions for optimization problems.

This module provides implementations of various objective functions
used in optimization algorithms.

Classes:
    Objective: Abstract base class for objective functions
    LeastSquares: L2 norm fitting objective
    MagnitudeFitting: Amplitude-only matching objective

Methods:
    evaluate: Compute objective value
"""

import numpy as np
from abc import ABC, abstractmethod
from coho.config.models import ObjectiveProperties
from typing import Callable

__all__ = [
    'LeastSquares',
]


class Objective(ABC):
    """Base class for objective functions."""

    def __init__(self, properties: ObjectiveProperties, target: np.ndarray, operator: Callable) -> None:
        """Initialize objective."""
        self.properties = properties
        self.target = target
        self.operator = operator

    @abstractmethod
    def evaluate(self, estimate: np.ndarray) -> float:
        """Compute objective value. """
        pass

    @abstractmethod
    def gradient(self, estimate: np.ndarray) -> np.ndarray:
        """Compute gradient of objective."""
        pass


class LeastSquares(Objective):
    """L2 norm fitting objective."""

    def evaluate(self, estimate: np.ndarray) -> float:
        """Compute L2 norm difference."""
        residual = self.operator.forward(estimate) - self.target
        return np.sum(residual ** 2)
    
    def gradient(self, estimate: np.ndarray) -> np.ndarray:
        """Compute gradient of L2 norm difference."""
        residual = self.operator.forward(estimate) - self.target
        return 2 * self.operator.adjoint(residual)

