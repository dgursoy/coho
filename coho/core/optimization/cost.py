"""Objective functions for optimization."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from ..component.wave import Wave

class Cost(ABC):
    """Base class for objective functions."""

    def __init__(self, target: np.ndarray, operator: Callable) -> None:
        """Initialize objective."""
        self.target = target
        self.operator = operator

    @abstractmethod
    def evaluate(self, estimate: Wave) -> float:
        """Compute objective value. """
        pass

    @abstractmethod
    def gradient(self, estimate: Wave) -> Wave:
        """Compute gradient of objective."""
        pass

class LeastSquares(Cost):
    """L2 norm fitting objective."""

    def __init__(self, target: np.ndarray, operator: Callable) -> None:
        """Initialize objective."""
        super().__init__(target, operator)
        self.cost_history = []

    def evaluate(self, estimate: Wave) -> float:
        """Compute L2 norm difference."""
        residual = self.operator.apply(estimate) - self.target
        cost = np.sum(residual ** 2)
        self.cost_history.append(float(cost))
        return cost
    
    def gradient(self, estimate: Wave) -> Wave:
        """Compute gradient of L2 norm difference."""
        residual = self.operator.apply(estimate) - self.target
        return 2 * self.operator.adjoint(residual)

