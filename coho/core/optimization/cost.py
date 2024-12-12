"""Objective functions for optimization."""

import torch
from abc import ABC, abstractmethod
from typing import Callable
from ..component.wave import Wave

class Cost(ABC):
    """Base class for objective functions."""

    def __init__(self, target: torch.Tensor, operator: Callable, operator_args: dict = None) -> None:
        """Initialize objective."""
        self.target = target
        self.operator = operator
        self.operator_args = operator_args or {}
        self.variable_names = []  # List of variable names to optimize

    def set_variable_names(self, names: list) -> None:
        """Set names of variables to optimize."""
        self.variable_names = names

    @abstractmethod
    def evaluate(self, estimates: list) -> float:
        """Compute objective value."""
        pass

    @abstractmethod
    def gradient(self, estimates: list) -> list:
        """Compute gradient of objective."""
        pass

class LeastSquares(Cost):
    """L2 norm fitting objective."""

    def __init__(self, target: torch.Tensor, operator: Callable, operator_args: dict = None) -> None:
        """Initialize objective."""
        super().__init__(target, operator, operator_args)
        self.cost_history = []

    def evaluate(self, estimates: list) -> float:
        """Compute L2 norm difference."""
        # Create kwargs for operator
        kwargs = {name: est for name, est in zip(self.variable_names, estimates)}
        kwargs.update(self.operator_args)
        
        # Forward pass
        forward = self.operator.apply(**kwargs)
        residual = forward['modulated'] - self.target
        cost = torch.sum(residual ** 2)
        self.cost_history.append(float(cost))
        return float(cost)
    
    def gradient(self, estimates: list) -> list:
        """Compute gradient of L2 norm difference."""
        # Create kwargs for operator
        kwargs = {name: est for name, est in zip(self.variable_names, estimates)}
        kwargs.update(self.operator_args)
        
        # Forward pass
        forward = self.operator.apply(**kwargs)
        residual = forward['modulated'] - self.target
        
        # Adjoint pass
        adjoint = self.operator.adjoint(intensity=2 * residual)
        
        # Return gradients in same order as variable_names
        return [adjoint[name] for name in self.variable_names]

