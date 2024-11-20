# factories/optimization_factories.py

"""Factories for optimization components.

This module provides factories for creating optimization-related
components like solvers and objective functions.

Classes:
    SolverFactory: Creates solver instances
    ObjectiveFactory: Creates objective function instances
"""

from .base_factory import ComponentFactory
from ..core.optimization.solver import *
from ..core.optimization.objective import *
from ..config.models import *

__all__ = ['SolverFactory', 'ObjectiveFactory']


class SolverFactory(ComponentFactory[SolverProperties, Solver]):
    """Factory for optimization solver creation."""
    def __init__(self):
        super().__init__()
        self.register('gradient_descent', GradientDescent)


class ObjectiveFactory(ComponentFactory[ObjectiveProperties, Objective]):
    """Factory for objective function creation."""
    def __init__(self):
        super().__init__()
        self.register('least_squares', LeastSquares)
