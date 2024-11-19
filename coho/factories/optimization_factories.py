# factories/optimization_factories.py

"""Factories for optimization components.

This module provides factories for creating optimization-related
components like solvers and objective functions.

Classes:
    SolverFactory: Creates solver instances
    ObjectiveFactory: Creates objective function instances
"""

from .base_factory import ComponentFactory
from ..core.optimization.solvers import *
from ..core.optimization.objectives import *
from ..config.models import *

__all__ = ['SolverFactory', 'ObjectiveFactory']

# Component type mappings
SOLVER_TYPES = {
    'gradient_descent': GradientDescent
}

OBJECTIVE_TYPES = {
    'least_squares': LeastSquares,
    'magnitude_fitting': MagnitudeFitting
}


class SolverFactory(ComponentFactory[SolverProperties, Solver]):
    """Factory for optimization solver creation."""
    def __init__(self):
        super().__init__(SOLVER_TYPES)


class ObjectiveFactory(ComponentFactory[ObjectiveProperties, Objective]):
    """Factory for objective function creation."""
    def __init__(self):
        super().__init__(OBJECTIVE_TYPES)
