# core/optimization/__init__.py

"""Core optimization components for inverse problems.

This module provides base classes for optimization algorithms and
objective functions used in solving inverse problems.

Components:
    Solver: Optimization algorithms
        gradient_descent: First-order gradient method

    Objective: Cost functions
        least_squares: L2 norm fitting
        magnitude_fitting: Amplitude-only matching
"""

from .solvers import (
    GradientDescent
)
from .objectives import (
    LeastSquares
)

__all__ = [
    'GradientDescent',
    'LeastSquares',
    'MagnitudeFitting'
]
