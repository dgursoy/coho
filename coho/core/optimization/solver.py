# core/optimization/solver.py

"""Base classes for iterative solvers.

Classes:
    IterativeSolver: Abstract base class for iterative solvers.
    ConvergenceCriterion: Abstract base class for convergence checks.
"""

from abc import ABC, abstractmethod
import numpy as np
from coho.core.optimization.objective import Objective
from coho.config.models import SolverProperties

__all__ = [
    'GradientDescent'
]   


class Solver(ABC):
    """Base class for all solvers."""
    
    def __init__(self, properties: SolverProperties) -> None:
        """Initialize solver.
        
        Args:
            id: Unique identifier
            parameters: Configuration dict
        """
        self.properties = properties
    
    @abstractmethod
    def solve(self) -> np.ndarray:
        """Solve the problem."""
        pass
    

class IterativeSolver(Solver):
    """Base class for iterative solvers."""
    def __init__(self, properties: SolverProperties, objective: Objective) -> None:
        super().__init__(properties)
        self.objective = objective
        self._initialize_solver()

    def _initialize_solver(self) -> None:
        """Initialize solver."""
        self.current = self.properties.initial_guess
    
    def solve(self, target: np.ndarray) -> np.ndarray:
        """Run iterations until convergence or max iterations."""
        
        for _ in range(self.properties.iterations):
            self.current += self.update(target)    
            
        return self.current
    
    @abstractmethod
    def update(self, target: np.ndarray) -> np.ndarray:
        """Perform one iteration update."""
        pass


class GradientDescent(IterativeSolver):
    """Gradient descent solver."""
    
    def update(self) -> np.ndarray:
        """Perform gradient descent update."""
        return -self.properties.step_size * self.objective.gradient(self.current)
    