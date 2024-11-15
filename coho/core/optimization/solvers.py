# core/optimization/solvers.py

"""Base classes for iterative solvers.

Classes:
    IterativeSolver: Abstract base class for iterative solvers.
    ConvergenceCriterion: Abstract base class for convergence checks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from typing import Any, Optional
import numpy as np
from coho.core.optimization.objectives import Objective


class Solver(ABC):
    """Base class for all solvers."""
    
    def __init__(
        self,
        id: Any,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize solver.
        
        Args:
            id: Unique identifier
            parameters: Configuration dict
        """
        self.id = id
        self.parameters = parameters or {}
    
    @abstractmethod
    def solve(self, objective: Objective) -> Dict[str, Any]:
        """Solve the problem.
        
        Args:
            objective: Objective class instance
            
        Returns:
            Dict containing solution information
        """
        pass


class IterativeSolver(Solver):
    """Base class for iterative solvers."""
    
    def __init__(
        self,
        id: Any,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(id, parameters)
        self.iterations = self.parameters.get("iterations", 100)
    
    def solve(self, objective: Objective) -> Dict[str, Any]:
        """Run iterations until convergence or max iterations."""
        previous = np.zeros(self.x.shape)
        
        for _ in range(self.iterations):
            current = self.update(objective, previous)    
            
        return current
    
    @abstractmethod
    def update(self, objective: Objective) -> np.ndarray:
        """Perform one iteration update.
        
        Args:
            objective: Objective class instance
            
        Returns:
            Updated solution
        """
        pass


class GradientDescent(IterativeSolver):
    """Simple gradient descent optimization solver."""
    
    def __init__(
        self,
        id: Any,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(id, parameters)
        self.step_size = self.parameters.get("step_size", 0.01)
    
    def update(self, objective: Objective) -> np.ndarray:
        """Perform gradient descent update."""
        self.x = self.x - self.step_size * objective.gradient(self.x)
        return self.x
    
