"""Solver classes for optimization."""

# Standard imports
from abc import ABC, abstractmethod

# Local imports
from .cost import Cost
from ..component.wave import Wave

class Solver(ABC):
    """Base class for all solvers."""
    
    def __init__(self, step_size: float, iterations: int, initial_guess: Wave) -> None:
        """Initialize solver."""
        self.step_size = step_size
        self.iterations = iterations
        self.initial_guess = initial_guess
    
    @abstractmethod
    def solve(self) -> Wave:
        """Solve the problem."""
        pass

class IterativeSolver(Solver):
    """Base class for iterative solvers."""
    def __init__(self, 
                 cost: Cost,
                 step_size: float = 0.1, 
                 iterations: int = 100, 
                 initial_guess: Wave = None) -> None:
        super().__init__(step_size, iterations, initial_guess)
        self.cost = cost
        self._initialize_solver()

    def _initialize_solver(self) -> None:
        """Initialize solver."""
        self.current = self.initial_guess
    
    def solve(self) -> Wave:
        """Run iterations until convergence or max iterations."""
        
        for i in range(self.iterations):
            # Evaluate current cost
            cost = self.cost.evaluate(self.current)
            print(f"Iteration {i+1}/{self.iterations}: Cost = {cost}")
            # Update current estimate
            self.current += self.update()    
            
        return self.current
    
    @abstractmethod
    def update(self) -> Wave:
        """Perform one iteration update."""
        pass

class GradientDescent(IterativeSolver):
    """Gradient descent solver."""
    
    def update(self) -> Wave:
        """Perform gradient descent update."""
        return -self.step_size * self.cost.gradient(self.current)
    