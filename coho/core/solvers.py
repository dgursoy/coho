"""Optimization solvers for parameter estimation."""

import jax
from typing import Callable, List, Optional, Any, Dict
from dataclasses import dataclass
from .models import ModelParameters
from .metrics import MetricsCallback
from .monitor import check_divergence

@dataclass
class OptimizeResult:
    """Generic optimization result container.
    
    Attributes:
        parameters: Final optimized parameters
        metrics_history: History of metric values during optimization
        success: Whether optimization converged successfully
        message: Description of optimization outcome
    """
    parameters: ModelParameters
    metrics_history: Dict[str, List[float]]
    success: bool = False
    message: str = ""


class GradientSolver:
    """Gradient descent solver for parameter optimization.
    
    Performs gradient-based optimization of model parameters using
    automatic differentiation through JAX.
    """
    
    def __init__(self, cost_fn: Callable,
                 learning_rate: float = 0.1,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 metrics_fn: Optional[MetricsCallback] = None):
        """Initialize solver.
        
        Args:
            cost_fn: Cost function to minimize
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance on relative change
            metrics_fn: Optional callback for computing metrics
        """
        self.cost_fn = cost_fn
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.metrics_fn = metrics_fn
        self.grad_fn = jax.grad(cost_fn, argnums=0)
    
    def solve(self, parameters: ModelParameters, target: Any,
             callback: Optional[Callable] = None) -> OptimizeResult:
        """Run optimization to find optimal parameters.
        
        Args:
            parameters: Initial parameters with optimizable/fixed specification
            target: Target output to match
            callback: Optional callback function for monitoring progress
        
        Returns:
            OptimizeResult: Optimization results including metric history
        """
        opt_params = parameters.optimizable.copy()
        fixed_params = parameters.fixed
        metrics_history: Dict[str, List[float]] = {}
        current_params = None
        previous_params = None
        best_params = None
        best_cost = float('inf')
        
        for step in range(self.max_iterations):
            # Update parameters
            previous_params = current_params
            current_params = ModelParameters(optimizable=opt_params, fixed=fixed_params)
            
            # Compute cost and add to metrics
            cost = self.cost_fn(opt_params, target=target, **fixed_params)
            metrics_history.setdefault('cost', []).append(float(cost))
            
            # Track best parameters
            if cost < best_cost:
                best_cost = cost
                best_params = current_params
                
            # Compute additional metrics if provided
            metrics = {}
            if self.metrics_fn is not None:
                metrics = self.metrics_fn(current_params, previous_params, target)
                if metrics is None:
                    metrics = {}
            
            # Add metrics to history
            for name, value in metrics.items():
                metrics_history.setdefault(name, []).append(value)
            
            # Call callback if provided
            if callback is not None:
                callback(step, current_params, metrics_history)
            
            # Check convergence using relative errors if available
            rel_metrics = [v[-1] for k, v in metrics_history.items() 
                         if k.startswith('Rel(')]
            if rel_metrics and max(rel_metrics) < self.tolerance:
                break
            
            # Compute gradients and update
            grads = self.grad_fn(opt_params, target=target, **fixed_params)
            opt_params = {
                k: v - self.learning_rate * grads[k]
                for k, v in opt_params.items()
            }
            
            # Check for divergence
            if check_divergence(metrics_history):
                return self._create_result(
                    best_params,  # Keep best parameters seen
                    metrics_history,
                    success=False,
                    message="Optimization diverged"
                )
        
        return self._create_result(
            current_params,
            metrics_history,
            success=True,
            message="Converged within tolerance"
        )
    
    def _create_result(self,
                      parameters: ModelParameters,
                      metrics_history: Dict[str, List[float]],
                      success: bool,
                      message: str) -> OptimizeResult:
        """Helper to create optimization result container.
        
        Args:
            parameters: Final parameter values
            metrics_history: History of all computed metrics
            success: Whether optimization succeeded
            message: Description of outcome
        
        Returns:
            OptimizeResult: Container with optimization results
        """
        return OptimizeResult(
            parameters=parameters,
            metrics_history=metrics_history,
            success=success,
            message=message
        )
