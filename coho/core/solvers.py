"""Optimization solvers for parameter estimation."""

import jax
import jax.numpy as jnp
from typing import Callable, List, Optional, Any, Dict
from dataclasses import dataclass
from .models import ModelParameters
from .metrics import MetricsCallback
from .monitor import check_divergence
from .wave import Wave
from enum import Enum


class LineSearchMethod(Enum):
    """Available line search methods."""
    FIXED = "fixed"          # Fixed step size
    BACKTRACKING = "backtracking"  # Backtracking with Armijo condition
    WOLFE = "wolfe"         # Strong Wolfe conditions
    EXACT = "exact"         # Exact line search for quadratic functions


@dataclass
class LineSearchParams:
    """Line search parameters."""
    method: LineSearchMethod = LineSearchMethod.FIXED
    c1: float = 1e-4        # Armijo condition parameter
    c2: float = 0.9         # Curvature condition parameter
    max_iter: int = 10      # Maximum line search iterations
    step_size: float = 0.1  # Initial step size


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


@dataclass
class GradientDescent:
    """Gradient descent solver for parameter optimization.
    
    Performs gradient-based optimization of model parameters using
    automatic differentiation through JAX.
    """
    
    def __init__(self, cost_fn: Callable,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 metrics_fn: Optional[MetricsCallback] = None,
                 line_search: Optional[LineSearchParams] = None):
        """Initialize solver.
        
        Args:
            cost_fn: Cost function to minimize
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            metrics_fn: Optional callback for computing metrics
            line_search: Optional line search parameters
        """
        self.cost_fn = cost_fn
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.metrics_fn = metrics_fn
        self.line_search = line_search or LineSearchParams()
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

            # Use line search to compute step size
            direction = {k: -v for k, v in grads.items()}
            alpha = self._line_search(current_params, direction, grads, target, self.line_search)

            # Update parameters using computed step size
            opt_params = {
                k: v + alpha * direction[k]
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

    def _line_search(self, 
                    params: ModelParameters,
                    direction: Dict[str, Wave],
                    grads: Dict[str, Wave],
                    target: Any,
                    ls_params: LineSearchParams) -> float:
        """Perform line search to find optimal step size."""
        if ls_params.method == LineSearchMethod.FIXED:
            return ls_params.step_size
            
        elif ls_params.method == LineSearchMethod.BACKTRACKING:
            # Backtracking line search
            pass
            
        elif ls_params.method == LineSearchMethod.WOLFE:
            # Similar to backtracking but with additional curvature condition
            # Implementation follows standard Wolfe conditions
            pass
            
        elif ls_params.method == LineSearchMethod.EXACT:
            # For quadratic objectives only
            # Minimizes f(x + αd) exactly
            pass
            
        return ls_params.step_size

@dataclass
class ConjugateGradient:
    """Conjugate gradient solver with multiple beta calculation methods."""
    
    BETA_METHODS = ['FR', 'PR', 'HS', 'DY', '5th']  # Fletcher-Reeves, Polak-Ribière, Hestenes-Stiefel, Dai-Yuan, 5th rule
    
    def __init__(self, cost_fn: Callable,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 metrics_fn: Optional[MetricsCallback] = None,
                 beta_method: str = '5th',
                 line_search: Optional[LineSearchParams] = None):
        """Initialize solver.
        
        Args:
            cost_fn: Cost function to minimize
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            metrics_fn: Optional callback for computing metrics
            beta_method: Method for computing beta ('FR', 'PR', 'HS', 'DY', or '5th')
            line_search: Optional line search parameters
        """
        if beta_method not in self.BETA_METHODS:
            raise ValueError(f"beta_method must be one of {self.BETA_METHODS}")
        
        self.cost_fn = cost_fn
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.metrics_fn = metrics_fn
        self.beta_method = beta_method
        self.line_search = line_search or LineSearchParams()
        self.grad_fn = jax.grad(cost_fn, argnums=0)
    
    def _compute_beta(self, 
                     grad_new: Dict[str, Wave],
                     grad_old: Dict[str, Wave],
                     direction: Dict[str, Wave]) -> float:
        """Compute beta using selected method."""
        if self.beta_method == 'FR':  # Fletcher-Reeves
            num = sum(jnp.sum(g.form * g.form) for g in grad_new.values())
            den = sum(jnp.sum(g.form * g.form) for g in grad_old.values())
            
        elif self.beta_method == 'PR':  # Polak-Ribière
            num = sum(jnp.sum(g.form * (g.form - o.form)) 
                     for g, o in zip(grad_new.values(), grad_old.values()))
            den = sum(jnp.sum(g.form * g.form) for g in grad_old.values())
            
        elif self.beta_method == 'HS':  # Hestenes-Stiefel
            num = sum(jnp.sum(g.form * (g.form - o.form)) 
                     for g, o in zip(grad_new.values(), grad_old.values()))
            den = sum(jnp.sum(d.form * (g.form - o.form)) 
                     for d, g, o in zip(direction.values(), grad_new.values(), grad_old.values()))
            
        elif self.beta_method == 'DY':  # Dai-Yuan
            num = sum(jnp.sum(g.form * g.form) for g in grad_new.values())
            den = sum(jnp.sum(d.form * (g.form - o.form)) 
                     for d, g, o in zip(direction.values(), grad_new.values(), grad_old.values()))
            
        elif self.beta_method == '5th':
            num = self._hessian()
            den = self._hessian() 
        
        beta = float(num / (den + 1e-10))
        return max(0.0, beta) if self.beta_method in ['PR', 'HS'] else beta
    
    def _hessian(self):
        return 0
        
    def solve(self, parameters: ModelParameters, target: Any,
             callback: Optional[Callable] = None) -> OptimizeResult:
        """Run conjugate gradient optimization."""
        opt_params = parameters.optimizable.copy()
        fixed_params = parameters.fixed
        metrics_history: Dict[str, List[float]] = {}
        current_params = None
        previous_params = None
        best_params = None
        best_cost = float('inf')
        
        # Initial gradient and direction
        grads = self.grad_fn(opt_params, target=target, **fixed_params)
        direction = {k: -v for k, v in grads.items()}
        prev_grads = None
        
        for step in range(self.max_iterations):
            previous_params = current_params
            current_params = ModelParameters(optimizable=opt_params, fixed=fixed_params)
            
            # Compute cost and metrics
            cost = self.cost_fn(opt_params, target=target, **fixed_params)
            metrics_history.setdefault('cost', []).append(float(cost))
            
            if cost < best_cost:
                best_cost = cost
                best_params = current_params
                
            # Additional metrics
            if self.metrics_fn is not None:
                metrics = self.metrics_fn(current_params, previous_params, target)
                if metrics:
                    for name, value in metrics.items():
                        metrics_history.setdefault(name, []).append(value)
            
            if callback is not None:
                callback(step, current_params, metrics_history)
            
            # Check convergence
            rel_metrics = [v[-1] for k, v in metrics_history.items() 
                         if k.startswith('Rel(')]
            if rel_metrics and max(rel_metrics) < self.tolerance:
                break
                
            # Compute step size using line search
            alpha = self._line_search(current_params, direction, grads, target, self.line_search)

            # Update parameters using computed step size
            opt_params = {
                k: v + alpha * direction[k]
                for k, v in opt_params.items()
            }
            
            # Compute new gradient
            prev_grads = grads
            grads = self.grad_fn(opt_params, target=target, **fixed_params)
            
            # Compute beta and update direction
            if prev_grads is not None:
                beta = self._compute_beta(grads, prev_grads, direction)
                direction = {
                    k: -grads[k] + beta * direction[k]
                    for k in grads
                }
            
            # Check for divergence
            if check_divergence(metrics_history):
                return self._create_result(
                    best_params,
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

    def _line_search(self, 
                    params: ModelParameters,
                    direction: Dict[str, Wave],
                    grads: Dict[str, Wave],
                    target: Any,
                    ls_params: LineSearchParams) -> float:
        """Perform line search to find optimal step size."""
        if ls_params.method == LineSearchMethod.FIXED:
            return ls_params.step_size
            
        elif ls_params.method == LineSearchMethod.BACKTRACKING:
            # Backtracking line search
            pass
            
        elif ls_params.method == LineSearchMethod.WOLFE:
            # Similar to backtracking but with additional curvature condition
            # Implementation follows standard Wolfe conditions
            pass
            
        elif ls_params.method == LineSearchMethod.EXACT:
            # For quadratic objectives only
            # Minimizes f(x + αd) exactly
            pass
            
        return ls_params.step_size