"""Metrics calculators for optimization."""

import jax.numpy as jnp
from typing import Dict, Optional, Any, Protocol
from .models import ModelParameters
from .wave import Wave

class MetricsCallback(Protocol):
    """Protocol for computing optimization metrics.
    
    This protocol defines the interface that any metrics calculator must implement.
    It ensures that metrics calculators provide a consistent way to compute and
    return metrics during optimization.
    """
    def __call__(self, 
                 current: ModelParameters,
                 previous: Optional[ModelParameters] = None,
                 target: Optional[Any] = None) -> Dict[str, float]:
        """Compute metrics for current optimization state.
        
        Args:
            current: Current parameter values
            previous: Previous parameter values (for relative metrics)
            target: Target output to match
            
        Returns:
            Dict of metric names to values
        """
        ...


class WaveMetrics:
    """Wave-specific metrics calculator.
    
    Computes relative and absolute errors for all Wave parameters
    in the optimization.
    """
    
    def __init__(self, waves: Optional[Dict[str, Wave]] = None):
        """Initialize metrics calculator.
        
        Args:
            waves: Optional dict mapping parameter names to waves
        """
        self.waves = waves or {}
    
    def __call__(self, 
                 current: ModelParameters,
                 previous: Optional[ModelParameters] = None,
                 target: Optional[Any] = None) -> Dict[str, float]:
        """Compute metrics for all Wave parameters.
        
        Args:
            current: Current parameter values
            previous: Previous parameter values (for relative metrics)
            target: Target output to match
            
        Returns:
            Dict of metric names to values
        """
        metrics = {}
        
        # Relative errors
        if previous is not None:
            for name, value in current.optimizable.items():
                if isinstance(value, Wave):
                    rel_error = self.compute_relative_error(
                        value, previous.optimizable[name]
                    )
                    metrics[f"Rel({name})"] = rel_error
        
        # Absolute errors
        for name, value in current.optimizable.items():
            if isinstance(value, Wave) and name in self.waves:
                abs_error = self.compute_absolute_error(
                    value, self.waves[name]
                )
                metrics[f"Abs({name})"] = abs_error
                
        return metrics
    
    @staticmethod
    def compute_relative_error(current: Wave, previous: Wave) -> float:
        """Compute relative error between two waves."""
        return (jnp.mean(jnp.abs(current.form - previous.form)) / 
                jnp.mean(jnp.abs(previous.form)))
    
    @staticmethod
    def compute_absolute_error(wave: Wave, reference: Wave) -> float:
        """Compute absolute error against reference wave."""
        return jnp.mean(jnp.abs(wave.form - reference.form))
