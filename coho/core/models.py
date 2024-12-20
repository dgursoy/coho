"""Forward models for wave propagation."""

import jax
import jax.numpy as jnp
from functools import wraps
from dataclasses import dataclass
from typing import Dict, Any, Set, Optional
from .wave import Wave
from .operators import detect, propagate, modulate


def optimizable(*param_names: str):
    """Decorator to mark which parameters can be optimized."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        wrapper.optimizable_params = set(param_names)
        return wrapper
    return decorator


@dataclass
class ModelParameters:
    """Container for model parameters with validation."""
    optimizable: Dict[str, Any]
    fixed: Dict[str, Any]
    
    def __init__(self, optimizable: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize parameters.
        
        Args:
            optimizable: Dict of parameters to optimize
            **kwargs: Additional fixed parameters
        """
        self.optimizable = optimizable or {}
        self.fixed = kwargs


class Model:
    """Base class for all forward models."""
    
    def forward(self, parameters: ModelParameters) -> jnp.ndarray:
        """Forward pass of the model."""
        raise NotImplementedError
    
    def __call__(self, parameters: ModelParameters) -> jnp.ndarray:
        return self.forward(parameters)
    
    @property
    def optimizable_parameters(self) -> Set[str]:
        """Get names of parameters that can be optimized."""
        return getattr(self.forward, 'optimizable_params', set())


class DualPropagationModel(Model):
    """Model for dual propagation wave problems."""
    
    def __init__(self):
        self._forward = jax.jit(
            jax.vmap(
                self._forward_single,
                in_axes=(None, None, 0, 0)
            )
        )
    
    def _forward_single(self, wave1: Wave, wave2: Wave, 
                       dist1: float, dist2: float) -> jnp.ndarray:
        """Single forward propagation for one distance pair."""
        prop1 = propagate(wave1, dist1)
        mod = modulate(prop1, wave2)
        prop2 = propagate(mod, dist2)
        return detect(prop2)
    
    @optimizable('wave1')
    def forward(self, parameters: ModelParameters) -> jnp.ndarray:
        """Forward pass with explicit parameter handling."""
        wave1 = parameters.optimizable.get('wave1') or parameters.fixed['wave1']
        wave2 = parameters.optimizable.get('wave2') or parameters.fixed['wave2']
        dist1 = parameters.fixed['dist1']
        dist2 = parameters.fixed['dist2']
        return self._forward(wave1, wave2, dist1, dist2)
    