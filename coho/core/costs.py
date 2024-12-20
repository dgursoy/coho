"""Cost functions for optimization."""

import jax
import jax.numpy as jnp
from typing import Dict, Any
from .models import Model, ModelParameters


class Cost:
    """Base class for cost functions."""
    
    def __init__(self, model: Model):
        self.model = model
        self._compute = jax.jit(self.compute)
    
    def compute(self, prediction: jnp.ndarray, target: jnp.ndarray) -> float:
        """Compute cost between prediction and target."""
        raise NotImplementedError
    
    def __call__(self, opt_params: Dict[str, Any], *, target: jnp.ndarray, **fixed_params) -> float:
        """Compute full cost including forward pass.
        
        Args:
            opt_params: Optimizable parameters
            target: Target output to match
            **fixed_params: Fixed parameters
        """
        # Create Parameters object for model
        parameters = ModelParameters(
            optimizable=opt_params,
            **fixed_params
        )
        prediction = self.model(parameters)
        return self._compute(prediction, target)


class MSECost(Cost):
    """Mean squared error cost function."""
    
    def compute(self, prediction: jnp.ndarray, target: jnp.ndarray) -> float:
        return jnp.mean(jnp.square(prediction - target))
