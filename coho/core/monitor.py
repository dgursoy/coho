"""Monitoring utilities for optimization progress."""

from typing import Dict, List
from .models import ModelParameters
import jax.numpy as jnp

def print_metrics(step: int, params: ModelParameters, metrics_history: Dict[str, List[float]]):
    """Print optimization progress with ordered metrics.
    
    Args:
        step: Current iteration number
        params: Current parameter values
        metrics_history: History of all metrics
    """
    if step % 10 == 0:
        msg_parts = [f"Step {step}"]
        
        # Add cost if available
        if 'cost' in metrics_history:
            msg_parts.append(f"cost: {metrics_history['cost'][-1]:.4e}")
        
        # Add relative errors
        rel_metrics = sorted(
            (k, v[-1]) for k, v in metrics_history.items() 
            if k.startswith('Rel(')
        )
        for name, value in rel_metrics:
            msg_parts.append(f"{name}: {value:.4e}")
        
        # Add absolute errors
        abs_metrics = sorted(
            (k, v[-1]) for k, v in metrics_history.items() 
            if k.startswith('Abs(')
        )
        for name, value in abs_metrics:
            msg_parts.append(f"{name}: {value:.4e}")
        
        print(", ".join(msg_parts)) 

def check_divergence(metrics_history: Dict[str, List[float]], 
                    threshold: float = 10.0) -> bool:
    """Check if optimization is diverging.
    
    Args:
        metrics_history: History of metrics
        threshold: Factor increase that indicates divergence
        
    Returns:
        True if optimization appears to be diverging
    """
    if 'cost' in metrics_history and len(metrics_history['cost']) > 1:
        costs = metrics_history['cost']
        current = costs[-1]
        previous = costs[-2]
        
        # Check for NaN or sudden cost increase
        if (jnp.isnan(current) or 
            jnp.isinf(current) or 
            (current > threshold * previous)):
            return True
    return False