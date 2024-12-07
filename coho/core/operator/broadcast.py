"""Broadcast wave across multiple parameter values."""

# Standard imports
import torch

# Local imports
from .base import Operator, TensorDict
from ..component import Wave
from ..utils.decorators import (
    requires_unstacked,
    requires_stacked
)

class Broadcast(Operator):
    """Broadcast wave across multiple parameter values."""
    
    def _prepare_values(self, values: TensorDict) -> TensorDict:
        """Convert parameter values to float tensors."""
        return {
            name: torch.as_tensor(vals, dtype=torch.float64)
            for name, vals in values.items()
        }
    
    def apply(self, wave: Wave, values: TensorDict) -> Wave:
        """Forward broadcast.
        
        Args:
            wave: Wave to broadcast
            values: Dict mapping parameter names to their values
        """
        values = self._prepare_values(values)
        n_values = len(next(iter(values.values())))  # Length of first value array
        
        # Verify all value arrays have same length
        if not all(len(v) == n_values for v in values.values()):
            raise ValueError("All parameter value arrays must have the same length")
            
        # If wave is already broadcasted, reshape instead of adding new dimension
        if wave.form.ndim > 2:
            wave.form = wave.form.reshape(-1, *wave.form.shape[-2:])
        else:
            wave.form = wave.form.unsqueeze(0)
        
        # Broadcast to correct number of values
        wave.form = wave.form.expand(n_values, *wave.form.shape[-2:])
        
        # Set all parameter values
        for name, vals in values.items():
            setattr(wave, name, vals.to(wave.form.device))
        return wave
    
    def adjoint(self, wave: Wave, values: TensorDict) -> Wave:
        """Adjoint broadcast."""
        values = self._prepare_values(values)
        wave.form = torch.mean(wave.form, dim=0)
        
        # Set first value for each parameter
        for name, vals in values.items():
            setattr(wave, name, vals[0].to(wave.form.device))
        return wave

class Stack(Operator):
    """Stacks a wave along stack (first) dimension in-place."""
    
    @requires_unstacked
    def apply(self, wave: Wave, stack_size: int) -> Wave:
        """Stack n copies of wave along stack dimension in-place."""
        wave.form = wave.form.expand(stack_size, *wave.form.shape[-2:])
        return wave
    
    @requires_stacked
    def adjoint(self, wave: Wave) -> Wave:
        """Average along stack dimension in-place."""
        wave.form = wave.form.mean(dim=0, keepdim=True)
        return wave
