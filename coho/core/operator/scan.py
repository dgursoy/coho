"""Classes for simulating wavefront scanning."""

# Standard imports
from typing import Union, List, Dict
import numpy as np

# Local imports
from .base import Operator
from .decorators import validate_form
from ..component import Wave

class Broadcast(Operator):
    """Broadcast wave across multiple parameter values."""
    
    def _prepare_values(self, values: Dict[str, Union[List[float], np.ndarray]]) -> Dict[str, np.ndarray]:
        """Convert parameter values to float arrays."""
        return {
            name: np.asarray(vals, dtype=float)
            for name, vals in values.items()
        }
    
    @validate_form
    def apply(self, wave: Wave, values: Dict[str, Union[List[float], np.ndarray]]) -> Wave:
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
            wave.form = wave.form[np.newaxis, ...]
        
        # Broadcast to correct number of values
        wave.form = np.broadcast_to(wave.form, (n_values, *wave.form.shape[-2:]))
        
        # Set all parameter values
        for name, vals in values.items():
            setattr(wave, name, vals)
            
        return wave
       
    def adjoint(self, wave: Wave, values: Dict[str, Union[List[float], np.ndarray]]) -> Wave:
        """Adjoint broadcast."""
        values = self._prepare_values(values)
        wave.form = np.mean(wave.form, axis=0)
        
        # Set first value for each parameter
        for name, vals in values.items():
            setattr(wave, name, vals[0])
            
        return wave
