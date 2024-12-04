"""Classes for simulating wavefront scanning."""

# Standard imports
from typing import Union, List
import numpy as np

# Local imports
from .base import Operator
from ..component import Wave

class Broadcast(Operator):
    """Broadcast wave across multiple parameter values."""
    
    def __init__(self, param_name: str = 'position'):
        """Initialize broadcaster.
        
        Args:
            param_name: Name of the wave parameter to broadcast over
        """
        self.param_name = param_name

    def _prepare_values(self, values: Union[List[float], np.ndarray]) -> np.ndarray:
        """Convert parameter values to float array."""
        return np.asarray(values, dtype=float)

    def apply(self, wave: Wave, values: Union[List[float], np.ndarray]) -> Wave:
        """Forward broadcast."""
        values = self._prepare_values(values)
        
        # If wave is already broadcasted, reshape instead of adding new dimension
        if wave.form.ndim > 2:
            wave.form = wave.form.reshape(-1, *wave.form.shape[-2:])
        else:
            wave.form = wave.form[np.newaxis, ...]
        
        # Now broadcast to correct number of values
        wave.form = np.broadcast_to(wave.form, (len(values), *wave.form.shape[-2:]))
        setattr(wave, self.param_name, values)
        return wave
    
    def adjoint(self, wave: Wave, values: Union[List[float], np.ndarray]) -> Wave:
        """Adjoint broadcast."""
        values = self._prepare_values(values)
        wave.form = np.mean(wave.form, axis=0)
        setattr(wave, self.param_name, values[0])
        return wave

    def __str__(self) -> str:
        """Simple string representation."""
        return f"Broadcast operator ({self.param_name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(param_name='{self.param_name}')"
