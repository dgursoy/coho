"""Classes for simulating wavefront scanning."""

# Standard imports
from typing import Union, List
import numpy as np

# Local imports
from .base import Operator
from ..component import Wave

class Broadcast(Operator):
    """Broadcast wave to multiple positions."""
    
    def _prepare_position(self, position: Union[List[float], np.ndarray]) -> np.ndarray:
        """Convert position to float array."""
        return np.asarray(position, dtype=float)

    def apply(self, wave: Wave, position: Union[List[float], np.ndarray]) -> Wave:
        """Forward broadcast."""
        position = self._prepare_position(position)
        
        # If wave is already broadcasted, reshape instead of adding new dimension
        if wave.form.ndim > 2:
            wave.form = wave.form.reshape(-1, *wave.form.shape[-2:])
        else:
            wave.form = wave.form[np.newaxis, ...]
        
        # Now broadcast to correct number of positions
        wave.form = np.broadcast_to(wave.form, (len(position), *wave.form.shape[-2:]))
        wave.position = position
        return wave
    
    def adjoint(self, wave: Wave, position: Union[List[float], np.ndarray]) -> Wave:
        """Adjoint broadcast."""
        position = self._prepare_position(position)
        wave.form = np.mean(wave.form, axis=0)
        wave.position = position[0]
        return wave

    def __str__(self) -> str:
        """Simple string representation."""
        return "Broadcast operator"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}()"
