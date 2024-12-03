"""Classes for simulating wavefront interactions with optical elements."""

# Standard imports
import numpy as np

# Local imports
from coho.core.component import Wave
from .base import Operator

__all__ = [
    'Modulate', 
    'Detect'
    ]

class Modulate(Operator):
    """Modulate wavefront by another wavefront."""

    def apply(self, wave: Wave, modulator: Wave) -> Wave:
        """Forward modulation."""
        if not self._positions_match(wave.position, modulator.position):
            raise ValueError("Positions of waves do not match.")
        return wave * modulator

    def adjoint(self, wave: Wave, modulator: Wave) -> Wave:
        """Adjoint modulation."""
        if not self._positions_match(wave.position, modulator.position):
            raise ValueError("Positions of waves do not match.")
        return wave / modulator
    
    @staticmethod
    def _positions_match(pos1, pos2) -> bool:
        """Check if all elements in positions match, allowing for repeated arrays."""
        pos1, pos2 = np.atleast_1d(pos1), np.atleast_1d(pos2)
        return np.all(pos1 == pos2) or \
               np.all(pos1 == pos2[0]) or \
               np.all(pos1[0] == pos2)

    def __str__(self) -> str:
        """Simple string representation."""
        return "Wave modulation operator"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}()"

class Detect(Operator):
    """Detect wavefront intensity."""

    def apply(self, wave: Wave) -> np.ndarray:
        """Wavefront to intensity."""
        self.wave = wave # Save wave for adjoint
        return wave.amplitude

    def adjoint(self, intensity: np.ndarray) -> Wave:
        """Intensity to wavefront."""
        self.wave.form = intensity # No phase information
        return self.wave

    def __str__(self) -> str:
        """Simple string representation."""
        return "Wave detection operator"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}()"