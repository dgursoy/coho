"""Detectors."""

import numpy as np

# Local imports
from coho.core.component import Component

__all__ = [
    'StandardDetector',
]

class Detector(Component):
    """Base detector class."""
    pass

class StandardDetector(Detector):
    """Standard detector."""
    def _generate_image(self) -> np.ndarray:
        """Generate a standard detector profile."""
        return np.zeros((self.size, self.size))

