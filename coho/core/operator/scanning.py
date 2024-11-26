"""Classes for geometric transformations."""

from scipy.ndimage import rotate, shift
import numpy as np
from typing import Tuple

# Local imports
from . import Operator

__all__ = [
    'Rotate', 
    'Translate'
    ]

class Rotate(Operator):
    """Rotate an image."""
    
    def apply(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image."""
        return rotate(image, angle, reshape=False, order=1)

    def adjoint(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Adjoint model."""
        return self.apply(image, -angle)

class Translate(Operator):
    """Translate an image."""
    
    def apply(self, image: np.ndarray, translation: Tuple[float, float]) -> np.ndarray:
        """Translate image."""
        return shift(image, [translation.x, translation.y], order=1)

    def adjoint(self, image: np.ndarray, translation: Tuple[float, float]) -> np.ndarray:
        """Adjoint model."""
        return self.apply(image, -translation)  