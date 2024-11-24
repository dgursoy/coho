# core/simulation/element.py

"""Base element class for optical simulation."""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from scipy.ndimage import rotate, shift
from coho.config.models import OpticProperties, SampleProperties

__all__ = [
    'Element'
]

class Element(ABC):
    """Base class for all optical elements."""

    def __init__(self, properties: Union[OpticProperties, SampleProperties]):
        """Initialize the element with specified properties."""
        self.properties = properties
        self._profile = None

    @property
    def size(self) -> int:
        return self.properties.grid.size
    
    @property
    def profile(self):
        """Lazily generate and return the profile."""
        if self._profile is None:
            self._profile = self._generate_profile()
            self._profile = self._apply_rotation()
            self._profile = self._apply_translation()
        return self._profile
    
    def _apply_rotation(self) -> np.ndarray:
        """Apply rotation to the profile."""
        rotation = self.properties.geometry.rotation
        return rotate(self._profile, rotation, reshape=False, order=1)
    
    def _apply_translation(self) -> np.ndarray:
        """Apply translation to the profile."""
        translation = self.properties.geometry.position
        return shift(self._profile, [translation.x, translation.y], order=1)

    def clear_cache(self):
        """Clear cached computations."""
        self._profile = None

    @abstractmethod
    def _generate_profile(self) -> np.ndarray:
        pass
        
