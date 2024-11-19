# core/simulation/element.py

"""Base element class for optical simulation."""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from scipy.ndimage import rotate
from coho.config.models import OpticProperties, SampleProperties

__all__ = [
    'Element'
]

class Element(ABC):
    """Base class for all optical elements."""

    def __init__(self, properties: Union[OpticProperties, SampleProperties]):
        self.properties = properties
        self._initialize_element()

    @property
    def size(self) -> int:
        return self.properties.grid.size
    
    def _initialize_element(self):
        self.profile = self.generate_profile()
        self.profile = self._apply_rotation()

    def _apply_rotation(self) -> np.ndarray:
        """Apply rotation to the profile."""
        rotation = self.properties.geometry.rotation
        return rotate(self.profile, rotation, mode='constant', order=1)

    @abstractmethod
    def generate_profile(self) -> np.ndarray:
        pass
        
