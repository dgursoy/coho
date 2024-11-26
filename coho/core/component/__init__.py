"""Core components for optical systems."""

# Base class imports
from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np
from coho.config.models.component import ComponentBase

class Component(ABC):
    """Base class for components."""
    
    def __init__(self, properties: ComponentBase):
        """Initialize component."""
        # Properties
        self.physical = properties.physical
        self.profile = properties.profile
        self.grid = properties.grid
        self.geometry = properties.geometry
        
        # Cache
        self._image = None
    
    @property
    def size(self) -> int:
        """Grid size for the component."""
        return self.grid.size
    
    @property
    def spacing(self) -> float:
        """Grid spacing for the component."""
        return self.grid.spacing
    
    @property
    def position(self) -> Tuple[float, float]:
        """Component position."""
        return self.geometry.position
    
    @property
    def distance(self) -> float:
        """Component distance."""
        return self.geometry.distance
    
    @property
    def rotation(self) -> float:
        """Component rotation."""
        return self.geometry.rotation
    
    @property
    def image(self) -> np.ndarray:
        """Lazily generate and return the component image."""
        if self._image is None:
            self._image = self._generate_image()
        return self._image
    
    def clear_cache(self) -> None:
        """Clear cached computations."""
        self._image = None

    @abstractmethod
    def _generate_image(self) -> np.ndarray:
        """Generate the component image."""
        pass
    
# Submodules
from .wavefronts import *
from .optics import *
from .samples import *
from .detectors import *

__all__ = [
    'UniformWavefront',
    'GaussianWavefront',
    'CustomWavefront',
    'CircularOptic',
    'CodedOptic',
    'CustomOptic',
    'GaussianOptic',
    'RectangularOptic',
    'BaboonSample',
    'BarbaraSample',
    'CameramanSample',
    'CheckerboardSample',
    'CustomSample',
    'HouseSample',
    'IndianSample',
    'LenaSample',
    'PeppersSample',
    'SheppLoganSample',
    'ShipSample',
    'StandardDetector',
]
    