"""Core components for optical systems."""

# Base class imports
import numpy as np
from abc import ABC

# Local imports
from coho.config.models import ComponentBase

class Component(ABC):
    """Base class for components."""
    
    def __init__(self, properties: ComponentBase):
        """Initialize component."""
        self.physical = properties.physical
        self.profile = properties.profile

        # Cache the complex form
        self.complexform = None

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
    