"""Samples."""

import numpy as np
from pathlib import Path

# Local imports
from . import Component

__all__ = [
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
]

RESOURCE_PATH = Path(__file__).parent.parent.parent / 'resources' / 'images'

class Sample(Component):
    """Base sample class."""
    
    def __init__(self, properties):
        """Initialize the sample with specified properties."""
        super().__init__(properties)

        # Generate the complex form
        self.form = np.expand_dims(self.generate_form(), axis=0)

    def generate_form(self) -> np.ndarray:
        """Load and normalize the base image."""
        form = np.load(RESOURCE_PATH / f'{self.file_name}.npy')
        return form / np.max(form)

class CustomSample(Sample):
    file_name = 'custom'

class BaboonSample(Sample):
    file_name = 'baboon'
    
class BarbaraSample(Sample):
    file_name = 'barbara'

class CameramanSample(Sample):
    file_name = 'cameraman'

class CheckerboardSample(Sample):
    file_name = 'checkerboard'

class HouseSample(Sample):
    file_name = 'house'

class IndianSample(Sample):
    file_name = 'indian'

class LenaSample(Sample):
    file_name = 'lena'

class PeppersSample(Sample):
    file_name = 'peppers'

class SheppLoganSample(Sample):
    file_name = 'shepp_logan'

class ShipSample(Sample):
    file_name = 'ship'

class UniformSample(Sample):
    file_name = 'uniform'
