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

RESOURCE_PATH = Path(__file__).parent.parent.parent / 'resources' / 'profiles'

class Sample(Component):
    """Base sample class."""
    pass

class CustomSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a custom sample profile."""
        file_path = self.profile.file_path
        image = np.load(file_path)
        return image / np.max(image)

class BaboonSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a baboon sample profile."""
        image = np.load(RESOURCE_PATH / 'baboon.npy')
        return image / np.max(image)
    
class BarbaraSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a Barbara sample profile."""
        image = np.load(RESOURCE_PATH / 'barbara.npy')
        return image / np.max(image)

class CameramanSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a cameraman sample profile."""
        image = np.load(RESOURCE_PATH / 'cameraman.npy')
        return image / np.max(image)

class CheckerboardSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a checkerboard sample profile."""
        image = np.load(RESOURCE_PATH / 'checkerboard.npy')
        return image / np.max(image)
    
class HouseSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a house sample profile."""
        image = np.load(RESOURCE_PATH / 'house.npy')
        return image / np.max(image)

class IndianSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a Indian sample profile."""
        image = np.load(RESOURCE_PATH / 'indian.npy')
        return image / np.max(image)

class LenaSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a Lena sample profile."""
        image = np.load(RESOURCE_PATH / 'lena.npy')
        return image / np.max(image)

class PeppersSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a peppers sample profile."""
        image = np.load(RESOURCE_PATH / 'peppers.npy')
        return image / np.max(image)

class SheppLoganSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a Shepp Logan sample profile."""
        image = np.load(RESOURCE_PATH / 'shepp_logan.npy')
        return image / np.max(image)

class ShipSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a ship sample profile."""
        image = np.load(RESOURCE_PATH / 'ship.npy')
        return image / np.max(image)

class UniformSample(Sample):
    def _generate_image(self) -> np.ndarray:
        """Generate a uniform sample profile."""
        return np.ones((self.size, self.size))
