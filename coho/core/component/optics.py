"""Optics."""

import numpy as np

# Local imports
from . import Component

__all__ = [
    'CircularOptic',
    'CodedOptic',
    'CustomOptic',
    'GaussianOptic',
    'RectangularOptic',
]

class Optic(Component):
    """Base optic class."""

    def __init__(self, properties):
        """Initialize the optic with specified properties."""
        super().__init__(properties)

        # Generate the complex form
        self.form = np.expand_dims(self.generate_form(), axis=0)

class CircularOptic(Optic):
    def generate_form(self) -> np.ndarray:
        """Generate a circular optic image."""
        radius = self.profile.radius
        size = self.size
        y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
        center = 0
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        form = np.zeros((size, size))
        form[mask] = 1
        return form

class CodedOptic(Optic):
    def generate_form(self) -> np.ndarray:
        """Generate a coded optic image."""
        bit_size = int(self.profile.bit_size)
        size = self.profile.size
        seed = self.profile.seed
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Generate the bit pattern
        num_bits = size // bit_size
        bits = np.random.choice([0, 1], size=(num_bits, num_bits))
        pattern = np.kron(bits, np.ones((bit_size, bit_size)))
        
        # Center the pattern
        form = np.zeros((size, size))
        start = (size - pattern.shape[0]) // 2
        end = start + pattern.shape[0]
        form[start:end, start:end] = pattern
        return form

class CustomOptic(Optic):
    def generate_form(self) -> np.ndarray:
        """Generate a custom optic form."""
        path = self.profile.path
        form = np.load(path)
        return form / np.max(form)
    
class GaussianOptic(Optic):
    def generate_form(self) -> np.ndarray:
        """Generate a Gaussian optic form."""
        sigma = self.profile.sigma 
        size = self.profile.size
        x = np.linspace(-size / 2, size / 2, size)
        y = np.linspace(-size / 2, size / 2, size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        return np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))

class RectangularOptic(Optic):
    def generate_form(self) -> np.ndarray:
        """Generate a rectangular optic form."""
        width = self.profile.width
        height = self.profile.height
        size = self.profile.size
        form = np.zeros((size, size))
        form[size//2-height//2:size//2+height//2, 
              size//2-width//2:size//2+width//2] = 1
        return form
