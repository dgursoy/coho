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
    pass

class CircularOptic(Optic):
    def _generate_image(self) -> np.ndarray:
        """Generate a circular optic image."""
        radius = self.profile.radius
        y, x = np.ogrid[:self.size, :self.size]
        center = self.size // 2
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        image = np.zeros((self.size, self.size))
        image[mask] = 1
        return image

class CodedOptic(Optic):
    def _generate_image(self) -> np.ndarray:
        """Generate a coded optic image."""
        bit_size = int(self.profile.bit_size)
        seed = self.profile.seed
        
        if seed is not None:
            np.random.seed(seed)
        
        image = np.zeros((self.size, self.size))
        num_bits = self.size // bit_size
        bits = np.random.choice([0, 1], size=(num_bits, num_bits))
        scaled_bits = np.kron(bits, np.ones((bit_size, bit_size)))
        
        start = (self.size - scaled_bits.shape[0]) // 2
        end = start + scaled_bits.shape[0]
        image[start:end, start:end] = scaled_bits
        return image

class CustomOptic(Optic):
    def _generate_image(self) -> np.ndarray:
        """Generate a custom optic image."""
        file_path = self.profile.file_path
        image = np.load(file_path)
        return image / np.max(image)
    
class GaussianOptic(Optic):
    def _generate_image(self) -> np.ndarray:
        """Generate a Gaussian optic image."""
        sigma = self.profile.sigma
        x = np.linspace(-self.size / 2, self.size / 2, self.size)
        y = np.linspace(-self.size / 2, self.size / 2, self.size)
        xx, yy = np.meshgrid(x, y)
        image = np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))
        return image

class RectangularOptic(Optic):
    def _generate_image(self) -> np.ndarray:
        """Generate a rectangular optic image."""
        width = self.profile.width
        height = self.profile.height
        image = np.zeros((self.size, self.size))
        image[self.size//2-height//2:self.size//2+height//2, 
                self.size//2-width//2:self.size//2+width//2] = 1
        return image
