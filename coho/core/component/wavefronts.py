"""Wavefronts."""

import numpy as np
from typing import Any

# Local imports
from . import Component

__all__ = [
    'UniformWavefront',
    'GaussianWavefront',
    'CustomWavefront',
]

class Wavefront(Component):
    """Base wavefront class."""
    
    def __init__(self, properties: Any):
        """Initialize the wavefront with specified properties."""
        super().__init__(properties)

        # Cache
        self._phasor = None

    @property
    def wavelength(self) -> float:
        """Wavelength derived from energy in keV."""
        return 1.23984193e-7 / self.physical.energy

    @property
    def wavenumber(self) -> float:
        """Wavenumber (2π divided by wavelength)."""
        return 2 * np.pi / self.wavelength

    @property
    def phasor(self):
        """Lazily compute and return the complex wavefront."""
        if self._phasor is None:
            amplitude = self.image * self.physical.amplitude
            phase = self.image * self.physical.phase
            self._phasor = amplitude * np.exp(1j * phase)
        return self._phasor

    @phasor.setter
    def phasor(self, value):
        """Set complex wavefront and clear image cache."""
        self._phasor = value

class CustomWavefront(Wavefront):
    def _generate_image(self) -> np.ndarray:
        """Generate a custom wavefront profile."""
        file_path = self.profile.file_path
        image = np.load(file_path)
        return image / np.max(image)
    
class GaussianWavefront(Wavefront):
    def _generate_image(self) -> np.ndarray:
        """Generate a Gaussian wavefront profile."""
        sigma = self.profile.sigma
        x = np.linspace(-self.size / 2, self.size / 2, self.size)
        y = np.linspace(-self.size / 2, self.size / 2, self.size)
        xx, yy = np.meshgrid(x, y)
        image = np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))
        return image
    
class UniformWavefront(Wavefront):
    def _generate_image(self) -> np.ndarray:
        """Generate a uniform wavefront profile."""
        return np.ones((self.size, self.size))
