# core/simulation/wavefront.py

"""Classes for representing optical wavefront states.

This module defines wavefront profiles and their interactions
with propagation systems.

Classes:
    Wavefront: Base class for all wavefront types
    ConstantWavefront: Uniform amplitude and phase
    GaussianWavefront: Gaussian profile
    RectangularWavefront: Rectangular profile
"""

from abc import ABC, abstractmethod
import numpy as np
from coho.config.models import WavefrontProperties
from scipy.ndimage import rotate, shift

__all__ = [
    'ConstantWavefront',
    'GaussianWavefront',
    'RectangularWavefront',
]


class Wavefront(ABC):
    """Base class for optical wavefronts."""
    
    def __init__(self, properties: WavefrontProperties):
        """Initialize the wavefront with specified properties."""
        self.properties = properties
        self._profile = None
        self._complex_wavefront = None

    @property
    def profile(self):
        """Lazily generate and return the profile."""
        if self._profile is None:
            self._profile = self._generate_profile()
            self._profile = self._apply_rotation()
            self._profile = self._apply_translation()
        return self._profile

    @property
    def complex_wavefront(self):
        """Lazily compute and return the complex wavefront."""
        if self._complex_wavefront is None:
            amplitude_profile = self.profile * self.properties.physical.amplitude
            phase_profile = self.profile * self.properties.physical.phase
            self._complex_wavefront = amplitude_profile * np.exp(1j * phase_profile)
        return self._complex_wavefront

    @property
    def wavelength(self) -> float:
        """Wavelength derived from energy in keV."""
        return 1.23984193e-7 / self.properties.physical.energy

    @property
    def wavenumber(self) -> float:
        """Wavenumber (2π divided by wavelength)."""
        return 2 * np.pi / self.wavelength

    @property
    def size(self) -> int:
        """Grid size for the wavefront."""
        return self.properties.grid.size

    @property
    def spacing(self) -> float:
        """Grid spacing for the wavefront."""
        return self.properties.grid.spacing

    @abstractmethod
    def _generate_profile(self) -> np.ndarray:
        """Generate the base pattern for the wavefront."""
        pass
        
    def _apply_rotation(self) -> np.ndarray:
        """Apply rotation to the profile."""
        rotation = self.properties.geometry.rotation
        return rotate(self.profile, rotation, reshape=False, order=1)
    
    def _apply_translation(self) -> np.ndarray:
        """Apply translation to the profile."""
        translation = self.properties.geometry.position
        return shift(self.profile, [translation.x, translation.y], order=1)

    def clear_cache(self):
        """Clear cached computations."""
        self._profile = None
        self._complex_wavefront = None


class ConstantWavefront(Wavefront):
    """Wavefront with uniform amplitude and phase."""

    def _generate_profile(self) -> np.ndarray:
        """Generate a uniform profile."""
        return np.ones((self.size, self.size))


class GaussianWavefront(Wavefront):
    """Wavefront with a Gaussian amplitude profile."""

    def _generate_profile(self) -> np.ndarray:
        """
        Generate a Gaussian profile.

        Returns:
            Gaussian distribution over the grid.
        """
        sigma = self.properties.profile.sigma

        x = np.linspace(-self.size / 2, self.size / 2, self.size)
        y = np.linspace(-self.size / 2, self.size / 2, self.size)
        xx, yy = np.meshgrid(x, y)
        return np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))


class RectangularWavefront(Wavefront):
    """Wavefront with a rectangular amplitude profile."""

    def _generate_profile(self) -> np.ndarray:
        """
        Generate a rectangular profile.

        Returns:
            np.ndarray: Binary rectangle array over the grid.
        """
        width = self.properties.profile.width
        height = self.properties.profile.height

        profile = np.zeros((self.size, self.size))
        x_start = (self.size - width) // 2
        y_start = (self.size - height) // 2
        profile[y_start:y_start + height, x_start:x_start + width] = 1.0
        return profile
