# core/simulation/wavefront.py

"""Classes for representing optical wavefront states.

This module defines wavefront profiles and their interactions
with propagation systems.

Classes:
    Wavefront: Base class for all wavefront types
    ConstantWavefront: Uniform amplitude and phase
    GaussianWavefront: Gaussian profile
    RectangularWavefront: Rectangular profile
    BatchWavefront: Container for multiple wavefronts with varying parameters
"""

from abc import ABC, abstractmethod
import numpy as np
from coho.config.models import WavefrontProperties
from ..experiment.batcher import Batch
from scipy.ndimage import rotate, shift

__all__ = [
    'ConstantWavefront',
    'GaussianWavefront',
    'RectangularWavefront',
    'BatchWavefront',
]

class Wavefront(ABC):
    """Base class for optical wavefronts."""
    
    def __init__(self, properties: WavefrontProperties):
        """
        Initialize the wavefront with specified properties.
        """
        self.properties = properties
        self._initialize_wavefront()

    def _initialize_wavefront(self):
        """Initialize both amplitude and phase patterns."""
        self.profile = self.generate_profile()
        self.profile = self._apply_rotation()
        self.profile = self._apply_translation()
        amplitude_profile = self.profile * self.properties.physical.amplitude
        phase_profile = self.profile * self.properties.physical.phase
        self.complex_wavefront = amplitude_profile * np.exp(1j * phase_profile)

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
    def generate_profile(self) -> np.ndarray:
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


class ConstantWavefront(Wavefront):
    """Wavefront with uniform amplitude and phase."""

    def generate_profile(self) -> np.ndarray:
        """Generate a uniform profile."""
        return np.ones((self.size, self.size))


class GaussianWavefront(Wavefront):
    """Wavefront with a Gaussian amplitude profile."""

    def generate_profile(self) -> np.ndarray:
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

    def generate_profile(self) -> np.ndarray:
        """
        Generate a rectangular profile.

        Returns:
            np.ndarray: Binary rectangle array over the grid.
        """
        width = self.properties.profile.width
        height = self.properties.profile.height

        pattern = np.zeros((self.size, self.size))
        x_start = (self.size - width) // 2
        y_start = (self.size - height) // 2
        pattern[y_start:y_start + height, x_start:x_start + width] = 1.0
        return pattern


class BatchWavefront(Batch):
    """Container for multiple wavefronts with varying parameters."""
    
    def __init__(self, component_class, base_properties, parameter_arrays):
        """Initialize batch wavefront container.
        
        Args:
            component_class: Wavefront class to instantiate
            base_properties: Base properties for all wavefronts
            parameter_arrays: Dict of parameter paths and their value arrays
        """
        super().__init__(component_class, base_properties, parameter_arrays)
        self.complex_wavefronts = np.array([state.complex_wavefront for state in self.states]) 
