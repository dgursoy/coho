# core/simulation/wavefront.py

"""Classes for representing optical wavefront states.

This module defines wavefront profiles and their interactions
with propagation systems.

Classes:
    Wavefront: Base class for all wavefront types
    ConstantWavefront: Uniform amplitude and phase
    GaussianWavefront: Gaussian profile
    RectangularWavefront: Rectangular profile

Constants:
    PHYSICAL_CONSTANTS: Fundamental physics constants
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


PHYSICAL_CONSTANTS = {
    'PLANCK_CONSTANT': 6.58211928e-19,  # keV*s
    'SPEED_OF_LIGHT': 299792458e+2,     # cm/s
}


class Wavefront(ABC):
    """Base class for optical wavefronts."""

    def __init__(self, id: Any, parameters: Optional[Dict[str, Any]] = None):
        """Initialize wavefront.

        Args:
            id: Unique identifier
            parameters: Configuration dict
                energy: Photon energy (keV)
                shape: Grid size (pixels)
                spacing: Grid spacing (cm)
                amplitude: Base amplitude
                phase: Base phase (rad)
        """
        parameters = parameters or {}
        self.id = id
        self.energy = parameters.get('physical', {}).get('energy')
        self.shape = parameters.get('grid', {}).get('size')
        self.spacing = parameters.get('grid', {}).get('spacing')
        
        self.amplitude = self.generate_amplitude(parameters)
        self.phase = self.generate_phase(parameters)

    @property
    def wavelength(self) -> float:
        """Calculate wavelength.

        Returns:
            Wavelength in cm
        """
        return (2 * np.pi * PHYSICAL_CONSTANTS['PLANCK_CONSTANT'] * 
                PHYSICAL_CONSTANTS['SPEED_OF_LIGHT'] / self.energy)

    @property
    def wavenumber(self) -> float:
        """Calculate wavenumber.

        Returns:
            Wavenumber in cm⁻¹
        """
        return 2 * np.pi / self.wavelength

    def generate_amplitude(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate amplitude profile.

        Args:
            parameters: Amplitude settings

        Returns:
            Scaled amplitude array
        """
        amplitude = parameters.get('physical', {}).get("amplitude")
        return self.generate_pattern(parameters) * amplitude

    def generate_phase(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate phase profile.

        Args:
            parameters: Phase settings

        Returns:
            Scaled phase array
        """
        phase = parameters.get('physical', {}).get("phase")
        return self.generate_pattern(parameters) * phase

    @abstractmethod
    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate base pattern.

        Args:
            parameters: Pattern settings

        Returns:
            Pattern array
        """
        pass


class ConstantWavefront(Wavefront):
    """Uniform amplitude and phase wavefront."""

    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate uniform pattern.

        Args:
            parameters: Pattern settings

        Returns:
            Unit array
        """
        return np.ones((self.shape, self.shape))


class GaussianWavefront(Wavefront):
    """Gaussian profile wavefront."""

    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate Gaussian pattern.

        Args:
            parameters: Pattern settings
                sigma: Standard deviation
                shape: Grid size

        Returns:
            Gaussian array
        """
        sigma = parameters.get('profile', {}).get("sigma")

        x = np.linspace(-self.shape/2, self.shape/2, self.shape)
        y = np.linspace(-self.shape/2, self.shape/2, self.shape)
        xx, yy = np.meshgrid(x, y)
        return np.exp(-((xx**2 + yy**2) / (2 * sigma**2)))


class RectangularWavefront(Wavefront):
    """Rectangular profile wavefront."""

    def generate_pattern(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate rectangular pattern.

        Args:
            parameters: Pattern settings
                width: Rectangle width
                height: Rectangle height
                shape: Grid size

        Returns:
            Binary rectangle array
        """
        width = parameters.get('wave', {}).get("width")
        height = parameters.get('wave', {}).get("height")

        pattern = np.zeros((self.shape, self.shape))
        x_start = (self.shape - width) // 2
        y_start = (self.shape - height) // 2
        pattern[y_start:y_start + height, x_start:x_start + width] = 1.0
        return pattern
