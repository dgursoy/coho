# core/wavefront.py

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
    WAVEFRONT_DEFAULTS: Default wavefront parameters
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

PHYSICAL_CONSTANTS = {
    'PLANCK_CONSTANT': 6.58211928e-19,  # keV*s
    'SPEED_OF_LIGHT': 299792458e+2,     # cm/s
}

WAVEFRONT_DEFAULTS = {
    'AMPLITUDE': 1.0,   # amplitude
    'PHASE': 0.0,       # radians
    'ENERGY': 10.0,     # keV
    'SHAPE': 512,       # pixels
    'SPACING': 0.001,   # cm
    'SIGMA': 64,        # pixels
    'WIDTH': 256,       # pixels
    'HEIGHT': 256,      # pixels
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
        self.energy = parameters.get('energy', WAVEFRONT_DEFAULTS['ENERGY'])
        self.shape = parameters.get('shape', WAVEFRONT_DEFAULTS['SHAPE'])
        self.spacing = parameters.get('spacing', WAVEFRONT_DEFAULTS['SPACING'])
        
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
        amplitude = parameters.get("amplitude", WAVEFRONT_DEFAULTS['AMPLITUDE'])
        return self.generate_pattern(parameters) * amplitude

    def generate_phase(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Generate phase profile.

        Args:
            parameters: Phase settings

        Returns:
            Scaled phase array
        """
        phase = parameters.get("phase", WAVEFRONT_DEFAULTS['PHASE'])
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
        shape = parameters.get("shape", WAVEFRONT_DEFAULTS['SHAPE'])
        return np.ones((shape, shape))


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
        sigma = parameters.get("sigma", WAVEFRONT_DEFAULTS['SIGMA'])
        shape = parameters.get("shape", WAVEFRONT_DEFAULTS['SHAPE'])

        x = np.linspace(-shape/2, shape/2, shape)
        y = np.linspace(-shape/2, shape/2, shape)
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
        shape = parameters.get("shape", WAVEFRONT_DEFAULTS['SHAPE'])
        width = parameters.get("width", WAVEFRONT_DEFAULTS['WIDTH'])
        height = parameters.get("height", WAVEFRONT_DEFAULTS['HEIGHT'])

        pattern = np.zeros((shape, shape))
        x_start = (shape - width) // 2
        y_start = (shape - height) // 2
        pattern[y_start:y_start + height, x_start:x_start + width] = 1.0
        return pattern
