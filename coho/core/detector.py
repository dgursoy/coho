# core/detector.py

"""Optical detector implementations.

This module provides detector classes for wavefront measurements.

Classes:
    Detector: Abstract base detector
        Methods:
            record_intensity: Record measurement
            acquire_images: Get recorded data
            
    IntegratingDetector: Continuous intensity detector
        Methods:
            record_intensity: Record |E|² intensity
            
    PhotonCountingDetector: Discrete photon detector (planned)
        Methods:
            record_intensity: Not implemented

Attributes:
    DEFAULTS: Default detector parameters
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np


# Default parameters
DEFAULTS = {
    "resolution": 512,    # Grid points
    "pixel_size": 0.001,  # cm
    "position": 5.0       # cm
}


class Detector(ABC):
    """Abstract base detector class.
    
    Attributes:
        id: Unique identifier
        images: List of measurements
        parameters: Configuration dict
    """

    def __init__(self, id: Any, parameters: Optional[Dict[str, Any]] = None):
        """Initialize detector.

        Args:
            id: Unique identifier
            parameters: Configuration dict
                resolution: Grid points
                pixel_size: Pixel size (cm)
                position: Z position (cm)
        """
        self.id = id
        self.parameters = parameters or DEFAULTS.copy()
        self.images: List[np.ndarray] = []

    @abstractmethod
    def record_intensity(self, amplitude: np.ndarray) -> None:
        """Record wavefront measurement.

        Args:
            amplitude: Wavefront amplitude array
        """
        pass

    def acquire_images(self) -> List[np.ndarray]:
        """Get recorded measurements.

        Returns:
            List of measurement arrays
        """
        return self.images


class IntegratingDetector(Detector):
    """Continuous intensity detector.
    
    Measures wavefront intensity as |E|² from amplitude.
    Suitable for high-light conditions.
    """

    def record_intensity(self, amplitude: np.ndarray) -> None:
        """Record intensity measurement.

        Computes intensity as |E|² = |amplitude|²

        Args:
            amplitude: Wavefront amplitude array
        """
        intensity = abs(amplitude) ** 2
        self.images.append(intensity)


class PhotonCountingDetector(Detector):
    """Discrete photon counting detector.
    
    Future implementation will include:
    - Poisson statistics
    - Quantum efficiency
    - Dark counts
    """

    def record_intensity(self, amplitude: np.ndarray) -> None:
        """Record photon counts (not implemented).

        Args:
            amplitude: Wavefront amplitude array

        Raises:
            NotImplementedError: Not yet implemented
        """
        raise NotImplementedError("Photon counting not implemented")
