# core/simulation/detector.py

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
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from coho.config.models import DetectorProperties
from coho.core.simulation.wavefront import Wavefront

__all__ = [
    'IntegratingDetector',
    'PhotonCountingDetector'
]

class Detector(ABC):
    """Abstract base detector class.
    """

    def __init__(self, properties: DetectorProperties):
        """Initialize detector.
        """
        self.properties = properties
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize detector images.
        """
        self.images: List[np.ndarray] = []

    @abstractmethod
    def detect(self, wavefront: Wavefront) -> None:
        """Record wavefront measurement."""
        pass

    def acquire(self) -> List[np.ndarray]:
        """Get recorded measurements.""
        """""
        return self.images


class IntegratingDetector(Detector):
    """Continuous intensity detector."""

    def detect(self, wavefront: Wavefront) -> None:
        """Record intensity measurement."""
        intensity = abs(wavefront.complex_wavefront) ** 2
        self.images.append(intensity)

class PhotonCountingDetector(Detector):
    """Discrete photon counting detector."""

    def detect(self, wavefront: Wavefront) -> None:
        """Record photon counts (not implemented). """
        raise NotImplementedError("PhotonCountingDetector not implemented")
