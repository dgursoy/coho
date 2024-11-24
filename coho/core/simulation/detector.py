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

from abc import ABC
from typing import List, Optional
import numpy as np
from coho.config.models import DetectorProperties
from coho.core.simulation.wavefront import Wavefront, BatchWavefront

__all__ = [
    'IntegratingDetector',
    'PhotonCountingDetector',
    'BatchDetector'
]

class Detector(ABC):
    """Base detector class."""
    
    def __init__(self, properties: Optional[DetectorProperties] = None):
        self.properties = properties or {}
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize detector images."""
        self.images: List[np.ndarray] = []

    def detect(self, wavefront: Wavefront) -> np.ndarray:
        """Record and return wavefront measurement."""
        intensity = abs(wavefront.complex_wavefront) ** 2
        self.images.append(intensity)
        return intensity

    def acquire(self) -> np.ndarray:
        """Get all recorded measurements."""
        return np.array(self.images)


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
    

class BatchDetector(Detector):
    """Detector for vectorized batch measurements.
    
    This class enables vectorized operations on batches of wavefronts,
    avoiding loops and improving computational efficiency.
    """
    
    def detect(self, batch_wavefront: BatchWavefront) -> np.ndarray:
        """Record and return vectorized measurements for batch of wavefronts.
        
        Args:
            batch_wavefront: Batch of wavefronts to measure
            
        Returns:
            np.ndarray: Array of measurements (batch_size, height, width)
                computed using vectorized operations
        """
        # Vectorized intensity calculation for all wavefronts at once
        intensities = np.abs(batch_wavefront.complex_wavefronts) ** 2
        return intensities
    