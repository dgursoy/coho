"""Holographic scan experiment."""

# Standard imports
from typing import Union, List
import numpy as np

# Local imports
from .base import Pipeline
from ..operator import Broadcast, Propagate, Modulate, Detect
from ..component import Wave

class MultiDistanceHolography(Pipeline):
    """Multi-distance holography pipeline."""
    
    def __init__(self, 
                 reference: Wave, 
                 detector: Wave, 
                 sample_positions: Union[np.ndarray, float, List[float]]
        ):
        """Initialize with reference and detector waves."""
        self.sample_positions = np.atleast_1d(sample_positions)
        
        # Prepare reference wave
        prepared_reference = self._prepare_reference(reference)
        
        # Prepare detector wave
        detector_positions = np.full_like(self.sample_positions, detector.position)
        prepared_detector = Broadcast('position').apply(detector, values=detector_positions)
        
        # Calculate propagation distances
        sample_to_detector = detector.position - self.sample_positions
        
        # Initialize pipeline with prepared waves
        super().__init__([
            (Broadcast('position'), {'values': self.sample_positions}),
            (Modulate(), {'modulator': prepared_reference}),
            (Propagate(), {'distance': sample_to_detector}),
            (Modulate(), {'modulator': prepared_detector}),
            (Detect(), {})
        ])
    
    def _prepare_reference(self, reference: Wave) -> Wave:
        """Prepare reference wave by broadcasting and propagating."""
        wave_positions = np.full_like(self.sample_positions, reference.position)
        wave = Broadcast('position').apply(reference, values=wave_positions)
        distances = self.sample_positions - wave.position
        return Propagate().apply(wave, distance=distances)
    
    def apply(self, sample: Wave) -> np.ndarray:
        """Forward pipeline: sample to intensity."""
        return super().apply(sample)
    
    def adjoint(self, intensity: np.ndarray) -> Wave:
        """Adjoint pipeline: intensity to sample."""
        return super().adjoint(intensity)
    
    def __str__(self) -> str:
        """Simple string representation."""
        return f"Multi-distance holography at {len(self.reference.position)} positions"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}( \
            reference={self.reference.position.tolist()}, \
            detector={self.detector.position.tolist()})"
