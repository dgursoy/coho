"""Holographic scan experiment."""

# Standard imports
from typing import Union, List, Tuple, Dict
import numpy as np

# Local imports
from .base import Pipeline
from ..operator import Broadcast, Propagate, Modulate, Detect, Shift, Crop, Operator
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
        
        # Prepare waves
        self.reference = self._prepare_reference(reference)
        self.detector = self._prepare_detector(detector)
        
        # Calculate distances
        self.sample_to_detector = np.subtract(detector.position, self.sample_positions)
        
        # Initialize pipeline
        super().__init__([
            (Broadcast(), {'values': {'position': self.sample_positions}}),
            (Modulate(), {'modulator': self.reference}),
            (Propagate(), {'distance': self.sample_to_detector}),
            (Modulate(), {'modulator': self.detector}),
            (Detect(), {})
        ])
    
    def _prepare_reference(self, reference: Wave) -> Wave:
        """Prepare reference wave by broadcasting and propagating."""
        wave_positions = np.full_like(self.sample_positions, reference.position)
        wave = Broadcast().apply(reference, {'position': wave_positions})
        distances = np.subtract(self.sample_positions, wave_positions)
        return Propagate().apply(wave, distances)
    
    def _prepare_detector(self, detector: Wave) -> Wave:
        """Prepare detector wave by broadcasting and propagating."""
        detector_positions = np.full_like(self.sample_positions, detector.position)
        return Broadcast().apply(detector, {'position': detector_positions})
    
    def apply(self, sample: Wave) -> np.ndarray:
        """Forward pipeline: sample to intensity."""
        return super().apply(sample)
    
    def adjoint(self, intensity: np.ndarray) -> Wave:
        """Adjoint pipeline: intensity to sample."""
        return super().adjoint(intensity)
    
class CodedHolography(Pipeline):
    """Coded holography pipeline."""
    
    def __init__(self, 
                 reference: Wave, 
                 detector: Wave, 
                 sample: Wave,
                 code: Wave,
                 code_position_x: Union[np.ndarray, float, List[float]],
                 code_position_y: Union[np.ndarray, float, List[float]]
        ):
        """Initialize with reference and detector waves."""
        self.code_position_x = np.atleast_1d(code_position_x)
        self.code_position_y = np.atleast_1d(code_position_y)
        self.code = code
        
        # Prepare reference wave
        wave = Broadcast().apply(reference, 
                                 values={'x': self.code_position_x, 
                                         'y': self.code_position_y})
        distance = self.code.position - wave.position
        prepared_reference = Propagate().apply(wave, distance=distance)

        # Prepare code wave
        prepared_code = Broadcast().apply(code, 
                                          values={'x': self.code_position_x, 
                                                  'y': self.code_position_y})
        
        prepared_code = Shift().apply(prepared_code, self.code_position_y, self.code_position_x)
        prepared_code = Crop().apply(prepared_reference, prepared_code)
        
        prepared_reference = Modulate().apply(prepared_reference, prepared_code)
        prepared_reference = Propagate().apply(prepared_reference, sample.position - prepared_reference.position)
        
        # Prepare detector wave
        prepared_detector = Broadcast().apply(detector, 
                                              values={'x': self.code_position_x, 
                                                      'y': self.code_position_y})

        # Calculate propagation distances
        sample_to_detector = detector.position - prepared_reference.position
        
        # Initialize pipeline with prepared waves
        super().__init__([
            (Broadcast(), {'values': {'x': self.code_position_x, 'y': self.code_position_y}}),
            (Modulate(), {'modulator': prepared_reference}),
            (Propagate(), {'distance': sample_to_detector}),
            (Modulate(), {'modulator': prepared_detector}),
            (Detect(), {})
        ])
    
    def apply(self, sample: Wave) -> np.ndarray:
        """Forward pipeline: sample to intensity."""
        return super().apply(sample)
    
    def adjoint(self, intensity: np.ndarray) -> Wave:
        """Adjoint pipeline: intensity to sample."""
        return super().adjoint(intensity)