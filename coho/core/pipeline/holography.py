"""Holographic scan experiment."""

# Standard imports
from typing import Union, List
import torch

# Local imports
from .base import Pipeline
from ..operator import Broadcast, Propagate, Modulate, Detect, Shift, Crop
from ..component import Wave

class MultiDistanceHolography(Pipeline):
    """Multi-distance holography pipeline."""
    
    def __init__(self, 
                 reference: Wave, 
                 detector: Wave, 
                 sample_positions: Union[torch.Tensor, float, List[float]]
        ):
        """Initialize with reference and detector waves."""
        self.sample_positions = torch.as_tensor(sample_positions, dtype=torch.float64)
        
        # Prepare waves
        self.reference = self._prepare_reference(reference)
        self.detector = self._prepare_detector(detector)
        
        # Calculate distances
        self.sample_to_detector = detector.position - self.sample_positions
        
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
        wave_positions = torch.full_like(self.sample_positions, reference.position)
        wave = Broadcast().apply(reference, {'position': wave_positions})
        distances = self.sample_positions - wave_positions
        return Propagate().apply(wave, distances)
    
    def _prepare_detector(self, detector: Wave) -> Wave:
        """Prepare detector wave by broadcasting and propagating."""
        detector_positions = torch.full_like(self.sample_positions, detector.position)
        return Broadcast().apply(detector, {'position': detector_positions})
    
    def apply(self, sample: Wave) -> torch.Tensor:
        """Forward pipeline: sample to intensity."""
        return super().apply(sample)
    
    def adjoint(self, intensity: torch.Tensor) -> Wave:
        """Adjoint pipeline: intensity to sample."""
        return super().adjoint(intensity)

class CodedHolography(Pipeline):
    """Coded holography pipeline."""
    
    def __init__(self, 
                 reference: Wave, 
                 detector: Wave, 
                 sample: Wave,
                 code: Wave,
                 code_position_x: Union[torch.Tensor, float, List[float]],
                 code_position_y: Union[torch.Tensor, float, List[float]]
        ):
        """Initialize with reference and detector waves."""
        self.code_position_x = torch.as_tensor(code_position_x, dtype=torch.float64)
        self.code_position_y = torch.as_tensor(code_position_y, dtype=torch.float64)
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
    
    def apply(self, sample: Wave) -> torch.Tensor:
        """Forward pipeline: sample to intensity."""
        return super().apply(sample)
    
    def adjoint(self, intensity: torch.Tensor) -> Wave:
        """Adjoint pipeline: intensity to sample."""
        return super().adjoint(intensity)