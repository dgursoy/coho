"""Holographic scan experiment."""

# Standard imports
import torch

# Local imports
from .base import Pipeline, CompositeOperator, TensorLike
from ..component import Wave
from ..operator import Broadcast, Propagate, Modulate, Detect, Shift, Crop

class MultiDistanceHolographyTest(CompositeOperator):
    """Multi-distance holography pipeline."""
    
    def __init__(self):
        """Initialize with reference and detector waves."""
        
        # Initialize pipeline
        super().__init__([
            (Broadcast(), {'values': {'position': self.pos}}),
            (Modulate(), {'modulator': self.ref}),
            (Propagate(), {'distance': self.sample_to_det}),
            (Modulate(), {'modulator': self.det}),
            (Detect(), {})
        ])
    
    def apply(self, sample: Wave) -> torch.Tensor:
        """Forward pipeline: sample to intensity."""
        return super().apply(sample)
    
    def adjoint(self, intensity: torch.Tensor) -> Wave:
        """Adjoint pipeline: intensity to sample."""
        return super().adjoint(intensity)

class MultiDistanceHolography(Pipeline):
    """Multi-distance holography pipeline."""
    
    def __init__(self, ref: Wave, det: Wave, pos: TensorLike):
        """Initialize with reference and detector waves."""
        # Sample positions
        self.pos = torch.as_tensor(pos, dtype=torch.float64)
        
        # Prepare waves
        self.ref = self._prepare_ref(ref)
        self.det = self._prepare_det(det)
        
        # Calculate distances
        self.sample_to_det = det.position - self.pos
        
        # Initialize pipeline
        super().__init__([
            (Broadcast(), {'values': {'position': self.pos}}),
            (Modulate(), {'modulator': self.ref}),
            (Propagate(), {'distance': self.sample_to_det}),
            (Modulate(), {'modulator': self.det}),
            (Detect(), {})
        ])
    
    def _prepare_ref(self, ref: Wave) -> Wave:
        """Prepare ref wave by broadcasting and propagating."""
        wave_positions = torch.full_like(self.pos, ref.position)
        wave = Broadcast().apply(ref, {'position': wave_positions})
        distances = self.pos - wave_positions
        return Propagate().apply(wave, distances)
    
    def _prepare_det(self, det: Wave) -> Wave:
        """Prepare det wave by broadcasting and propagating."""
        det_positions = torch.full_like(self.pos, det.position)
        return Broadcast().apply(det, {'position': det_positions})
    
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
                 code_position_x: TensorLike,
                 code_position_y: TensorLike
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