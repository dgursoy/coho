"""Classes for simulating wavefront interactions with optical elements."""

# Standard imports
import torch

# Local imports
from .base import Operator, TensorLike
from ..component import Wave
from ..utils.decorators import (
    requires_matching,
    as_tensor
)

__all__ = [
    'Modulate', 
    'Detect',
    'Shift',
    'Crop'
]

class Modulate(Operator):
    """Modulate wavefront by another wavefront."""

    @requires_matching('energy', 'spacing', 'position')
    def apply(self, reference: Wave, modulator: Wave) -> Wave:
        """Forward modulation."""
        return reference * modulator

    @requires_matching('energy', 'spacing', 'position')
    def adjoint(self, reference: Wave, modulator: Wave) -> Wave:
        """Adjoint modulation."""
        return reference / modulator

class Detect(Operator):
    """Detect wavefront intensity."""

    def apply(self, wave: Wave) -> torch.Tensor:
        """Wavefront to intensity."""
        self.wave = wave  # Save wave for adjoint
        return wave.amplitude
    
    def adjoint(self, intensity: torch.Tensor) -> Wave:
        """Intensity to wavefront."""
        self.wave.form = intensity  # No phase information
        return self.wave

class Shift(Operator):
    """Shift operator."""

    @as_tensor('y', 'x')
    def apply(self, wave: Wave, y: TensorLike, x: TensorLike) -> Wave:
        """Apply shifts to wave."""
        # Shift each form in batch
        shifted_forms = []
        for form, y, x in zip(wave.form, y, x):
            shifted = torch.roll(torch.roll(form, int(y), dims=-2), int(x), dims=-1)
            shifted_forms.append(shifted)
        wave.form = torch.stack(shifted_forms)
        return wave

    @as_tensor('y', 'x')
    def adjoint(self, wave: Wave, y: TensorLike, x: TensorLike) -> Wave:
        """Adjoint shift operation."""
        return self.apply(wave, -y, -x)

class Crop(Operator):
    """Crop wavefront to match dimensions of another wave."""

    def apply(self, reference: Wave, modulator: Wave) -> Wave:
        """Forward crop operation: modifies modulator to match reference size."""
        return modulator.crop_to_match(reference, pad_value=1.0)

    def adjoint(self, reference: Wave, modulator: Wave) -> Wave:
        """Adjoint crop operation: restores original modulator shape."""
        return reference.crop_to_match(modulator, pad_value=1.0)

