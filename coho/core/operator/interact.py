"""Classes for simulating wavefront interactions with optical elements."""

# Standard imports
import numpy as np
from typing import Tuple

# Local imports
from coho.core.component import Wave
from .base import Operator
from .decorators import (
    validate_form,
    validate_matching_energy,
    validate_matching_position, 
    validate_matching_dimensions,
)

__all__ = [
    'Modulate', 
    'Detect',
    'Shift',
    'Crop'
    ]

class Modulate(Operator):
    """Modulate wavefront by another wavefront."""

    @validate_matching_dimensions
    @validate_matching_energy
    @validate_matching_position
    def apply(self, reference: Wave, modulator: Wave) -> Wave:
        """Forward modulation."""
        return reference * modulator

    @validate_matching_dimensions
    @validate_matching_energy
    @validate_matching_position
    def adjoint(self, reference: Wave, modulator: Wave) -> Wave:
        """Adjoint modulation."""
        return reference / modulator

class Detect(Operator):
    """Detect wavefront intensity."""

    def apply(self, wave: Wave) -> np.ndarray:
        """Wavefront to intensity."""
        self.wave = wave # Save wave for adjoint
        return wave.amplitude
    
    def adjoint(self, intensity: np.ndarray) -> Wave:
        """Intensity to wavefront."""
        self.wave.form = intensity # No phase information
        return self.wave
    
class Shift(Operator):
    """Shift operator."""

    @validate_form
    def apply(self, wave: Wave, y_shifts: np.ndarray, x_shifts: np.ndarray) -> Wave:
        """Apply shifts to wave."""
        # Shift each form in batch
        shifted_forms = [
            np.roll(np.roll(form, int(y), axis=-2), int(x), axis=-1)
            for form, y, x in zip(wave.form, y_shifts, x_shifts)
        ]
        wave.form = np.stack(shifted_forms)
        return wave
    
    def adjoint(self, wave: Wave, y_shifts: np.ndarray, x_shifts: np.ndarray) -> Wave:
        """Adjoint shift operation."""
        return self.apply(wave, -y_shifts, -x_shifts)

class Crop(Operator):
    """Crop wavefront to match dimensions of another wave."""

    def apply(self, reference: Wave, modulator: Wave) -> Wave:
        """Forward crop operation: modifies modulator to match reference size. """
        return modulator.crop_to_match(reference, pad_value=1.0)

    def adjoint(self, reference: Wave, modulator: Wave) -> Wave:
        """Adjoint crop operation: restores original modulator shape. """
        return reference.crop_to_match(modulator, pad_value=1.0)

