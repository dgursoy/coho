"""Classes for simulating wavefront interactions with optical elements."""

import numpy as np
import xraylib
from typing import Union

# Local imports
from . import Operator
from ..component.wavefronts import Wavefront
from ..component.optics import Optic
from ..component.samples import Sample
from ..component.detectors import Detector

__all__ = [
    'Detect',
    'Interact',
]

class Detect(Operator):
    """Detection handler."""

    def __init__(self):
        """Initialize the detection handler."""
        self.wavefront = None

    def _set_wavefront(self, wavefront: Wavefront):
        """Set the wavefront."""
        self.wavefront = wavefront
    
    def apply(self, wavefront: Wavefront, component: Detector) -> np.ndarray:
        """Apply square of the absolute value."""
        self._set_wavefront(wavefront)
        return np.square(np.abs(wavefront.phasor))
    
    def adjoint(self, intensity: np.ndarray, component: Detector) -> Wavefront:
        """Adjoint of square root of the absolute value."""
        self.wavefront.phasor = np.sqrt(intensity) * np.exp(1j * 0)
        return self.wavefront

class Interact(Operator):
    """Wavefront modulation handler."""

    def _compute_refractive_index(self, wavefront: Wavefront, component: Union[Optic, Sample, Detector]) -> float:
        """Compute the complex refractive index of the optical component."""
        return xraylib.Refractive_Index(
            component.physical.formula,
            wavefront.physical.energy,
            component.physical.density
        )
    
    def _interact(self, wavefront: Wavefront, component: Union[Optic, Sample, Detector]) -> float:
        """Compute phase shift."""
        refractive_index = self._compute_refractive_index(wavefront, component)
        return wavefront.wavenumber * (refractive_index - 1) * component.physical.thickness * component.image
        
    def apply(self, wavefront: Wavefront, component: Union[Optic, Sample, Detector]) -> Wavefront:
        """Apply amplitude and phase modifications."""
        modulation = self._interact(wavefront, component)
        wavefront.phasor *= np.exp(1j * modulation)
        return wavefront
    
    def adjoint(self, wavefront: Wavefront, component: Union[Optic, Sample, Detector]) -> Wavefront:
        """Reverse amplitude and phase modifications."""
        modulation = self._interact(wavefront, component)
        wavefront.phasor *= np.exp(-1j * modulation)
        return wavefront

