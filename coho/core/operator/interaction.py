"""Classes for simulating wavefront interactions with optical elements."""

import numpy as np
import xraylib
from typing import Union, TypeAlias

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

# Type alias for components
Component: TypeAlias = Union[Optic, Sample, Detector]

class Detect(Operator):
    """Detection handler."""

    def __init__(self, wavefront: Wavefront = None):
        """Initialize the detection operator."""
        # Cache the wavefront
        self.wavefront = wavefront
    
    def apply(self, wavefront: Wavefront, component: Detector) -> np.ndarray:
        """Convert complex wavefront to measured intensity."""
        self.wavefront = wavefront
        return np.abs(wavefront.complexform) ** 2  
    
    def adjoint(self, intensity: np.ndarray, component: Detector) -> Wavefront:
        """Convert measured intensity back to complex wavefront."""
        self.wavefront.complexform = np.sqrt(intensity)  # amplitude only, phase = 0
        return self.wavefront

class Interact(Operator):
    """Wavefront modulation handler."""

    def _compute_refractive_index(self, wavefront: Wavefront, component: Component) -> np.ndarray:
        """Compute the complex refractive index of the optical component."""
        # Get values directly without intermediate reshaping
        formula = component.physical.formula
        energy = wavefront.physical.energy
        density = component.physical.density
        return xraylib.Refractive_Index(formula, energy, density)
    
    def _get_exponent(self, wavefront: Wavefront, component: Component) -> float:
        """Compute the exponent of the transfer function."""
        # Get terms with proper shapes
        wavenumber = wavefront.wavenumber
        thickness = component.physical.thickness
        delta_n = self._compute_refractive_index(wavefront, component) - 1 
        return (wavenumber * delta_n * thickness) * component.form
    
    def _apply_transfer_function(self, wavefront: Wavefront, component: Component, conjugate: bool = False) -> Wavefront:
        """Apply transfer function."""
        # Generate transfer function if not specified by the component
        if component.complexform is None:
            exponent = self._get_exponent(wavefront, component)
            sign = -1 if conjugate else 1
            transfer_function = np.exp(sign * 1j * exponent)
            component.complexform = transfer_function
            
        # Handle tiling in forward direction
        if component.complexform.shape[0] > wavefront.complexform.shape[0]:
            # Ensure complex dtype when tiling
            wavefront.complexform = np.tile(wavefront.complexform, 
                                          (component.complexform.shape[0], 1, 1)).astype(np.complex128)
            
        # Ensure both arrays are complex before multiplication
        if not np.iscomplexobj(wavefront.complexform):
            wavefront.complexform = wavefront.complexform.astype(np.complex128)
            
        wavefront.complexform *= component.complexform
        return wavefront
    
    def apply(self, wavefront: Wavefront, component: Component) -> Wavefront:
        """Forward propagation through the optical component."""
        return self._apply_transfer_function(wavefront, component)
    
    def adjoint(self, wavefront: Wavefront, component: Component) -> Wavefront:
        """Reverse propagation through the optical component."""
        # Apply conjugate transfer function
        wavefront = self._apply_transfer_function(wavefront, component, conjugate=True)
        
        # Handle un-tiling (mean) in adjoint direction
        if component.complexform.shape[0] > 1 and wavefront.complexform.shape[0] > 1:
            wavefront.complexform = np.mean(wavefront.complexform, axis=0, keepdims=True)
            
        return wavefront

