"""Classes for wavefront propagation."""

import numpy as np
from typing import Union

# Local imports
from . import Operator
from ..component.wavefronts import Wavefront
    

__all__ = [
    'FresnelPropagate',
    'FourierPropagate',
    ]

class FresnelPropagate(Operator):
    """Fresnel propagation operator."""
    
    def __init__(self):
        """Initialize the Fresnel propagation operator."""
        # Cache frequency grid
        self._f2 = None 
    
    def _get_f2(self, size: int, spacing: float) -> np.ndarray:
        """Create or retrieve cached frequency grid."""
        if self._f2 is None:
            fx = np.fft.fftfreq(size, spacing)
            fy = np.fft.fftfreq(size, spacing)
            fxx, fyy = np.meshgrid(fx, fy, indexing='ij')
            self._f2 = fxx**2 + fyy**2
        return self._f2
    
    def apply(self, wavefront: Wavefront, distance: Union[float, list, np.ndarray]) -> Wavefront:
        """Forward Fresnel propagation."""
        # Get current wavefront properties
        size = wavefront.profile.size
        spacing = wavefront.physical.spacing
        wavelength = wavefront.wavelength

        # Prepare frequency grid and propagation kernel
        self._get_f2(size, spacing)
        distance = np.asarray(distance).reshape(-1, 1, 1)
        kernel = np.exp(-1j * wavelength * distance * self._f2)

        # Perform propagation in Fourier space
        wavefront.complexform = np.fft.ifft2(
            np.fft.fft2(wavefront.complexform, axes=(-2, -1)) * kernel, 
            axes=(-2, -1)
        )

        return wavefront
    
    def adjoint(self, wavefront: Wavefront, distance: Union[float, list, np.ndarray]) -> Wavefront:
        """Adjoint Fresnel propagation.""" 
        return self.apply(wavefront, -distance)
    
class FourierPropagate(Operator):
    """Fourier propagation operator."""
    
    def apply(self, wavefront: Wavefront) -> Wavefront:
        """Forward Fourier propagation."""
        wavefront.complexform = np.fft.fftshift(np.fft.fft2(wavefront.complexform, axes=(-2, -1)))
        return wavefront
    
    def adjoint(self, wavefront: Wavefront) -> Wavefront:
        """Adjoint Fourier propagation."""
        wavefront.complexform = np.fft.ifftshift(np.fft.ifft2(wavefront.complexform, axes=(-2, -1)))
        return wavefront
