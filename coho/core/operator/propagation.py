"""Classes for wavefront propagation."""

import numpy as np

# Local imports
from . import Operator
from ..component.wavefronts import Wavefront

__all__ = [
    'FresnelPropagate'
    ]

class FresnelPropagate(Operator):
    """Near-field (Fresnel) propagator."""

    def apply(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Propagate with Fresnel method."""
        # Update wavefront position
        wavefront.geometry.distance += distance

        # Setup frequency grid
        nx, ny = wavefront.phasor.shape
        x = np.fft.fftfreq(nx, d=wavefront.spacing)
        y = np.fft.fftfreq(ny, d=wavefront.spacing)
        fx, fy = np.meshgrid(x, y)

        # Create propagation kernel
        kernel = np.exp(-1j * wavefront.wavelength * distance * (fx**2 + fy**2))

        # Apply propagation
        field_ft = np.fft.fft2(wavefront.phasor)
        field_ft_prop = field_ft * kernel
        wavefront.phasor = np.fft.ifft2(field_ft_prop)
        return wavefront
    
    def adjoint(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Adjoint model."""
        return self.apply(wavefront, -distance)
