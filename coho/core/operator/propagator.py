# core/operator/propagator.py

"""Classes for wavefront propagation in optical simulations.

This module provides implementations of various propagation methods
for near-field (Fresnel) and far-field (Fraunhofer) calculations.

Classes:
    Propagator: Abstract base class for propagation methods
    FresnelPropagator: Near-field propagation implementation
    FraunhoferPropagator: Far-field propagation implementation

Methods:
    propagate: Move wavefront through space
    _construct_complex_field: Create complex field from wavefront
    _update_wavefront: Update wavefront from propagated field
"""

import numpy as np
from abc import ABC, abstractmethod
from coho.config.models import PropagatorProperties
from coho.core.simulation.wavefront import Wavefront

__all__ = [
    'FresnelPropagator', 'FraunhoferPropagator'
]

class Propagator(ABC):
    """Base class for wavefront propagation."""

    def __init__(self, properties: PropagatorProperties):
        """Initialize propagator."""
        self.properties = properties

    @abstractmethod
    def propagate(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Move wavefront through space."""
        wavefront.properties.geometry.position.z += distance

    def forward(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Forward model (alias for propagate)."""
        return self.propagate(wavefront, distance)
    
    def adjoint(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Adjoint model."""
        return self.propagate(wavefront, -distance)


class FresnelPropagator(Propagator):
    """Near-field (Fresnel) propagator."""

    def propagate(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Propagate with Fresnel method."""
        super().propagate(wavefront, distance)

        # Setup frequency grid
        nx, ny = wavefront.complex_wavefront.shape
        x = np.fft.fftfreq(nx, d=wavefront.spacing)
        y = np.fft.fftfreq(ny, d=wavefront.spacing)
        fx, fy = np.meshgrid(x, y)

        # Create propagation kernel
        kernel = np.exp(-1j * wavefront.wavelength * distance * (fx**2 + fy**2))

        # Apply propagation
        field_ft = np.fft.fft2(wavefront.complex_wavefront)
        field_ft_prop = field_ft * kernel
        wavefront.complex_wavefront = np.fft.ifft2(field_ft_prop)
        
        return wavefront


class FraunhoferPropagator(Propagator):
    """Far-field (Fraunhofer) propagator."""

    def propagate(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Propagate with Fraunhofer method."""
        super().propagate(wavefront, distance)
        raise NotImplementedError("FraunhoferPropagator not implemented")
