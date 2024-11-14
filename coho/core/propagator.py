# core/propagator.py

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

from typing import Optional, Dict, Any  
import numpy as np
from abc import ABC, abstractmethod
from .wavefront import Wavefront


class Propagator(ABC):
    """Base class for wavefront propagation."""

    def __init__(self, id: Any, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialize propagator.

        Args:
            id: Unique identifier
            parameters: Configuration dict
        """
        self.id = id
        self.parameters = parameters or {}

    @abstractmethod
    def propagate(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Move wavefront through space.

        Args:
            wavefront: Input wavefront
            distance: Distance (meters)

        Returns:
            Modified wavefront

        Raises:
            ValueError: Negative distance
        """
        if distance < 0:
            raise ValueError("Distance must be positive")

    def _construct_complex_field(self, wavefront: Wavefront) -> np.ndarray:
        """Create complex field representation.

        Args:
            wavefront: Input wavefront

        Returns:
            Complex field array
        """
        return wavefront.amplitude * np.exp(1j * wavefront.phase)

    def _update_wavefront(self, wavefront: Wavefront, field: np.ndarray) -> Wavefront:
        """Update wavefront from field.

        Args:
            wavefront: Wavefront to update
            field: Propagated complex field

        Returns:
            Updated wavefront
        """
        wavefront.amplitude = np.abs(field)
        wavefront.phase = np.angle(field)
        return wavefront


class FresnelPropagator(Propagator):
    """Near-field (Fresnel) propagator.

    Uses FFT-based convolution with quadratic phase.
    Valid when:
    - Small propagation distance
    - Paraxial approximation holds
    - Parallel observation plane
    """

    def propagate(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Propagate with Fresnel method.

        Args:
            wavefront: Input wavefront
            distance: Distance (meters)

        Returns:
            Propagated wavefront
        """
        super().propagate(wavefront, distance)
        
        # Create complex field
        field = self._construct_complex_field(wavefront)

        # Setup frequency grid
        nx, ny = field.shape
        x = np.fft.fftfreq(nx, d=wavefront.spacing)
        y = np.fft.fftfreq(ny, d=wavefront.spacing)
        fx, fy = np.meshgrid(x, y)

        # Create propagation kernel
        kernel = np.exp(-1j * wavefront.wavelength * distance * (fx**2 + fy**2))

        # Apply propagation
        field_ft = np.fft.fft2(field)
        field_ft_prop = field_ft * kernel
        field_prop = np.fft.ifft2(field_ft_prop)

        return self._update_wavefront(wavefront, field_prop)


class FraunhoferPropagator(Propagator):
    """Far-field (Fraunhofer) propagator.

    Uses single Fourier transform.
    Valid when distance >> π(x²+y²)/λ:
    - x,y: aperture dimensions
    - λ: wavelength
    """

    def propagate(self, wavefront: Wavefront, distance: float) -> Wavefront:
        """Propagate with Fraunhofer method.

        Args:
            wavefront: Input wavefront
            distance: Distance (meters)

        Returns:
            Propagated wavefront

        Raises:
            NotImplementedError: Not implemented
        """
        super().propagate(wavefront, distance)
        raise NotImplementedError("Fraunhofer propagation not implemented")
