"""Classes for wavefront propagation."""

# Standard imports
import numpy as np
from typing import Union, List

# Local imports
from .base import Operator
from ..component import Wave
from .decorators import (
    validate_spacing,
    validate_energy,
    validate_position,
    validate_form
)

__all__ = ['Propagate']

class Propagate(Operator):
    """Fresnel propagation operator."""

    def __init__(self):
        self._kernel_cache = {}  # Cache for propagation kernels

    @validate_energy
    @validate_spacing
    def _get_kernel(self, wave: Wave, distance: np.ndarray) -> np.ndarray:
        """Get or compute propagation kernel."""
        # Simple key using essential parameters
        key = (wave.energy, wave.spacing, float(distance.mean()))

        # Check cache
        if key not in self._kernel_cache:
            self._kernel_cache[key] = np.exp(-1j * wave.wavelength * distance * wave.freq2)
        return self._kernel_cache[key]

    @validate_form
    @validate_position
    def _propagate(self, wave: Wave, distance: np.ndarray) -> Wave:
        """Core propagation in Fourier domain."""
        # Get kernel
        kernel = self._get_kernel(wave, distance)

        # Propagate
        wave.form = np.fft.ifft2(
            np.fft.fft2(wave.form, axes=(-2, -1)) * kernel,
            axes=(-2, -1)
        )

        # Update position
        # print(wave.position)
        wave.position += distance.ravel()
        # print(wave.position)
        return wave

    def apply(self, wave: Wave, distance: Union[float, List[float], np.ndarray]) -> Wave:
        """Forward Fresnel propagation."""
        distance = np.asarray(distance, dtype=float)[..., None, None]
        return self._propagate(wave, distance)

    def adjoint(self, wave: Wave, distance: Union[float, List[float], np.ndarray]) -> Wave:
        """Adjoint Fresnel propagation (backward propagation)."""
        distance = np.asarray(distance, dtype=float)[..., None, None]
        return self._propagate(wave, -distance)

