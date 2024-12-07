"""Classes for wavefront propagation."""

# Standard imports
import torch

# Local imports
from .base import Operator, TensorLike
from ..component import Wave
from ..utils.decorators import (
    as_tensor,
    requires_attrs
)

__all__ = ['Propagate']

class Propagate(Operator):
    """Fresnel propagation operator."""

    def __init__(self):
        self._kernel_cache = {}  # Cache for propagation kernels
    
    @requires_attrs('energy', 'spacing')
    def _get_kernel(self, wave: Wave, distance: torch.Tensor) -> torch.Tensor:
        """Get or compute propagation kernel."""
        # Simple key using essential parameters
        key = (wave.energy, wave.spacing, distance.float().mean())
        
        # Check cache
        if key not in self._kernel_cache:
            self._kernel_cache[key] = torch.exp(-1j * wave.wavelength * distance * wave.freq2)
        return self._kernel_cache[key]
        
    def _propagate(self, wave: Wave, distance: torch.Tensor) -> Wave:
        """Core propagation in Fourier domain."""
        # Get kernel
        kernel = self._get_kernel(wave, distance)
        
        # Propagate using torch FFT
        wave.form = torch.fft.ifft2(
            torch.fft.fft2(wave.form, dim=(-2, -1)) * kernel,
            dim=(-2, -1)
        )
        
        # Update position
        wave.position += distance.squeeze(-1).squeeze(-1)
        return wave

    @as_tensor('distance')
    def apply(self, wave: Wave, distance: TensorLike) -> Wave:
        """Forward Fresnel propagation."""
        distance = distance.to(dtype=torch.float64)[..., None, None]
        return self._propagate(wave, distance)

    @as_tensor('distance')
    def adjoint(self, wave: Wave, distance: TensorLike) -> Wave:
        """Adjoint Fresnel propagation (backward propagation)."""
        distance = distance.to(dtype=torch.float64)[..., None, None]
        return self._propagate(wave, -distance)
