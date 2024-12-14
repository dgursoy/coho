"""Operator module for wave manipulation.

This module provides operators for common wave operations:
- Propagate: Fresnel propagation of waves
- Modulate: Complex multiplication of waves
- Detect: Amplitude detection of waves

Each operator supports both automatic differentiation and custom gradients
through PyTorch's autograd system.
"""

# Standard imports
import torch
from torch.nn import Module
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from typing import Union, Type, Any, Tuple, Optional

# Local imports
from .wave import Wave
from .utils.decorators import (
    requires_matching,
)

__all__ = [
    'Propagate', 
    'Modulate', 
    'Detect',   
]

# Base class for all operators

class Operator(Module):
    """Base class for wave operators.
    
    Provides common infrastructure for wave operations with optional
    custom gradient computation.
    
    Args:
        use_custom_grads: Whether to use custom gradient computation
    """
    def __init__(self, use_custom_grads: bool = False) -> None:
        super().__init__()
        self.use_custom_grads = use_custom_grads

    def forward(self, *args, **kwargs) -> Union[Wave, Tensor]:
        """Forward pass with optional gradient tracking.
        
        Uses either automatic differentiation or custom gradients
        based on use_custom_grads flag.
        """
        if self.use_custom_grads:
            return self._with_custom_grads().apply(*args)
        return self._forward(*args, **kwargs)


# Wave Operators
# Each operator follows the pattern:
# 1. Main operator class with common forward computation
# 2. Custom gradient function class for analytical gradients

class Propagate(Operator):
    """Propagate wave by distance using Fresnel propagator.
    
    Supports both automatic differentiation and custom gradients.
    Custom gradient implementation uses analytical gradient of Fresnel propagator.
    """
    
    # Core computation shared by both gradient paths
    @staticmethod
    def _forward(wave: Wave, distance: Tensor) -> Wave:
        """Fresnel propagation computation.
        
        Args:
            wave: Input wave
            distance: Propagation distance
        Returns:
            Wave: Propagated wave
        """
        distance = distance[..., None, None]
        kernel = torch.exp(-1j * wave.wavelength * distance * wave.freq2)
        wave.form = torch.fft.ifft2(
            torch.fft.fft2(wave.form, dim=(-2, -1)) * kernel,
            dim=(-2, -1)
        )
        wave.position = wave.position + distance.squeeze(-1).squeeze(-1)
        return wave

    # Gradient implementations
    @staticmethod
    def _with_custom_grads() -> Type[Function]:
        """Returns custom gradient function class."""
        return _PropagateFunction


# Custom gradient implementation
class _PropagateFunction(Function):
    """Custom gradient implementation for Fresnel propagation.
    
    Forward: Regular Fresnel propagation
    Backward: Conjugate Fresnel propagation (negative distance)
    """
    
    @staticmethod
    def forward(ctx: FunctionCtx, wave: Wave, distance: Tensor) -> Wave:
        ctx.save_for_backward(distance)
        return Propagate._forward(wave, distance)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Wave) -> Tuple[Wave, Optional[Tensor]]:
        distance, = ctx.saved_tensors
        return Propagate._forward(grad_output, -distance), None


class Modulate(Operator):
    """Modulate (multiply) two waves.
    
    Supports both automatic differentiation and custom gradients.
    Custom gradient implementation uses product rule for complex multiplication.
    """
    
    # Core computation shared by both gradient paths
    @staticmethod
    @requires_matching('energy', 'spacing', 'position')
    def _forward(wave1: Wave, wave2: Wave) -> Wave:
        """Complex multiplication of two waves.
        
        Args:
            wave1: First wave (modified in-place)
            wave2: Second wave
        Returns:
            Wave: Modulated wave (wave1 * wave2)
        """
        return wave1 * wave2

    # Gradient implementations
    @staticmethod
    def _with_custom_grads() -> Type[Function]:
        """Returns custom gradient function class."""
        return _ModulateFunction


# Custom gradient implementation
class _ModulateFunction(Function):
    """Custom gradient implementation for wave modulation.
    
    Forward: Complex multiplication
    Backward: Product rule - d(w1*w2) = dw1*w2 + w1*dw2
    """
    
    @staticmethod
    def forward(ctx: FunctionCtx, wave1: Wave, wave2: Wave) -> Wave:
        ctx.save_for_backward(wave1.form, wave2.form)
        return Modulate._forward(wave1, wave2)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Wave) -> Tuple[Wave, Wave]:
        wave1_form, wave2_form = ctx.saved_tensors
        grad1 = grad_output.copy()
        grad2 = grad_output.copy()
        grad1.form *= wave2_form
        grad2.form *= wave1_form
        return grad1, grad2


class Detect(Operator):
    """Detect wave amplitude.
    
    Supports both automatic differentiation and custom gradients.
    Custom gradient implementation uses complex derivative of absolute value.
    """
    
    # Core computation shared by both gradient paths
    @staticmethod
    def _forward(wave: Wave) -> Tensor:
        """Common forward computation."""
        return wave.amplitude

    # Gradient implementations
    @staticmethod
    def _with_custom_grads() -> Type[Function]:
        """Returns custom gradient function class."""
        return _DetectFunction


# Custom gradient implementation
class _DetectFunction(Function):
    """Custom gradient implementation for amplitude detection.
    
    Forward: Compute amplitude |ψ|
    Backward: Complex derivative d|ψ|/dψ = ψ/|ψ|
    """
    
    @staticmethod
    def forward(ctx: FunctionCtx, wave: Wave) -> Tensor:
        ctx.save_for_backward(wave.form)
        ctx.wave = wave
        return Detect._forward(wave)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Wave) -> Wave:
        wave_form, = ctx.saved_tensors
        amplitude = wave_form.abs().clamp(min=1e-10)
        grad_wave = ctx.wave.copy()
        grad_wave.form = grad_output * wave_form / amplitude
        return grad_wave
