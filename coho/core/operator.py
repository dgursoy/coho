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
from typing import Union, Type, Tuple, Optional

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
    
    Provides common infrastructure for wave operations with:
    - Optional custom gradient computation
    - Memory optimization using in-place ops when possible
    - Automatic gradient preservation when needed
    
    Args:
        use_custom_grads: Whether to use custom gradient computation
    """
    def __init__(self, use_custom_grads: bool = False) -> None:
        super().__init__()
        self.use_custom_grads = use_custom_grads

    def forward(self, *args, **kwargs) -> Union[Wave, Tensor]:
        """Forward pass with smart gradient handling.
        
        Uses:
        - Custom gradients if specified
        - In-place operations when no gradients needed
        - Out-of-place operations for gradient tracking
        """
        if self.use_custom_grads:
            return self._with_custom_grads().apply(*args)
        return self._forward(*args, **kwargs)


# Wave Operators
# Each operator follows the pattern:
# 1. Main operator class with common forward computation
# 2. Custom gradient function class for analytical gradients

class Propagate(Operator):
    """Fresnel propagation of waves.
    
    Memory optimization:
    - Uses in-place operations when no gradients needed
    - Creates new tensors when gradient tracking required
    - Preserves wave properties in both cases
    """
    
    # Core computation shared by both gradient paths
    @staticmethod
    def _forward(wave: Wave, distance: Tensor) -> Wave:
        """Fresnel propagation computation.
        
        Args:
            wave: Input wave
            distance: Propagation distance in meters
            
        Returns:
            Wave: Propagated wave, either in-place or new based on requires_grad
        """
        distance = distance[..., None, None]
        kernel = torch.exp(-1j * wave.wavelength * distance * wave.freq2)
        fft_form = torch.fft.fft2(wave.form, dim=(-2, -1)) * kernel
        
        if wave.requires_grad:
            # Out-of-place for gradient tracking
            form = torch.fft.ifft2(fft_form, dim=(-2, -1))
            new_wave = wave._like_me(form)
        else:
            # In-place when no gradients needed
            wave.form = torch.fft.ifft2(fft_form, dim=(-2, -1))
            new_wave = wave
            
        new_wave.position = wave.position + distance.squeeze(-1).squeeze(-1)
        return new_wave

    # Gradient implementations
    @staticmethod
    def _with_custom_grads() -> Type[Function]:
        """Returns custom gradient function class."""
        return _PropagateFunction


# Custom gradient implementation
class _PropagateFunction(Function):
    """Custom gradient implementation for Fresnel propagation.
    
    Forward pass caches tensors needed for backward pass while using
    the core Propagate._forward implementation for computation.
    
    Forward: Regular Fresnel propagation (using Propagate._forward)
    Backward: Conjugate Fresnel propagation (negative distance)
    """
    
    @staticmethod
    def forward(ctx: FunctionCtx, wave: Wave, distance: Tensor) -> Wave:
        """Forward pass that caches for backward while using core implementation."""
        ctx.save_for_backward(wave.form, distance)
        return Propagate._forward(wave, distance)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Wave) -> Tuple[Wave, Optional[Tensor]]:
        wave_form, distance = ctx.saved_tensors
        if wave_form.requires_grad:
            grad_wave = grad_output._like_me(wave_form)
            return Propagate._forward(grad_wave, -distance), None
        return None, None


class Modulate(Operator):
    """Complex multiplication of waves.
    
    Memory optimization:
    - Leverages Wave class arithmetic operations
    - Automatically handles gradient requirements
    """
    
    # Core computation shared by both gradient paths
    @staticmethod
    @requires_matching('energy', 'spacing', 'position')
    def _forward(wave1: Wave, wave2: Wave) -> Wave:
        """Complex multiplication of two waves.
        
        Args:
            wave1: First wave
            wave2: Second wave
            
        Returns:
            Wave: Modulated wave (wave1 * wave2)
        """
        if not (wave1.requires_grad or wave2.requires_grad):
            # In-place when no gradients needed
            wave1 *= wave2
            return wave1
        # Out-of-place for gradient tracking
        return wave1 * wave2

    # Gradient implementations
    @staticmethod
    def _with_custom_grads() -> Type[Function]:
        """Returns custom gradient function class."""
        return _ModulateFunction


# Custom gradient implementation
class _ModulateFunction(Function):
    """Custom gradient implementation for wave modulation.
    
    Forward pass caches wave forms needed for backward pass while using
    the core Modulate._forward implementation for computation.
    
    Forward: Complex multiplication (using Modulate._forward)
    Backward: Product rule - d(w1*w2) = dw1*w2 + w1*dw2
    """
    
    @staticmethod
    def forward(ctx: FunctionCtx, wave1: Wave, wave2: Wave) -> Wave:
        """Forward pass that caches for backward while using core implementation."""
        ctx.save_for_backward(wave1.form, wave2.form)
        return Modulate._forward(wave1, wave2)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Wave) -> Tuple[Wave, Wave]:
        wave1_form, wave2_form = ctx.saved_tensors
        
        grad1 = grad2 = None
        if wave1_form.requires_grad:
            grad1 = grad_output.copy()
            grad1.form *= wave2_form
        if wave2_form.requires_grad:
            grad2 = grad_output.copy()
            grad2.form *= wave1_form
            
        return grad1, grad2


class Detect(Operator):
    """Amplitude detection of waves.
    
    Memory optimization:
    - Uses in-place operations for amplitude when possible
    - Creates new tensors when gradient tracking required
    """
    
    # Core computation shared by both gradient paths
    @staticmethod
    def _forward(wave: Wave) -> Tensor:
        """Compute wave amplitude.
        
        Args:
            wave: Input wave
            
        Returns:
            Tensor: Wave amplitude
        """
        if wave.requires_grad:
            # Out-of-place for gradient tracking
            return wave.amplitude.clone()
        # In-place when no gradients needed
        return torch.abs(wave.form)

    # Gradient implementations
    @staticmethod
    def _with_custom_grads() -> Type[Function]:
        """Returns custom gradient function class."""
        return _DetectFunction


# Custom gradient implementation
class _DetectFunction(Function):
    """Custom gradient implementation for amplitude detection.
    
    Forward pass caches wave form needed for backward pass while using
    the core Detect._forward implementation for computation.
    
    Forward: Compute amplitude |ψ| (using Detect._forward)
    Backward: Complex derivative d|ψ|/dψ = ψ/|ψ|
    """
    
    @staticmethod
    def forward(ctx: FunctionCtx, wave: Wave) -> Tensor:
        """Forward pass that caches for backward while using core implementation."""
        ctx.save_for_backward(wave.form)
        ctx.wave = wave
        return Detect._forward(wave)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Wave) -> Wave:
        wave_form, = ctx.saved_tensors

        if wave_form.requires_grad:
            amplitude = wave_form.abs().clamp(min=1e-10)
            grad_form = grad_output * wave_form / amplitude
            return Wave(grad_form)  # Return Wave with gradient
        return None
