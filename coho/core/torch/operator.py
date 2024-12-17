"""Wave field operators with custom gradient support.

Each operator provides two computation paths:
1. Custom gradients via PyTorch Function classes
2. Automatic gradients via PyTorch autograd

Memory is optimized by using in-place operations when gradients aren't needed.

Operators:
- Detect: Amplitude detection |ψ|
- Propagate: Fresnel propagation exp(-iλzk²)
- Modulate: Complex multiplication ψ₁ψ₂
"""

# Standard imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import Function as F
from typing import Optional, Tuple, Union

# Local imports
from .wave import Wave
from ..utils.decorators import (
    requires_matching,
)

__all__ = [
    'Detect', 
    'Propagate', 
    'Modulate',   
]


class Operator(nn.Module):
    """Base class for wave operators.
    
    Provides framework for operators with two gradient computation paths:
    1. Custom gradients using PyTorch Function
    2. Automatic gradients using PyTorch autograd
    """
    def __init__(self, use_custom_grads: bool = False) -> None:
        super().__init__()
        self.use_custom_grads = use_custom_grads
        
    def forward(self, *args, **kwargs) -> Union[Wave, Tensor]:
        if self.use_custom_grads:
            return self._forward_custom(*args, **kwargs)
        return self._forward_auto(*args, **kwargs)


class Detect(Operator):
    """Wave field amplitude detection.
    
    Computes |ψ| with optimized memory usage:
    - In-place operations when no gradients needed
    - Out-of-place operations for gradient tracking
    """
    
    @staticmethod
    def _forward_custom(wave: Wave) -> Tensor:
        return _DetectFunction.apply(wave.form)
        
    @staticmethod
    def _forward_auto(wave: Wave) -> Tensor:
        return torch.abs(wave.form) # No in-place operation for abs() 
    

class _DetectFunction(F):
    """Custom gradient implementation for amplitude detection."""
    
    @staticmethod
    def forward(ctx, wave_form: Tensor) -> Tensor:
        output = torch.abs(wave_form)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(wave_form)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        wave_form, = ctx.saved_tensors
        amplitude = wave_form.abs().clamp(min=1e-10)
        return grad_output * wave_form / amplitude


class Propagate(Operator):
    """Fresnel wave propagation.
    
    Computes ψ(z) = F⁻¹[F[ψ(0)]exp(-iλzk²)] with:
    - FFT-based implementation
    - Position tracking
    - Memory optimization
    """
    
    @staticmethod
    def _compute_kernel(wave: Wave, distance: Tensor) -> Tensor:
        """Compute Fresnel propagation kernel exp(-iλzk²)."""
        return torch.exp(-1j * wave.wavelength * distance[..., None, None] * wave.freq2)
    
    @staticmethod
    def _forward_custom(wave: Wave, distance: Tensor) -> Wave:
        output_form = _PropagateFunction.apply(
            wave.form, distance, wave.wavelength, wave.freq2
        )
        return wave.clone_with(form=output_form, position=wave.position + distance)
        
    @staticmethod
    def _forward_auto(wave: Wave, distance: Tensor) -> Wave:
        kernel = Propagate._compute_kernel(wave, distance)
        result = torch.fft.ifft2(torch.fft.fft2(wave.form, dim=(-2, -1)) * kernel, dim=(-2, -1))
        return wave.clone_with(form=result, position=wave.position + distance)

class _PropagateFunction(F):
    """Custom gradient implementation for Fresnel propagation."""
    
    _forward_cache = {}
    _backward_cache = {}
    
    @staticmethod
    def _compute_forward_kernel(wavelength: Tensor, distance: Tensor, freq2: Tensor) -> Tensor:
        """Compute forward propagation kernel."""
        return torch.exp(-1j * wavelength * distance * freq2)
    
    @staticmethod
    def _compute_backward_kernel(wavelength: Tensor, distance: Tensor, freq2: Tensor) -> Tensor:
        """Compute backward propagation kernel."""
        return torch.exp(1j * wavelength * distance * freq2)
    
    @staticmethod
    def forward(ctx, wave_form: Tensor, distance: Tensor, 
                wavelength: Tensor, freq2: Tensor) -> Tensor:
        expanded_distance = distance[..., None, None]
        kernel = torch.exp(-1j * wavelength * expanded_distance * freq2)
        fft_form = torch.fft.fft2(wave_form, dim=(-2, -1)) * kernel
        output = torch.fft.ifft2(fft_form, dim=(-2, -1))
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(wave_form, expanded_distance, wavelength, freq2)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        wave_form, expanded_distance, wavelength, freq2 = ctx.saved_tensors
        return wave_form - grad_output, None, None, None


class Modulate(Operator):
    """Wave field modulation.
    
    Computes ψ₁ψ₂ with property matching and memory optimization.
    Requires matching energy, spacing, and position between waves.
    """
    
    @staticmethod
    @requires_matching('energy', 'spacing', 'position')
    def _forward_custom(wave1: Wave, wave2: Wave) -> Wave:
        output_form = _ModulateFunction.apply(wave1.form, wave2.form)
        return wave1.clone_with(form=output_form)

    @staticmethod
    @requires_matching('energy', 'spacing', 'position')
    def _forward_auto(wave1: Wave, wave2: Wave) -> Wave:
        return wave1.clone_with(form=wave1.form * wave2.form)
    

class _ModulateFunction(F):
    """Custom gradient implementation for wave modulation."""
    
    @staticmethod
    def forward(ctx, wave1_form: Tensor, wave2_form: Tensor) -> Tensor:
        output = wave1_form * wave2_form
        saved_tensors = []
        if ctx.needs_input_grad[0]:
            saved_tensors.append(wave1_form)
        if ctx.needs_input_grad[1]:
            saved_tensors.append(wave2_form)
        if saved_tensors:
            ctx.save_for_backward(*saved_tensors)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if not hasattr(ctx, 'needs_grad1'):  # No gradients needed
            return None, None
        wave1_form, wave2_form = ctx.saved_tensors
        return (
            grad_output * wave2_form.conj() if ctx.needs_input_grad[0] else None,
            grad_output * wave1_form.conj() if ctx.needs_input_grad[1] else None
        )
    