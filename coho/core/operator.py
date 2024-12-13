"""Operator module."""

# Standard imports
from abc import abstractmethod
import torch
import torch.nn as nn
from torch import Tensor


# Local imports
from .wave import Wave
from .utils.decorators import (
    requires_unstacked,
    requires_matching,
)

__all__ = [
    'Propagate', 
    'Modulate', 
    'Detect',   
    'Vectorize'
]


class Operator(nn.Module):
    """Base class for wave operators."""
    def __init__(self, use_custom_grads=False):
        super().__init__()
        self.use_custom_grads = use_custom_grads

    def forward(self, *args, **kwargs):
        """Forward pass dispatching to appropriate gradient implementation."""
        if self.use_custom_grads:
            return self._with_usergrad(*args)
        else:
            return self._with_autograd(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def _with_usergrad(self):
        """User-defined gradient implementation."""
        pass

    @staticmethod
    @abstractmethod
    def _with_autograd(*args, **kwargs):
        """Automatic differentiation implementation."""
        pass


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
    def _with_usergrad():
        """Custom gradient using analytical Fresnel propagator gradient."""
        return _PropagateFunction

    @staticmethod
    def _with_autograd(wave: Wave, distance: Tensor) -> Wave:
        """Automatic differentiation path."""
        return Propagate._forward(wave, distance)

    # Main interface
    def forward(self, wave: Wave, distance: Tensor) -> Wave:
        """Propagate wave by distance.
        
        Args:
            wave: Input wave
            distance: Propagation distance
        Returns:
            Wave: Propagated wave
        """
        return super().forward(wave, distance)


# Custom gradient implementation
class _PropagateFunction(torch.autograd.Function):
    """Custom gradient implementation for Fresnel propagation.
    
    Forward: Regular Fresnel propagation
    Backward: Conjugate Fresnel propagation (negative distance)
    """
    
    @staticmethod
    def forward(ctx, wave: Wave, distance: Tensor):
        ctx.save_for_backward(distance)
        return Propagate._forward(wave, distance)

    @staticmethod
    def backward(ctx, grad_output):
        distance, = ctx.saved_tensors
        return Propagate._forward(grad_output, -distance)


class Modulate(Operator):
    """Modulate (multiply) two waves.
    
    Supports both automatic differentiation and custom gradients.
    Custom gradient implementation uses product rule for complex multiplication.
    """
    
    # Core computation shared by both gradient paths
    @staticmethod
    def _forward(wave1: Wave, wave2: Wave) -> Wave:
        """Complex multiplication of two waves.
        
        Args:
            wave1: First wave (modified in-place)
            wave2: Second wave
        Returns:
            Wave: Modulated wave (wave1 * wave2)
        """
        wave1 *= wave2
        return wave1

    # Gradient implementations
    @staticmethod
    def _with_usergrad():
        """Custom gradient using product rule."""
        return _ModulateFunction

    @staticmethod
    @requires_matching('energy', 'spacing', 'position')
    def _with_autograd(wave1: Wave, wave2: Wave) -> Wave:
        """Automatic differentiation path."""
        return Modulate._forward(wave1, wave2)

    # Main interface
    @requires_matching('energy', 'spacing', 'position')
    def forward(self, wave1: Wave, wave2: Wave) -> Wave:
        """Modulate two waves.
        
        Args:
            wave1: First wave (modified in-place)
            wave2: Second wave
        Returns:
            Wave: Modulated wave (wave1 * wave2)
        """
        return super().forward(wave1, wave2)


# Custom gradient implementation
class _ModulateFunction(torch.autograd.Function):
    """Custom gradient implementation for wave modulation.
    
    Forward: Complex multiplication
    Backward: Product rule - d(w1*w2) = dw1*w2 + w1*dw2
    """
    
    @staticmethod
    def forward(ctx, wave1: Wave, wave2: Wave):
        ctx.save_for_backward(wave1.form, wave2.form)
        return Modulate._forward(wave1, wave2)

    @staticmethod
    def backward(ctx, grad_output):
        wave1_form, wave2_form = ctx.saved_tensors
        grad1 = grad_output
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
        """Compute wave amplitude.
        
        Args:
            wave: Input wave
        Returns:
            Tensor: Wave amplitude |ψ|
        """
        return wave.amplitude

    # Gradient implementations
    @staticmethod
    def _with_usergrad():
        """Custom gradient using d|ψ|/dψ = ψ/|ψ|."""
        return _DetectFunction

    @staticmethod
    def _with_autograd(wave: Wave) -> Tensor:
        """Automatic differentiation path."""
        return Detect._forward(wave)

    # Main interface
    def forward(self, wave: Wave) -> Tensor:
        """Detect wave amplitude.
        
        Args:
            wave: Input wave
        Returns:
            Tensor: Wave amplitude |ψ|
        """
        return super().forward(wave)


# Custom gradient implementation
class _DetectFunction(torch.autograd.Function):
    """Custom gradient implementation for amplitude detection.
    
    Forward: Compute amplitude |ψ|
    Backward: Complex derivative d|ψ|/dψ = ψ/|ψ|
    """
    
    @staticmethod
    def forward(ctx, wave: Wave):
        ctx.save_for_backward(wave.form)
        ctx.wave = wave
        return Detect._forward(wave)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        wave_form, = ctx.saved_tensors
        amplitude = wave_form.abs().clamp(min=1e-10)
        grad_wave = ctx.wave.copy()
        grad_wave.form = grad_output * wave_form / amplitude
        return grad_wave
    

@requires_unstacked
def vectorize(wave: Wave, size: int) -> Wave:
    """Vectorize wave computation by expanding along first dimension.
    
    This function enables parallel processing of identical waves:
    Instead of sequential processing of n identical waves,
    we expand to n copies for one parallel computation.
    
    Example:
        wave = Wave(...)
        vec = vectorize(wave, size=len(distances))  # [n, H, W]
        results = propagate(vec, distances)  # One vectorized computation
    
    Args:
        wave: Input wave [1, H, W]
        size: Number of copies for parallel processing
    Returns:
        Wave: Expanded wave [size, H, W]
    """
    wave.form = wave.form.expand(size, *wave.form.shape[-2:])
    return wave
