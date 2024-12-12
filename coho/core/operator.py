"""Operator module."""

# Standard imports
from abc import abstractmethod
from typing import Union, TypeAlias, Callable, Any, List,  Dict
import torch

# Local imports
from .wave import Wave
from .utils.decorators import (
    requires_unstacked,
    requires_matching,
    requires_attrs,
    requires_cached_tensors,
    as_tensor,
)

__all__ = [
    'Propagate', 
    'Modulate', 
    'Detect',   
    'Move', 
    'Shift', 
    'Crop', 
    'Stack'
]

# Type aliases
TensorLike: TypeAlias = Union[float, List[float], torch.Tensor]
TensorDict: TypeAlias = Dict[str, TensorLike]

class Operator(torch.nn.Module):
    """Base class for operators.
    
    Provides:
    1. Forward operation through forward()
    2. Gradient computation through gradient()
    3. Caching mechanism for expensive computations
    """
    
    def __init__(self):
        super().__init__()
        self._cache = {}  # Cache for expensive computations

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward operator application."""
        pass

    @abstractmethod
    def gradient(self, grad_output: Any, *args, **kwargs) -> Any:
        """Gradient computation."""
        pass

    # Cache management methods
    def _get_or_compute(self, cache_type: str, key: tuple, function: Callable) -> Any:
        """Get from cache or compute and cache value."""
        cached = self._get_cached(cache_type, key)
        if cached is None:
            value = function()
            self._set_cached(cache_type, key, value)
            return value
        return cached

    def _get_cached(self, cache_type: str, key: tuple) -> Any:
        """Get cached value if it exists, None otherwise."""
        if cache_type not in self._cache:
            self._cache[cache_type] = {}
        return self._cache[cache_type].get(key)
    
    def _set_cached(self, cache_type: str, key: tuple, value: Any) -> None:
        """Set value in cache."""
        if cache_type not in self._cache:
            self._cache[cache_type] = {}
        self._cache[cache_type][key] = value

    def clear_cache(self, cache_type: str = None) -> None:
        """Clear specified cache or all caches."""
        if cache_type in self._cache:
            self._cache[cache_type] = {}
        else:
            self._cache = {}
    
    def get_cache_info(self) -> dict:
        """Get information about cache usage."""
        return {k: len(v) for k, v in self._cache.items()}

class Propagate(Operator):
    """Fresnel propagation operator."""

    def _propagate(self, wave: Wave, distance: torch.Tensor) -> Wave:
        """Core propagation in Fourier domain."""
        # Get kernel using base class caching
        kernel = self._get_or_compute(
            cache_type='kernel',
            key=(wave.energy, wave.spacing, distance.float().mean()),
            function=lambda: torch.exp(-1j * wave.wavelength * distance * wave.freq2)
        )
        
        # Propagate using torch FFT
        wave.form = torch.fft.ifft2(
            torch.fft.fft2(wave.form, dim=(-2, -1)) * kernel,
            dim=(-2, -1)
        )
        
        # Update position
        wave.position = wave.position + distance.squeeze(-1).squeeze(-1)
        
        return wave

    @as_tensor('distance')
    def forward(self, wave: Wave, distance: TensorLike) -> Wave:
        """Forward Fresnel propagation."""
        distance = distance.to(dtype=torch.float64)[..., None, None]
        return self._propagate(wave, distance)

    @as_tensor('distance')
    def gradient(self, grad_output: Wave, distance: TensorLike) -> Wave:
        """Compute gradient of Fresnel propagation."""
        distance = distance.to(dtype=torch.float64)[..., None, None]
        return self._propagate(grad_output, -distance)

    def _get_caller(self):
        """Helper to get caller info."""
        import inspect
        frame = inspect.currentframe()
        caller = frame.f_back.f_back
        return f"{caller.f_code.co_name} in {caller.f_code.co_filename}:{caller.f_lineno}"

class Modulate(Operator):
    """Modulate wavefront by another wavefront."""

    @requires_matching('energy', 'spacing', 'position')
    def forward(self, wave1: Wave, wave2: Wave) -> Wave:
        """Forward modulation."""
        return wave1 * wave2

    @requires_cached_tensors  # Needs waves from forward pass
    @requires_matching('energy', 'spacing', 'position')
    def gradient(self, grad_output: Wave, wave1: Wave, wave2: Wave) -> tuple[Wave, Wave]:
        """Gradient of modulation."""
        return grad_output / wave2, grad_output / wave1

class Detect(Operator):
    """Detect wavefront amplitude."""

    def forward(self, wave: Wave) -> torch.Tensor:
        """Wavefront to amplitude."""
        return wave.amplitude  # |ψ|
    
    def gradient(self, grad_output: torch.Tensor, wave: Wave) -> Wave:
        """Gradient of amplitude with respect to wave.
        
        For complex wave ψ, d|ψ|/dψ = ψ/|ψ|
        """
        amplitude = wave.amplitude.clamp(min=1e-10)  
        grad_wave = wave.copy()
        grad_wave.form = grad_output * wave.form / amplitude
        return grad_wave

# Plotting
import matplotlib.pyplot as plt
def plot_wave_stack(wave, title="Wave", figsize=(20, 8)):
    """Plot amplitude and phase for each wave in a stack.
    
    Args:
        wave: Wave object with stack dimension
        title: Base title for the plot
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    for m in range(wave.form.shape[0]):
        # Amplitude
        plt.subplot(2, 4, m+1)
        plt.title(f'{title[m]} {m+1} Amplitude')
        plt.imshow(wave.amplitude[m], cmap='gray')
        plt.colorbar()
        
        # Phase
        plt.subplot(2, 4, m+5)
        plt.title(f'{title[m]} {m+1} Phase')
        plt.imshow(wave.phase[m], cmap='gray')
        plt.colorbar()
    plt.tight_layout()
    plt.show()

class Move(Operator):
    """Move wavefront to new position.
    
    This is a bookkeeping operator that only updates wave position.
    It doesn't transform the wave or affect gradients directly.
    Downstream operators (like Propagate) will use this position
    information in their computations and gradient calculations.
    """
    
    @as_tensor('position')
    @requires_attrs('position')
    def forward(self, wave: Wave, position: TensorLike) -> Wave:
        """Move wave to new position."""
        wave.position = wave.position + position
        return wave
    
    @as_tensor('position')
    @requires_attrs('position')
    def gradient(self, grad_output: Wave, position: TensorLike) -> Wave:
        """Gradient of move operation."""
        grad_output.position = grad_output.position - position
        return grad_output

class Shift(Operator):
    """Shift operator.
    
    Unlike Move which only updates position metadata,
    Shift actually rolls/shifts the wave form data.
    """

    @as_tensor('y', 'x')
    def forward(self, wave: Wave, y: TensorLike, x: TensorLike) -> Wave:
        """Apply shifts to wave."""
        # Shift each form in batch
        shifted_forms = []
        for form, y, x in zip(wave.form, y, x):
            shifted = torch.roll(torch.roll(form, int(y), dims=-2), int(x), dims=-1)
            shifted_forms.append(shifted)
        wave.form = torch.stack(shifted_forms)
        return wave

    @as_tensor('y', 'x')
    def gradient(self, grad_output: Wave, y: TensorLike, x: TensorLike) -> Wave:
        """Gradient of shift operation.
        
        For a circular shift operation, the gradient is just
        the opposite shift of the upstream gradient.
        """
        return self.forward(grad_output, -y, -x)

class Crop(Operator):
    """Crop wavefront to match dimensions of another wave."""

    def forward(self, wave_original: Wave, wave: Wave) -> Wave:
        """Crop wave to match dimensions of another wave."""
        return wave_original.crop_to_match(wave, pad_value=1.0)

    @requires_cached_tensors  # Needs wave from forward pass
    def gradient(self, grad_output: Wave, wave_original: Wave) -> Wave:
        """Gradient of crop operation.
        
        The gradient needs to be resized back to the original 
        wave dimensions. Padding areas (where pad_value=1.0 was used)
        get zero gradient.
        """
        return grad_output.crop_to_match(wave_original, pad_value=1.0) 
    
class Stack(Operator):
    """Stacks a wave along stack (first) dimension in-place."""
    
    @requires_unstacked
    def forward(self, wave: Wave, stack_size: int) -> Wave:
        """Stack n copies of wave along stack dimension in-place."""
        wave.form = wave.form.expand(stack_size, *wave.form.shape[-2:])
        return wave
    
    def gradient(self, grad_output: Wave, stack_size: int) -> Wave:
        """Gradient of stack operation.
        
        Since forward duplicates the wave n times,
        gradient needs to average gradients from all copies.
        """
        grad_output.form = grad_output.form.mean(dim=0, keepdim=True)
        return grad_output


