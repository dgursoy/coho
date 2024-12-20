"""Wave field operators using JAX."""

import jax
import jax.numpy as jnp
from .wave import Wave
from functools import partial
from typing import Tuple
from dataclasses import dataclass


@jax.jit
def detect(wave: Wave) -> jnp.ndarray:
    """Wave field amplitude detection |ψ|²."""
    return jnp.abs(wave.form)


@jax.jit
def propagate(wave: Wave, distance: float) -> Wave:
    """Fresnel propagation for a single distance (no batching)."""
    # Squared spatial frequencies
    *_, ny, nx = wave.shape
    fx = jnp.fft.fftfreq(nx, wave.spacing)
    fy = jnp.fft.fftfreq(ny, wave.spacing)
    fx2, fy2 = jnp.meshgrid(fx**2, fy**2, indexing='ij')
    freq2 = fx2 + fy2
    
    # Transfer function (distance is now a scalar)
    kernel = jnp.exp(-1j * wave.wavelength * distance * freq2)
    
    # Apply propagation
    ft = jnp.fft.fft2(wave.form)
    propagated = jnp.fft.ifft2(ft * kernel)
    
    return Wave(
        form=propagated,
        energy=wave.energy,
        spacing=wave.spacing,
        mode=wave.mode
    )


@jax.jit
def modulate(wave1: Wave, wave2: Wave) -> Wave:
    """Wave field modulation ψ₁ψ₂."""
    return Wave(
        form=wave1.form * wave2.form,
        energy=wave1.energy,
        spacing=wave1.spacing,
        position=wave1.position,
        mode=wave1.mode
    )


@dataclass
class ResizeDimension:
    """Parameters for resizing a single dimension.
    
    This class encapsulates all the parameters needed to resize one dimension of an array,
    handling both upscaling and downscaling cases.
    
    Attributes:
        source_size: Original size of the dimension
        target_size: Desired size after resizing
        start_idx: Starting index for slicing/padding operation
        is_upscaling: True if target_size >= source_size, False otherwise
        
    Example:
        >>> dim = ResizeDimension(source_size=60, target_size=101, start_idx=20, is_upscaling=True)
        >>> # This indicates we're padding a 60-pixel dimension to 101 pixels,
        >>> # starting at index 20 to center the content
    """
    source_size: int
    target_size: int
    start_idx: int
    is_upscaling: bool


@partial(jax.jit, static_argnames=['target_size'])
def scale(wave: Wave, *, target_size: Tuple[int, int]) -> Wave:
    """Scale the wave field based on propagation distance using Fourier-domain resampling.
    
    Args:
        wave: Input wave field to be scaled
        target_size: Desired output dimensions (height, width)
        
    Returns:
        Scaled wave field with dimensions matching target_size
    """
    # Extract dimensions and compute scale factors
    *_, src_height, src_width = wave.shape
    src_size = (src_height, src_width)
    scale_factors = jnp.array([t / s for t, s in zip(target_size, src_size)])

    # Transform to Fourier domain
    ft = jnp.fft.fftshift(jnp.fft.fft2(wave.form))
    
    def analyze_dimension(src_size: int, tgt_size: int) -> ResizeDimension:
        """Analyze how to resize a dimension.
        
        Args:
            src_size: Source dimension size
            tgt_size: Target dimension size
            
        Returns:
            ResizeDimension object with computed parameters
        """
        is_up = tgt_size >= src_size
        size_a = src_size if is_up else tgt_size
        size_b = tgt_size if is_up else src_size
        
        # Compute correction for odd/even dimensions
        correction = size_a % 2
        if size_a % 2 == 1 and size_b % 2 == 1:
            correction -= 1
            
        start = (size_b - size_a) // 2 + correction
        return ResizeDimension(src_size, tgt_size, start, is_up)
    
    def resize_dimension(x: jnp.ndarray, dim: ResizeDimension, axis: int) -> jnp.ndarray:
        """Resize array along specified dimension.
        
        Args:
            x: Input array
            dim: ResizeDimension parameters
            axis: Axis to resize
            
        Returns:
            Resized array along specified dimension
        """
        start_indices = [0] * x.ndim
        start_indices[axis] = dim.start_idx
        
        if dim.is_upscaling:
            shape = list(x.shape)
            shape[axis] = dim.target_size
            padded = jnp.zeros(shape, dtype=x.dtype)
            return jax.lax.dynamic_update_slice(padded, x, start_indices)
        
        slice_sizes = list(x.shape)
        slice_sizes[axis] = dim.target_size
        return jax.lax.dynamic_slice(x, start_indices, slice_sizes)
    
    # Analyze dimensions
    dims = [
        analyze_dimension(src_height, target_size[0]),  # Height
        analyze_dimension(src_width, target_size[1]),   # Width
    ]
    
    # Apply resizing operations sequentially
    resized = ft
    for axis, dim in enumerate(dims):
        resized = resize_dimension(resized, dim, axis)
    
    # Compute phase corrections in frequency domain
    freqs = [jnp.fft.fftfreq(size) for size in target_size]
    fx, fy = jnp.meshgrid(*freqs, indexing='ij')
    phase_shifts = (scale_factors - 1) / 2
    phase_correction = jnp.exp(-2j * jnp.pi * (
        phase_shifts[0] * fx + phase_shifts[1] * fy
    ))
    
    # Transform back to spatial domain
    ft_scaled = jnp.fft.ifft2(jnp.fft.ifftshift(resized) * phase_correction)

    return Wave(
        form=ft_scaled, 
        spacing=wave.spacing, 
        position=wave.position, 
        mode=wave.mode, 
    )


@jax.jit
def shift(wave: Wave, offset: Tuple[float, float]) -> Wave:
    """Shift the wave field using Fourier-domain phase shifting.
    
    Args:
        wave: Input wave field to be shifted
        offset: Desired shift in (y, x) coordinates, in pixels.
               Positive values shift the wave in the positive direction.
        
    Returns:
        Shifted wave field
    """
    # Transform to Fourier domain
    ft = jnp.fft.fft2(wave.form)
    
    # Create frequency grids
    freqs = [jnp.fft.fftfreq(size) for size in wave.shape]
    fx, fy = jnp.meshgrid(*freqs, indexing='ij')
    
    # Compute phase shift
    # The signs match the standard image coordinate system:
    # positive x shifts right, positive y shifts down
    phase = 2j * jnp.pi * (offset[0] * fx - offset[1] * fy)
    phase_shift = jnp.exp(phase)
    
    # Apply shift and transform back
    shifted = jnp.fft.ifft2(ft * phase_shift)
    
    return Wave(
        form=shifted,
        spacing=wave.spacing,
        position=wave.position,
        mode=wave.mode
    )
