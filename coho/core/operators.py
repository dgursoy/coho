"""Wave field operators using JAX."""

import jax
import jax.numpy as jnp
from .wave import Wave


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
        spacing=wave.spacing
    )


@jax.jit
def modulate(wave1: Wave, wave2: Wave) -> Wave:
    """Wave field modulation ψ₁ψ₂."""
    return Wave(
        form=wave1.form * wave2.form,
        energy=wave1.energy,
        spacing=wave1.spacing
    )
