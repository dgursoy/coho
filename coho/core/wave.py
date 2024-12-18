"""Wave module using JAX."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, ClassVar


@jax.tree_util.register_pytree_node_class
@dataclass
class Wave:
    """Wave field with properties.
    
    Attributes:
        form: Complex wave field (ny, nx)
        energy: Photon energy in keV
        spacing: Pixel size in meters
    
    Properties:
        shape: Wave field shape (ny, nx)
        ndim: Number of dimensions (should be 2)
        size: Total number of elements
        wavelength: Wavelength in meters
        wavenumber: Wave number in radians per meter
    """
    # Instance attributes
    form: jnp.ndarray
    energy: float
    spacing: float
    
    # Physical constants
    HC: ClassVar[float] = 1.23984193e-6  # keV·m
    
    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray], Tuple[float, float]]:
        """Flatten the Wave into arrays and auxiliary data."""
        return (self.form,), (self.energy, self.spacing)
    
    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[float, float], 
                       arrays: Tuple[jnp.ndarray]) -> 'Wave':
        """Reconstruct Wave from flattened data."""
        return cls(form=arrays[0], energy=aux_data[0], spacing=aux_data[1])
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Wave field shape (ny, nx)."""
        return self.form.shape
    
    @property
    def ndim(self) -> int:
        """Number of dimensions (should be 2)."""
        return self.form.ndim
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.form.size
    
    @property
    def wavelength(self) -> float:
        """Wavelength in meters."""
        return self.HC / self.energy
    
    @property
    def wavenumber(self) -> float:
        """Wave number in radians per meter."""
        return 2 * jnp.pi / self.wavelength

    def __mul__(self, other: float) -> 'Wave':
        """Multiply wave form by a scalar."""
        return Wave(
            form=self.form * other,
            energy=self.energy,
            spacing=self.spacing
        )
    
    def __rmul__(self, other: float) -> 'Wave':
        """Right multiplication by a scalar."""
        return self.__mul__(other)
    
    def __sub__(self, other: 'Wave') -> 'Wave':
        """Subtract another wave's form."""
        return Wave(
            form=self.form - other.form,
            energy=self.energy,  # Keep original energy
            spacing=self.spacing  # Keep original spacing
        )
