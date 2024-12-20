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
        position: Position of the wave in meters
        mode: Wave propagation mode ('parallel' or 'focused')

    Properties:
        shape: Wave field shape (ny, nx)
        ndim: Number of dimensions (should be 2)
        size: Total number of elements
        wavelength: Wavelength in meters
        wavenumber: Wave number in radians per meter
    """
    # Dynamic field (will be traced)
    form: jnp.ndarray
    
    # Static fields (won't be traced)
    energy: float = 10.0
    spacing: float = 1e-4
    position: float = 0.0
    mode: str = 'focused'
    
    # Physical constants
    HC: ClassVar[float] = 1.23984193e-6  # keV·m
    
    def tree_flatten(self):
        """Separate dynamic and static values."""
        dynamic = (self.form,)
        static = {
            'energy': self.energy,
            'spacing': self.spacing,
            'position': self.position,
            'mode': self.mode
        }
        return dynamic, static
    
    @classmethod
    def tree_unflatten(cls, static, dynamic):
        """Reconstruct Wave from flattened data."""
        form, = dynamic
        return cls(form=form, **static)
    
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
    
    def __add__(self, other: 'Wave') -> 'Wave':
        """Add another wave's form."""
        return Wave(
            form=self.form + other.form,
            energy=self.energy,
            spacing=self.spacing,
            position=self.position,
            mode=self.mode,
            divergence=self.divergence
        )
    
    def __sub__(self, other: 'Wave') -> 'Wave':
        """Subtract another wave's form."""
        return Wave(
            form=self.form - other.form,
            energy=self.energy,  # Keep original energy
            spacing=self.spacing,  # Keep original spacing
            position=self.position,
            mode=self.mode,
            divergence=self.divergence
        )
    
    def __neg__(self) -> 'Wave':
        """Negate the wave form."""
        return Wave(
            form=-self.form,
            energy=self.energy,
            spacing=self.spacing,
            position=self.position,
            mode=self.mode,
            divergence=self.divergence
        )

    def __mul__(self, other: float) -> 'Wave':
        """Multiply wave form by a scalar."""
        return Wave(
            form=self.form * other,
            energy=self.energy,
            spacing=self.spacing,
            position=self.position,
            mode=self.mode,
            divergence=self.divergence
        )
    
    def __rmul__(self, other: float) -> 'Wave':
        """Right multiplication by a scalar."""
        return self.__mul__(other)
