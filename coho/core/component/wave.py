# Standard imports
import numpy as np
from typing import Union, Tuple, List

class Wave:
    """A class representing a complex wave field."""
    
    def __init__(self,
                 form: np.ndarray, 
                 energy: float = None, 
                 spacing: float = None, 
                 position: Union[float, np.ndarray] = None):
        """Initialize a wave."""
        form = np.asarray(form, dtype=np.complex128)
        self.form = form[np.newaxis, ...] if form.ndim == 2 else form
        self.energy = energy
        self.spacing = spacing
        self.position = position 
        self._freq2 = None  # Cache for freq2

    @property
    def wavelength(self) -> float:
        """Wavelength derived from energy in keV."""
        return np.divide(1.23984193e-7, self.energy)
    
    @property
    def wavenumber(self) -> float:
        """Wavenumber (2π divided by wavelength)."""
        return np.divide(2 * np.pi, self.wavelength)

    @property
    def real(self) -> np.ndarray:
        """Real part of wave form."""
        return self.form.real

    @property
    def imag(self) -> np.ndarray:
        """Imaginary part of wave form."""
        return self.form.imag

    @property
    def amplitude(self) -> np.ndarray:
        """Amplitude (absolute value) of wave form."""
        return np.abs(self.form)

    @property
    def phase(self) -> np.ndarray:
        """Phase of wave form."""
        return np.angle(self.form)
    
    def conjugate(self) -> 'Wave':
        """Conjugate the wave."""
        self.form = np.conjugate(self.form)
        return self

    def _prepare_broadcast(self, other: Union['Wave', float, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare arrays for multiplication without broadcasting."""
        form1 = self.form
        
        # Prepare other
        if isinstance(other, (float, int)):
            form2 = other
        else:
            form2 = other.form
        
        return form1, form2
    
    def __mul__(self, other: Union['Wave', float, int]) -> 'Wave':
        """Multiplication."""
        form1, form2 = self._prepare_broadcast(other)
        return Wave(
            form=form1 * form2,
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def __rmul__(self, other: Union[float, int]) -> 'Wave':
        """Right multiplication (scalar * wave)."""
        return self.__mul__(other)

    def __imul__(self, other: Union['Wave', float, int]) -> 'Wave':
        """In-place multiplication."""
        form1, form2 = self._prepare_broadcast(other)
        self.form = form1 * form2
        return self

    def __truediv__(self, other: Union['Wave', float, int], eps: float = 1e-10) -> 'Wave':
        """Division."""
        form1, form2 = self._prepare_broadcast(other)
        return Wave(
            form=form1 / (form2 + eps),
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def __itruediv__(self, other: Union['Wave', float, int], eps: float = 1e-10) -> 'Wave':
        """In-place division."""
        form1, form2 = self._prepare_broadcast(other)
        self.form = form1 / (form2 + eps)
        return self
    
    def __add__(self, other: Union['Wave', float, int]) -> 'Wave':
        """Addition."""
        form1, form2 = self._prepare_broadcast(other)
        return Wave(
            form=form1 + form2,
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def __iadd__(self, other: Union['Wave', float, int]) -> 'Wave':
        """In-place addition."""
        form1, form2 = self._prepare_broadcast(other)
        self.form = form1 + form2
        return self
    
    def __sub__(self, other: Union['Wave', float, int]) -> 'Wave':
        """Subtraction."""
        form1, form2 = self._prepare_broadcast(other)
        return Wave(
            form=form1 - form2,
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def __isub__(self, other: Union['Wave', float, int]) -> 'Wave':
        """In-place subtraction."""
        form1, form2 = self._prepare_broadcast(other)
        self.form = form1 - form2
        return self
    
    def __str__(self) -> str:
        """Simple string representation."""
        return f"Wave of shape {self.form.shape}"

    def __repr__(self) -> str:
        """String representation of Wave."""
        return (
            f"Wave(shape={self.shape}, "
            f"energy={self.energy}, "
            f"spacing={self.spacing}, "
            f"position={self.position})"
        )

    @property
    def intensity(self) -> np.ndarray:
        """Intensity of the wave (|ψ|²)."""
        return np.abs(self.form) ** 2

    @property
    def norm(self) -> float:
        """L2 norm of the wave form."""
        return np.sqrt((np.abs(self.form) ** 2).sum())

    def normalize(self, eps: float = 1e-10) -> 'Wave':
        """Normalize the wave."""
        self.form /= np.max(self.form) + eps
        return self

    def overlap(self, other: 'Wave') -> complex:
        """Calculate complex overlap integral <ψ1|ψ2>."""
        return np.sum(np.conjugate(self.form) * other.form)

    @property
    def freqs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get 2D spatial frequency coordinates (fy, fx).
        
        Works with both 2D and ND arrays, operating on last two dimensions.
        """
        if self.spacing is None:
            raise ValueError("Wave must have spacing defined")
        
        # Get last two dimensions
        *batch_dims, ny, nx = self.form.shape
        
        # Create base frequencies
        fx = np.fft.fftfreq(nx, d=self.spacing)
        fy = np.fft.fftfreq(ny, d=self.spacing)
        
        # Create 2D meshgrid
        fy, fx = np.meshgrid(fy, fx, indexing='ij')
        
        # Add batch dimensions if needed
        for _ in batch_dims:
            fy = fy[np.newaxis, ...]
            fx = fx[np.newaxis, ...]
        
        return fy, fx

    @property
    def freq2(self) -> np.ndarray:
        """Get squared spatial frequencies (fx² + fy²)."""
        if self._freq2 is None:
            fy, fx = self.freqs
            self._freq2 = fx**2 + fy**2
        return self._freq2

    @property
    def shape(self) -> Tuple[int, ...]:
        """Wave shape (ny, nx)."""
        return self.form.shape

    @property
    def size(self) -> int:
        """Total number of points."""
        return self.form.size

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.form.ndim

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Physical extent in meters (ymin, ymax, xmin, xmax)."""
        ny, nx = self.shape
        yext = (ny - 1) * self.spacing
        xext = (nx - 1) * self.spacing
        return (-yext/2, yext/2, -xext/2, xext/2)

    @property
    def coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Physical coordinates (y, x) in meters."""
        ny, nx = self.shape
        x = (np.arange(nx) - nx//2) * self.spacing
        y = (np.arange(ny) - ny//2) * self.spacing
        return np.meshgrid(y, x, indexing='ij')

    def crop(self, roi: Tuple[slice, slice]) -> 'Wave':
        """Crop wave to region of interest."""
        self.form = self.form[roi]
        return self

    def shift(self, shift: Union[int, Tuple[int, int]]) -> 'Wave':
        """Shift wave by pixels."""
        if isinstance(shift, (int, np.integer)):
            shift = (shift, shift)
        self.form = np.roll(np.roll(self.form, shift[0], axis=0), shift[1], axis=1)
        return self

    def rotate(self, angle: float, reshape: bool = False) -> 'Wave':
        """Rotate wave by angle in degrees."""
        from scipy.ndimage import rotate
        self.form = rotate(self.form, angle, reshape=reshape, order=1)
        return self

    def pad(self, pad_width: Union[int, Tuple, List], 
            mode: str = 'constant', constant_values: float = 0) -> 'Wave':
        """Pad wave with zeros or other values."""
        if isinstance(pad_width, (int, np.integer)):
            pad_width = [(pad_width, pad_width)] * self.ndim
        elif isinstance(pad_width, tuple) and len(pad_width) == 2:
            pad_width = [pad_width] * self.ndim
        
        self.form = np.pad(self.form, pad_width, mode=mode, constant_values=constant_values)
        return self
    
    @property
    def mean(self) -> np.ndarray:
        """Return mean along the first axis if ndim > 2."""
        if self.ndim <= 2:
            return self.form
        return np.mean(self.form, axis=0, keepdims=True)
    
    def zeros_like(self) -> 'Wave':
        """Create a new wave with zeros and same properties as current wave."""
        return Wave(
            form=np.zeros_like(self.form),
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )
    
    def ones_like(self) -> 'Wave':
        """Create a new wave with ones and same properties as current wave."""
        return Wave(
            form=np.ones_like(self.form),
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def copy(self) -> 'Wave':
        """Create a copy of the wave with same properties."""
        wave = Wave(
            form=self.form.copy(),  # numpy array copy
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )
        return wave

    def invalidate_cache(self):
        """Invalidate caches when wave form changes."""
        self._freq2 = None

    def multiply(self, n: int) -> 'Wave':
        """Create multiple copies of the wave along first dimension."""
        # Remove first dimension if it's 1
        base_form = self.form[0] if self.form.shape[0] == 1 else self.form
        
        # Create new form with n copies
        new_form = np.stack([base_form] * n)
        
        # Handle position
        if self.position is not None:
            new_position = np.full(n, self.position)
        else:
            new_position = None
        
        return Wave(
            form=new_form,  # Will be (n, ny, nx)
            energy=self.energy,
            spacing=self.spacing,
            position=new_position
        )

    def crop_to_match(self, reference: 'Wave', pad_value: float = 1.0) -> 'Wave':
        """Crop or pad wave to match reference wave dimensions."""
        if self.spacing != reference.spacing:
            raise ValueError("Waves must have the same spacing")
        
        result = self.copy()
        
        # Get spatial dimensions (last two)
        *batch_dims, ny, nx = self.shape
        *ref_batch, ref_ny, ref_nx = reference.shape
        
        # Calculate padding/cropping for each spatial dimension
        pad_y = max(0, (ref_ny - ny) // 2)
        pad_x = max(0, (ref_nx - nx) // 2)
        
        # Pad if needed
        if pad_y > 0 or pad_x > 0:
            # No padding for batch dimensions
            pad_width = [(0, 0)] * (self.ndim - 2) + [(pad_y, pad_y), (pad_x, pad_x)]
            result.pad(pad_width, mode='constant', constant_values=pad_value)
        
        # Calculate crop slices
        dy = (result.shape[-2] - ref_ny) // 2
        dx = (result.shape[-1] - ref_nx) // 2
        
        # Create slices (keep batch dimensions unchanged)
        slices = (slice(None),) * (self.ndim - 2) + (
            slice(dy, dy + ref_ny),
            slice(dx, dx + ref_nx)
        )
        
        result.crop(slices)
        return result


