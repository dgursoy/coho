"""Wave module."""

# Standard imports
from typing import Union, Tuple
import torch
from torch import Tensor

__all__ = ['Wave']

# Type aliases
Device = Union[str, torch.device]
Scalar = Union[float, int]
FormOrScalar = Union[Tensor, Scalar] 
OperandTensors = Tuple[Tensor, Tensor]

class Wave:
    """Complex wave field using PyTorch tensors.
    
    All operations maintain batch dimension (batch, n, n).
    Handles device placement and gradient computation.
    """

    # Core attributes - Define wave's tensor properties and caching behavior
    # List of tensor attributes that need to be moved to device
    _tensor_attrs = ['form', 'energy', 'spacing', 'position']
    
    # List of cached properties that should be cleared on device change
    _cached_attrs = ['_freq2']
    
    # Format specifications for repr
    _repr_formats = {
        'energy': '.1f',    # keV
        'spacing': '.1e',   # meters
        'position': '.1f',  # meters
    }

    def __init__(self, form: Tensor, energy: Tensor = None, 
                 spacing: Tensor = None, position: Tensor = None,
                 requires_grad: bool = False, device: Device = 'cpu') -> None:
        """Initialize wave with complex field and properties."""
        self.form = form
        self.energy = energy
        self.spacing = spacing
        self.position = position
        self.requires_grad = requires_grad
        self._freq2 = None
        self.to(device)

    # Basic properties - Core tensor attributes and device/grad management
    @property
    def device(self) -> torch.device:
        """Current device of wave tensors."""
        return self.form.device

    @device.setter
    def device(self, device: Union[str, torch.device]) -> None:
        """Move wave to specified device."""
        self.to(device)

    @property
    def requires_grad(self) -> bool:
        """Gradient computation status."""
        return self.form.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set gradient computation status."""
        self.form.requires_grad_(value)

    def to(self, device: Union[str, torch.device]) -> 'Wave':
        """Move wave to specified device."""
        if isinstance(device, str):
            device = torch.device(device)
        
        for attr in self._cached_attrs:
            setattr(self, attr, None)
        
        for attr in self._tensor_attrs:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device=device))
        return self

    # Physical properties - Wave characteristics derived from energy/wavelength
    @property
    def wavelength(self) -> float:
        """Wavelength in meters from energy (keV)."""
        return torch.divide(1.23984193e-7, self.energy)
    
    @property
    def wavenumber(self) -> float:
        """Wavenumber (2π/wavelength) in inverse meters."""
        return torch.divide(2 * torch.pi, self.wavelength)

    # Wave properties - Complex field characteristics
    @property
    def real(self) -> Tensor:
        """Real part of shape (batch, n, n)."""
        return self.form.real

    @property
    def imag(self) -> Tensor:
        """Imaginary part of shape (batch, n, n)."""
        return self.form.imag

    @property
    def intensity(self) -> Tensor:
        """Intensity |ψ|² of shape (batch, n, n)."""
        return torch.abs(self.form).square()

    @property
    def amplitude(self) -> Tensor:
        """Amplitude of shape (batch, n, n)."""
        return torch.abs(self.form)

    @property
    def phase(self) -> Tensor:
        """Phase angle of shape (batch, n, n)."""
        return torch.angle(self.form)

    @property
    def mean(self) -> Tensor:
        """Mean along batch dimension of shape (1, n, n)."""
        return torch.mean(self.form, dim=0, keepdim=True)

    # Spatial properties - Physical space and frequency domain coordinates
    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Physical extent (ymin, ymax, xmin, xmax) in meters."""
        ny, nx = self.shape[-2:]
        half_y = (ny - 1) * self.spacing.item() / 2
        half_x = (nx - 1) * self.spacing.item() / 2
        return -half_y, half_y, -half_x, half_x

    @property
    def coords(self) -> Tuple[Tensor, Tensor]:
        """Physical coordinates (y, x) in meters of shape (n, n)."""
        ny, nx = self.shape[-2:]
        x = (torch.arange(nx, device=self.device) - nx//2) * self.spacing
        y = (torch.arange(ny, device=self.device) - ny//2) * self.spacing
        return torch.meshgrid(y, x, indexing='ij')

    @property
    def freqs(self) -> Tuple[Tensor, Tensor]:
        """Spatial frequencies (fy, fx) of shape (batch, n, n)."""
        *batch_dims, ny, nx = self.form.shape
        fx = torch.fft.fftfreq(nx, d=self.spacing.item())
        fy = torch.fft.fftfreq(ny, d=self.spacing.item())
        fy, fx = torch.meshgrid(fy, fx, indexing='ij')
        return fy.to(self.device), fx.to(self.device)
    
    @property
    def freq2(self) -> Tensor:
        """Squared spatial frequencies (fx² + fy²)."""
        if self._freq2 is None:
            fy, fx = self.freqs
            self._freq2 = fx.square() + fy.square()
        return self._freq2

    # Shape properties - Tensor dimension information
    @property
    def shape(self) -> Tuple[int, ...]:
        """Wave shape (batch, n, n)."""
        return self.form.shape

    @property
    def size(self) -> int:
        """Total number of points."""
        return self.form.size

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.form.ndim

    # Factory methods - Create new waves with specific patterns
    @classmethod
    def zeros_like(cls, other: 'Wave') -> 'Wave':
        """Create wave of zeros with same properties as other."""
        return cls(
            form=torch.zeros_like(other.form),
            energy=other.energy,
            spacing=other.spacing,
            position=other.position,
            requires_grad=other.form.requires_grad
        )

    @classmethod
    def ones_like(cls, other: 'Wave') -> 'Wave':
        """Create wave of ones with same properties as other."""
        return cls(
            form=torch.ones_like(other.form),
            energy=other.energy,
            spacing=other.spacing,
            position=other.position,
            requires_grad=other.form.requires_grad
        )

    @classmethod
    def rand_like(cls, other: 'Wave') -> 'Wave':
        """Create wave with random amplitude [0,1] and phase [-π,π]."""
        amp = torch.rand_like(other.form, dtype=torch.float)
        phase = (torch.rand_like(other.form, dtype=torch.float) * 2 - 1) * torch.pi
        return cls(
            form=amp * torch.exp(1j * phase),
            energy=other.energy,
            spacing=other.spacing,
            position=other.position,
            requires_grad=other.form.requires_grad
        )

    # Basic operations - Wave field transformations
    def vectorize(self, size: int) -> 'Wave':
        """Expand wave along first dimension for parallel processing."""
        if len(self.form.shape) > 2:
            raise ValueError("Wave is already vectorized")
        self.form = self.form.expand(size, *self.form.shape[-2:])
        return self
    
    def normalize(self, *, eps: float = 1e-10) -> 'Wave':
        """Returns wave normalized by maximum amplitude."""
        max_val = torch.abs(self.form).amax(dim=(-2,-1), keepdim=True)
        return Wave(
            form=self.form / (max_val + eps),
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def normalize_(self, *, eps: float = 1e-10) -> 'Wave':
        """In-place version of normalize()."""
        self.form /= (torch.abs(self.form).amax(dim=(-2,-1), keepdim=True) + eps)
        return self

    def conjugate(self) -> 'Wave':
        """Returns complex conjugate of wave."""
        return Wave(
            form=torch.conj(self.form),
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def conjugate_(self) -> 'Wave':
        """In-place version of conjugate()."""
        self.form = torch.conj(self.form)
        return self

    def clone(self) -> 'Wave':
        """Returns deep copy with same properties and device."""
        attrs = {
            name: getattr(self, name).clone() 
            for name in self._tensor_attrs
            if hasattr(self, name)
        }
        attrs['requires_grad'] = self.form.requires_grad
        return Wave(**attrs)

    # Spatial operations - Manipulate wave's spatial representation
    def roll(self, shifts: Union[int, Tuple[int, int]]) -> 'Wave':
        """Returns wave rolled by (y, x) pixels."""
        if isinstance(shifts, (int, Tensor)):
            shifts = (shifts, shifts)
        return Wave(
            form=torch.roll(self.form, shifts=shifts, dims=(-2, -1)),
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def roll_(self, shifts: Union[int, Tuple[int, int]]) -> 'Wave':
        """In-place version of roll()."""
        if isinstance(shifts, (int, Tensor)):
            shifts = (shifts, shifts)
        self.form = torch.roll(self.form, shifts=shifts, dims=(-2, -1))
        return self
    
    def crop(self, roi: Tuple[slice, slice]) -> 'Wave':
        """Crop wave to region of interest (in-place)."""
        self.form = self.form[..., roi[0], roi[1]]
        return self

    def pad(self, padding: Union[int, Tuple[int, int]], *, 
            mode: str = 'constant', value: float = 0) -> 'Wave':
        """Returns padded wave in spatial dimensions."""
        if isinstance(padding, (int, Tensor)):
            padding = (padding, padding)
        pad = (padding[1], padding[1], padding[0], padding[0])
        return Wave(
            form=torch.nn.functional.pad(self.form, pad, mode=mode, value=value),
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def resize_as(self, other: 'Wave', *, value: float = 0.0) -> 'Wave':
        """Returns wave resized to match other's dimensions.
        
        Centers while padding/cropping to match size.
        """
        if self.spacing != other.spacing:
            raise ValueError("Waves must have same spacing")
        
        ny, nx = self.shape[-2:]
        other_ny, other_nx = other.shape[-2:]
        result = self.clone()
        
        if other_ny > ny or other_nx > nx:
            pad_y = max(0, (other_ny - ny) // 2)
            pad_x = max(0, (other_nx - nx) // 2)
            result = result.pad((pad_y, pad_x), mode='constant', value=value)
        
        if result.shape[-2:] != other.shape[-2:]:
            dy = (result.shape[-2] - other_ny) // 2
            dx = (result.shape[-1] - other_nx) // 2
            roi = (
                slice(dy, dy + other_ny),
                slice(dx, dx + other_nx)
            )
            result = result.crop(roi)
        
        return result

    # Math operations - Arithmetic between waves or with scalars
    def _maybe_get_operands(self, other: FormOrScalar) -> OperandTensors:
        """Get operands for arithmetic, handling tensor and scalar cases.
        
        Returns:
            tuple: (self.form, other_form) on same device
        """
        if isinstance(other, (float, int)):
            return self.form, other
            
        if hasattr(other, 'form'):
            if other.form.device != self.form.device:
                raise RuntimeError(
                    f"Expected same device, got {self.form.device} and {other.form.device}"
                )
            return self.form, other.form
        
        raise TypeError(f"Cannot operate with {type(other)}")

    def __add__(self, other: FormOrScalar) -> 'Wave':
        """Returns wave + other."""
        form1, form2 = self._maybe_get_operands(other)
        return Wave(
            form=form1 + form2,
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )
    
    def __sub__(self, other: FormOrScalar) -> 'Wave':
        """Returns wave - other."""
        form1, form2 = self._maybe_get_operands(other)
        return Wave(
            form=form1 - form2,
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )
    
    def __mul__(self, other: FormOrScalar) -> 'Wave':
        """Returns wave * other."""
        form1, form2 = self._maybe_get_operands(other)
        return Wave(
            form=form1 * form2,
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    def __truediv__(self, other: FormOrScalar, *, eps: float = 1e-10) -> 'Wave':
        """Returns wave / other with numerical stability."""
        form1, form2 = self._maybe_get_operands(other)
        return Wave(
            form=form1 / (form2 + eps),
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    # In-place math operations
    def __iadd__(self, other: FormOrScalar) -> 'Wave':
        """In-place version of add."""
        form1, form2 = self._maybe_get_operands(other)
        self.form = form1 + form2
        return self

    def __isub__(self, other: FormOrScalar) -> 'Wave':
        """In-place version of subtract."""
        form1, form2 = self._maybe_get_operands(other)
        self.form = form1 - form2
        return self

    def __imul__(self, other: FormOrScalar) -> 'Wave':
        """In-place version of multiply."""
        form1, form2 = self._maybe_get_operands(other)
        self.form = form1 * form2
        return self

    def __itruediv__(self, other: FormOrScalar, *, eps: float = 1e-10) -> 'Wave':
        """In-place version of divide."""
        form1, form2 = self._maybe_get_operands(other)
        self.form = form1 / (form2 + eps)
        return self
    
    # Reverse math operations
    def __rmul__(self, other: Scalar) -> 'Wave':
        """Returns other * wave."""
        return self.__mul__(other)
    
    def __rtruediv__(self, other: Scalar, *, eps: float = 1e-10) -> 'Wave':
        """Returns other / wave."""
        return Wave(
            form=other / (self.form + eps),
            energy=self.energy,
            spacing=self.spacing,
            position=self.position
        )

    # String representations
    def __str__(self) -> str:
        """Simple string representation."""
        return f"Wave of shape {self.form.shape} on {self.form.device}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        attrs = []
        for attr in self._tensor_attrs:
            if hasattr(self, attr):
                val = getattr(self, attr)
                fmt = self._repr_formats.get(attr, '')
                val_str = f"{float(val[0]):{fmt}}" if len(val) == 1 else "[...]"
                attrs.append(f"{attr}={val_str}")
        return f"Wave({', '.join(attrs)})"
