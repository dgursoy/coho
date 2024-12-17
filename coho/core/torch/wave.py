"""Wave module."""

# Standard imports
from typing import Union, Tuple, Optional, Dict
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn

__all__ = ['Wave']

# Type aliases
Device = Union[str, torch.device]
Scalar = Union[float, int]
FormOrScalar = Union[Tensor, Scalar] 
OperandTensors = Tuple[Tensor, Tensor]

class Wave(nn.Module):
    """Complex wave field using PyTorch tensors.
    
    All operations maintain batch dimension (batch, n, n).
    Handles device placement and gradient computation.
    """
    # Class-level parameter definitions
    PARAMETERS = ['form']  # Parameters that can be optimized
    BUFFERS = ['energy', 'spacing', 'position']  # Non-optimizable buffers
    
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
                 device: Device = 'cpu') -> None:
        """Initialize wave with complex field and properties.
        
        Args:
            form: Complex wave field of shape (batch, height, width)
            energy: Photon energy in keV
            spacing: Pixel size in meters
            position: Wave position in meters
            device: Target device for wave tensors
        """
        super().__init__()

        # Register parameters and buffers based on class definitions
        for name in self.PARAMETERS:
            value = locals()[name]
            if value is not None:
                self.register_parameter(name, nn.Parameter(value))
                
        for name in self.BUFFERS:
            value = locals()[name]
            if value is not None:
                self.register_buffer(name, value)

        # Initialize cached properties
        self._freq2 = None
        
        # Move to device
        self.to(device)

    # Core parameter management
    def clone(self) -> 'Wave':
        """Create a new Wave instance preserving all properties and their types."""
        return Wave(
            **{
                name: (nn.Parameter(value) if isinstance(value, nn.Parameter) else value)
                for name in self.PARAMETERS + self.BUFFERS
                if (value := getattr(self, name, None)) is not None
            },
            device=self.device
        )

    def clone_with(self, **updates) -> 'Wave':
        """Create new Wave with updated values, preserving computation graph."""
        wave = self.clone()
        
        for name, value in updates.items():
            if hasattr(self, name):
                if isinstance(getattr(self, name), nn.Parameter):
                    delattr(wave, name)  # Remove existing Parameter
                    # Always register as Parameter to maintain gradient flow
                    wave.register_parameter(name, nn.Parameter(value))
                else:
                    setattr(wave, name, value)
        
        return wave
    
    @property
    def device(self) -> torch.device:
        """Current device of wave tensors."""
        return self.form.device

    @device.setter
    def device(self, device: Union[str, device]) -> None:
        """Move wave to specified device."""
        self.to(device)

    @property
    def requires_grad(self) -> bool:
        """Whether gradients are computed for this wave's form tensor."""
        return self.form.requires_grad

    def to(self, device: Union[str, device]) -> 'Wave':
        """Move wave to specified device."""
        if isinstance(device, str):
            device = torch.device(device)
        
        # Clear cached properties
        for attr in self._cached_attrs:
            setattr(self, attr, None)
        
        super().to(device)
        return self

    # Physical properties - Wave characteristics derived from energy
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
        """Returns wave of zeros with same properties as template."""
        return other.clone_with(form=torch.zeros_like(other.form))

    @classmethod
    def ones_like(cls, other: 'Wave') -> 'Wave':
        """Returns wave of ones with same properties as template."""
        return other.clone_with(form=torch.ones_like(other.form))

    @classmethod
    def rand_like(cls, other: 'Wave') -> 'Wave':
        """Returns wave of random values with same properties as template."""
        return other.clone_with(form=torch.rand_like(other.form))

    # Basic operations - Wave field transformations
    def vectorize(self, size: int, attr: str = 'form') -> 'Wave':
        """Expand specified attribute to batch size by repeating."""
        value = getattr(self, attr)
        expanded = value.expand(size, *value.shape[1:])
        return self.clone_with(**{attr: expanded})
    
    def normalize(self, *, eps: float = 1e-10) -> 'Wave':
        """Returns normalized wave by maximum amplitude."""
        max_val = torch.abs(self.form).amax(dim=(-2,-1), keepdim=True)
        return self.clone_with(form=torch.div(self.form, max_val + eps))

    def normalize_(self, *, eps: float = 1e-10) -> 'Wave':
        """In-place version of normalize()."""
        max_val = torch.abs(self.form).amax(dim=(-2,-1), keepdim=True)
        self.form.div_(max_val + eps)  # True in-place using div_
        return self

    def conj(self) -> 'Wave':
        """Returns complex conjugate of wave."""
        return self.clone_with(form=torch.conj(self.form))

    def conj_(self) -> 'Wave':
        """In-place version of conj()."""
        self.form = torch.conj(self.form)
        return self

    # Spatial operations - Manipulate wave's spatial representation
    def roll(self, shifts: Union[int, Tuple[int, int]]) -> 'Wave':
        """Returns wave rolled by (y, x) pixels."""
        if isinstance(shifts, (int, Tensor)):
            shifts = (shifts, shifts)
        return self.clone_with(form=torch.roll(self.form, shifts=shifts, dims=(-2, -1)))

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
        return self.clone_with(form=F.pad(self.form, pad, mode=mode, value=value))

    def pad_(self, padding: Union[int, Tuple[int, int]], *, 
             mode: str = 'constant', value: float = 0) -> 'Wave':
        """In-place version of pad()."""
        if isinstance(padding, (int, Tensor)):
            padding = (padding, padding)
        pad = (padding[1], padding[1], padding[0], padding[0])
        self.form = F.pad(self.form, pad, mode=mode, value=value)
        return self

    def resize(self, size: Tuple[int, int], mode: str = 'bilinear') -> 'Wave':
        """Returns wave resized to size using amplitude/phase interpolation."""
        return self.clone_with(form=self._resize_form(size, mode))

    def resize_(self, size: Tuple[int, int], mode: str = 'bilinear') -> 'Wave':
        """In-place version of resize()."""
        self.form = self._resize_form(size, mode)
        return self

    # Math operations - Arithmetic between waves or with scalars
    def _maybe_get_operands(self, other: FormOrScalar) -> OperandTensors:
        """Get operands for arithmetic, handling tensor and scalar cases."""
        if isinstance(other, (float, int)):
            return self.form, other
        if hasattr(other, 'form'):
            return self.form, other.form

    # Regular arithmetic operations
    def __add__(self, other: FormOrScalar) -> 'Wave':
        """Returns wave + other."""
        form1, form2 = self._maybe_get_operands(other)
        return self.clone_with(form=form1 + form2)
    
    def __sub__(self, other: FormOrScalar) -> 'Wave':
        """Returns wave - other."""
        form1, form2 = self._maybe_get_operands(other)
        return self.clone_with(form=form1 - form2)
    
    def __mul__(self, other: FormOrScalar) -> 'Wave':
        """Returns wave * other."""
        form1, form2 = self._maybe_get_operands(other)
        return self.clone_with(form=form1 * form2)

    def __truediv__(self, other: FormOrScalar, *, eps: float = 1e-10) -> 'Wave':
        """Returns wave / other with numerical stability."""
        form1, form2 = self._maybe_get_operands(other)
        return self.clone_with(form=form1 / (form2 + eps))

    # In-place arithmetic operations
    def __iadd__(self, other: FormOrScalar) -> 'Wave':
        """In-place version of add."""
        _, form2 = self._maybe_get_operands(other)
        self.form = nn.Parameter(self.form + form2)
        return self

    def __isub__(self, other: FormOrScalar) -> 'Wave':
        """In-place version of subtract."""
        _, form2 = self._maybe_get_operands(other)
        self.form = nn.Parameter(self.form - form2)
        return self

    def __imul__(self, other: FormOrScalar) -> 'Wave':
        """In-place version of multiply."""
        _, form2 = self._maybe_get_operands(other)
        self.form = nn.Parameter(self.form * form2)
        return self

    def __itruediv__(self, other: FormOrScalar, *, eps: float = 1e-10) -> 'Wave':
        """In-place version of divide."""
        _, form2 = self._maybe_get_operands(other)
        self.form = nn.Parameter(self.form / (form2 + eps))
        return self
    
    # Reverse arithmetic operations
    def __rmul__(self, other: Scalar) -> 'Wave':
        """Returns other * wave."""
        return self.__mul__(other)  # Reuse __mul__ since multiplication is commutative

    def __rtruediv__(self, other: Scalar, *, eps: float = 1e-10) -> 'Wave':
        """Returns other / wave."""
        form1, form2 = self._maybe_get_operands(other)  # other is scalar, self.form is denominator
        return self.clone_with(form=form1 / (form2 + eps))

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
                val_str = f"{val.item():{fmt}}" if val.numel() == 1 else "[...]"
                attrs.append(f"{attr}={val_str}")
        return f"Wave({', '.join(attrs)})"

    # Helper methods
    def _inherit_grad(self, *tensors) -> bool:
        """Determine gradient requirement from parent tensors."""
        return any(t.requires_grad for t in tensors if hasattr(t, 'requires_grad'))
    
    def _resize_form(self, size: Tuple[int, int], mode: str = 'bilinear') -> Tuple[Tensor, Tensor]:
        """Helper to resize wave form."""
        amp_resized = F.interpolate(
            self.amplitude.unsqueeze(1), 
            size=size, 
            mode=mode
        ).squeeze(1)
        
        phase_resized = F.interpolate(
            self.phase.unsqueeze(1), 
            size=size, 
            mode=mode
        ).squeeze(1)
        
        return torch.polar(amp_resized, phase_resized)
    
    # Plotting
    def plot(self, title: str = "Wave", height: int = 6) -> None:
        """Plot amplitude and phase for up to 4 waves in the batch.
        
        Args:
            title: Base title for the plot
            height: Figure height in inches (width will be proportional to batch size)
        """
        import matplotlib.pyplot as plt
        
        # Get number of waves to plot (up to 4)
        n_waves = min(4, self.form.shape[0])
        
        # Adjust figure width based on number of waves (5 inches per wave)
        width = 4 * n_waves
        plt.figure(figsize=(width, height))
        
        for m in range(n_waves):
            # Amplitude plot
            plt.subplot(2, n_waves, m+1)
            plt.title(f'{title} {m+1} Amplitude')
            plt.imshow(self.form[m].abs().detach().cpu(), cmap='gray')
            plt.colorbar()
            
            # Phase plot
            plt.subplot(2, n_waves, m+n_waves+1)
            plt.title(f'{title} {m+1} Phase')
            plt.imshow(self.form[m].angle().detach().cpu(), cmap='gray')
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()