# Waves

## Contents

- [Wave Creation](#wave-creation)
- [Wave Attributes](#wave-attributes)
- [Wave Properties](#wave-properties)
- [Arithmetic Operators](#arithmetic-operators)
- [Wave Operators](#wave-operators)
- [Device Management](#device-management)
- [Manipulations](#manipulations)
- [Coordinates and Frequencies](#coordinates-and-frequencies)
- [Broadcasting](#broadcasting)

## Wave Creation

```python
import torch
from coho import Wave

# From a tensor
wave = Wave(torch.ones(128, 128), energy=10)

# From a complex tensor
wave = Wave(torch.ones(128, 128) * torch.exp(1j * torch.pi), energy=10)

# From an image
wave = Wave.from_image('path/to/image.png')

# With device specification
wave = Wave(torch.ones(128, 128), energy=10, device='cuda')

# With position and spacing
wave = Wave(torch.ones(128, 128), energy=10, position=0.5, spacing=1e-6)

# From a callable function
wave = Wave(gaussian)

# From another Wave
wave = Wave(Wave(existing_wave))
```

## Wave Attributes

- `form`: torch.Tensor (batch, ny, nx) - complex128
- `energy`: float - beam energy in keV
- `spacing`: float - pixel size in meters
- `position`: torch.Tensor - z position in meters
- `x`, `y`: torch.Tensor - lateral positions in meters

## Wave Properties

```python
# Wavelength and wavenumber
wavelength = wave.wavelength
wavenumber = wave.wavenumber

# Real, imaginary, amplitude, and phase
real_part = wave.real
imag_part = wave.imag
amplitude = wave.amplitude
phase = wave.phase

# Intensity and norm
intensity = wave.intensity
norm = wave.norm

# Shape, size, and dimensions
shape = wave.shape
size = wave.size
ndim = wave.ndim
```

## Arithmetic Operators

```python
# Basic operations
wave = wave1 + wave2
wave = wave1 - wave2
wave = wave1 * wave2
wave = wave1 / wave2

# Scalar operations
wave = wave1 * 2
wave = wave1 / 2

# In-place operations
wave += wave1
wave -= wave1
wave *= wave1
wave /= wave1
wave += 2
wave -= 2
wave *= 2
wave /= 2
```

## Wave Operators

```python
# Complex operations
wave = wave.conjugate()

# Power
wave = wave1 ** 2

# Absolute value
wave = wave.abs() 

# Overlap integral
overlap = wave1.overlap(wave2)

# Amplitude normalization
wave = wave.normalize()

# Copying
wave_copy = wave.copy()
zero = wave.zeros_like()
ones = wave.ones_like()
```

## Device Management

```python
# Move to GPU
wave_gpu = wave.to('cuda')

# Move to CPU
wave_cpu = wave.to('cpu')

# Check device
device = wave.form.device
```

## Manipulations

```python
# Spatial operations
wave = wave.shift((10, 15))
wave = wave.crop((slice(10, 50), slice(20, 60)))
wave = wave.pad((10, 10), mode='constant', constant_values=0)
```

## Coordinates and Frequencies

```python
# Get frequency coordinates
fy, fx = wave.freqs 
freq2 = wave.freq2 

# Get physical coordinates
y, x = wave.coords # Real space coordinates
extent = wave.extent# (ymin, ymax, xmin, xmax)
```

## Broadcasting

```python
wave = Wave(np.random.rand(128, 128), energy=10, position=0.5)

# Replicate the wave 5 times
wave = wave.replicate(5)

print(wave.form.shape)  # Output: (5, 128, 128)
print(wave.position.shape)  # Output: (5,)
```

