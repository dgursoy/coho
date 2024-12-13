# Waves

Waves are the basic building blocks of an imaging setup. They represent the wavefront of the light field and are used to simulate the propagation and modulation of light through an imaging system. We use the `Wave` class to create and manipulate waves. Basically, a wave is a complex-valued [tensor](https://pytorch.org/docs/stable/tensors.html) with additional properties such as energy, spacing, position, and frequency. Below, we cover the basics of how to create and manipulate waves.

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

Waves can be created from a variety of sources. Below, we cover the most common ways to create waves.

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
wave = Wave.copy()
```

## Wave Attributes

Waves have the following attributes:

- `form`: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) (..., ny, nx) - complex-valued tensor where the first dimension is the batch dimension
- `energy`: [float](https://docs.python.org/3/library/functions.html#float) or [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) - beam energy in keV (optional)
- `spacing`: [float](https://docs.python.org/3/library/functions.html#float) or [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) - pixel size in centimeters (optional)
- `position`: [float](https://docs.python.org/3/library/functions.html#float) or [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) - position along optical axis in centimeters (optional)
- `x`, `y`: [float](https://docs.python.org/3/library/functions.html#float) or [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) - lateral positions in centimeters (optional)

## Wave Properties

Waves have the following properties that are computed from the wave's attributes:

```python
# Wavelength and wavenumber
wavelength = wave.wavelength # Wavelength in centimeters
wavenumber = wave.wavenumber # Wavenumber in 1/centimeters

# Real, imaginary, amplitude, and phase
real_part = wave.real # Real part of wave
imag_part = wave.imag # Imaginary part of wave
amplitude = wave.amplitude # Amplitude of wave
phase = wave.phase # Phase of wave

# Intensity and norm
intensity = wave.intensity # Intensity of wave
norm = wave.norm # Normalization factor

# Shape, size, and dimensions
shape = wave.shape # Shape of wave
size = wave.size # Size of wave
ndim = wave.ndim # Number of dimensions

# Frequency coordinates
fy, fx = wave.freqs # Frequency coordinates (1, ny, nx)
freq2 = wave.freq2 # Frequency squared (1, ny, nx)

# Physical coordinates
y, x = wave.coords # Real space coordinates
extent = wave.extent # [ymin, ymax, xmin, xmax]
```

## Arithmetic Operators

We can perform arithmetic operations on waves to create new waves. These operations are performed element-wise on the wave's form. The other attributes are copied over from the left operand to the new wave.

```python
# Basic operations
wave = wave1 + wave2 # Add waves
wave = wave1 - wave2 # Subtract waves
wave = wave1 * wave2 # Multiply waves
wave = wave1 / wave2 # Divide waves

# Scalar operations
wave = wave1 * 2 # Multiply wave by scalar
wave = wave1 / 2 # Divide wave by scalar

# In-place operations
wave += wave1 # Add wave to wave
wave -= wave1 # Subtract wave from wave
wave *= wave1 # Multiply wave by wave
wave /= wave1 # Divide wave by wave
wave += 2 # Add scalar to wave
wave -= 2 # Subtract scalar from wave
wave *= 2 # Multiply wave by scalar
wave /= 2 # Divide wave by scalar
```

## Wave Methods

We can perform basic manipulations on waves using built-in methods. These methods are lightweight and can be used to perform common operations on waves. For more complex manipulations, we recommend using [operators](operators.md).

```python
# Basic operations
wave = wave.conjugate() # Complex conjugate
wave = wave ** 2 # Power   
wave = wave.abs() # Absolute value
overlap = wave1.overlap(wave2) # Overlap integral
wave = wave.normalize() # Amplitude normalization

# Spatial operations
wave = wave.shift((10, 15)) # Shift wave
wave = wave.crop((slice(10, 50), slice(20, 60))) # Crop wave
wave = wave.pad((10, 10), mode='constant', constant_values=0) # Pad wave

# Copying
wave_copy = wave.copy() # Copy wave
zero = wave.zeros_like() # Create zero wave
ones = wave.ones_like() # Create ones wave
```

## Device Management

We can move waves to different devices using the `to` method. This method is used to move the wave's form and position attributes, which are the most expensive to move, to the specified device. Other attributes such as energy or spacing are not moved.

```python
# Move to GPU
wave_gpu = wave.to('cuda') 

# Move to CPU
wave_cpu = wave.to('cpu')

# Check device
device = wave.form.device
```

## Stacking

We can stack waves along the first dimension using the `stack` method. This method is used to stack waves along the first dimension, which is the batch dimension. This is useful for performing operations on multiple waves at once. 

```python
wave = Wave(np.random.rand(128, 128), energy=10, position=0.5)

# By default, the wave is stacked along the first dimension
print(wave.form.shape)  # Output: (1, 128, 128)

# Replicate the wave 5 times
wave = wave.repeat(5)

print(wave.form.shape)  # Output: (5, 128, 128) Form is repeated
print(wave.position.shape)  # Output: (1,) Other attributes are not repeated
```

