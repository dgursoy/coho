# Waves

## Contents

- [Wave Creation](#wave-creation)
- [Wave Attributes](#wave-attributes)
- [Wave Properties](#wave-properties)
- [Arithmetic Operators](#arithmetic-operators)
- [Wave Operators](#wave-operators)
- [Transformations](#transformations)
- [Manipulations](#manipulations)
- [Coordinates and Extent](#coordinates-and-extent)
- [Broadcasting](#broadcasting) 
- [Visualization](#visualization)

## Wave Creation

```python
# From a list
wave = Wave([1, 2, 3], energy=10)

# From an image
wave = Wave.from_image('path/to/image.png')

# From a numpy array
wave = Wave(np.random.rand(128, 128) * np.exp(1j * np.random.rand(128, 128)))

# From a callable function
wave = Wave(gaussian)

# From another Wave
wave = Wave(Wave(existing_wave))
```

## Wave Attributes

- `form`: array (batch, ny, nx)
- `energy`: float 
- `spacing`: float
- `position`: float (batch,) (broadcastable to form)
- `shift`: tuple of ints (batch, 2) (broadcastable to form)

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
# Addition and subtraction
wave = wave1 + wave2
wave = wave1 - wave2

# Multiplication and division
wave = wave1 * wave2
wave = wave1 / 2

# In-place modifications
wave1 += wave2
wave1 *= 2
```

## Wave Operators

```python
# Power
wave = wave1 ** 2

# Conjugate
wave = wave.conjugate()

# Absolute value
wave = wave.abs() 

# Overlap integral
overlap = wave1.overlap(wave2)

# Normalize
wave = wave.normalize()
```

## Transformations

```python
# Fourier transform 
wave = wave.fft()

# Inverse Fourier transform
wave = wave.ifft()
```

## Manipulations

```python
# Shifting in space
wave = wave.shift((10, 15))

# Rotating
wave = wave.rotate(45)

# Cropping
wave = wave.crop((slice(10, 50), slice(20, 60)))

# Padding
wave = wave.pad((10, 10), mode='constant', constant_values=0)
```

## Coordinates and Extent

```python
# Get frequency coordinates
fy, fx = wave.freqs

# Get physical extent
extent = wave.extent

# Get physical coordinates
y, x = wave.coords
```

## Broadcasting

```python
wave = Wave(np.random.rand(128, 128), energy=10, position=0.5)

# Replicate the wave 5 times
wave = wave.replicate(5)

print(wave.form.shape)  # Output: (5, 128, 128)
print(wave.position.shape)  # Output: (5,)
```

## Visualization

```python
# Plot amplitude
wave.plot(batch=0, kind='amplitude')

# Plot phase
wave.plot(batch=0, kind='phase')

# Plot intensity
wave.plot(batch=0, kind='intensity')
```
