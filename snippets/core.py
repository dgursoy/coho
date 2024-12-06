import torch
from coho import Wave

# Device setup (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example 1: Simple 2D wave from real values
shape = (64, 64)  # (height, width)
real_form = torch.ones(shape, device=device)
wave1 = Wave(
    form=real_form,  # Will be automatically converted to complex
    energy=10.0,     # 10 keV
    spacing=1e-6,    # 1 micron pixel size
    device=device
)

# Example 2: Complex 2D wave
complex_form = torch.ones(shape, dtype=torch.complex128, device=device)
complex_form *= torch.exp(1j * torch.randn(shape, device=device))  # Random phases
wave2 = Wave(form=complex_form, energy=8.0, spacing=1e-6, device=device)

# Example 3: Batch of waves (e.g., for multiple positions)
batch_size = 5
batch_form = torch.ones((batch_size, *shape), dtype=torch.complex128, device=device)
positions = torch.linspace(-1e-3, 1e-3, batch_size, device=device)  # positions in meters
wave3 = Wave(
    form=batch_form,
    energy=12.0,
    spacing=1e-6,
    position=positions,
    device=device
)

# Example 4: Gaussian wave
def create_gaussian_wave(size, sigma=1.0, device=None):
    y = torch.linspace(-size//2, size//2, size, device=device)
    x = torch.linspace(-size//2, size//2, size, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    gaussian = torch.exp(-(X**2 + Y**2)/(2*sigma**2))
    return Wave(
        form=gaussian,
        energy=10.0,
        spacing=1e-6,
        device=device
    )
wave4 = create_gaussian_wave(64, sigma=10.0, device=device)

# Example 5: Wave with amplitude and phase
amplitude = torch.ones(shape, device=device)
phase = torch.zeros(shape, device=device)
phase[32:, :] = torch.pi  # Phase shift in half of the wave
complex_form = amplitude * torch.exp(1j * phase)
wave5 = Wave(form=complex_form, energy=10.0, spacing=1e-6, device=device)


# Example of wave operations
result = wave1 * wave2  # Multiplication of waves
normalized = wave1.normalize()  # Normalize the wave
intensity = wave1.intensity  # Get intensity

# Moving waves between devices
wave_cpu = wave1.to('cpu')
if torch.cuda.is_available():
    wave_gpu = wave1.to('cuda')

print(repr(wave4))

import matplotlib.pyplot as plt
plt.imshow(wave4.intensity[0])
plt.colorbar()
plt.show()