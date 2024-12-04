# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho.core.component import Wave
from coho.core.pipeline import MultiDistanceHolography


# Load test images
lena = np.load('./coho/resources/images/lena.npy') + 256.
cameraman = np.load('./coho/resources/images/cameraman.npy') + 256.

# Initialize waves
reference = Wave(lena, energy=10.0, spacing=1e-4, position=0.0).normalize()
sample = Wave(cameraman, energy=10.0, spacing=1e-4, position=0.0).normalize()
detector = Wave(np.ones_like(lena), energy=10.0, spacing=1e-4, position=400.0).normalize()

# Create sample wave at multiple positions
sample_positions = [100]  # Multiple sample positions

# Create and run pipeline
pipeline = MultiDistanceHolography(reference, detector, sample_positions)
intensity = pipeline.apply(sample)

# Plot intensity
plt.figure(figsize=(20, 5))
num_positions = len(sample_positions)
for i in range(num_positions):
    plt.subplot(1, num_positions, i+1)
    plt.imshow(intensity[i], cmap='gray')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Adjoint
sample = pipeline.adjoint(intensity)

# Plot adjoint
plt.figure(figsize=(20, 5))
plt.imshow(sample.amplitude, cmap='gray')
plt.tight_layout()
plt.show()