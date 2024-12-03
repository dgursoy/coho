# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho import Wave, Pipeline, Propagate, Modulate, Detect, Broadcast

# Load test images
lena = np.load('./coho/resources/images/lena.npy')
cameraman = np.load('./coho/resources/images/cameraman.npy')

# Initialize waves
wave = Wave(lena, energy=10.0, spacing=1e-4, position=0.0).normalize()
sample = Wave(cameraman, energy=10.0, spacing=1e-4, position=0.0).normalize() 
wave += 0.5
sample += 0.5
wave0 = wave.normalize()
sample0 = sample.normalize()

# Initialize detector with reference wave
detector = wave0.ones_like()
detector.position = 400

# Distances and positions
wave_positions = [0, 0, 0, 0]
sample_positions = [0, 100, 200, 300]
detector_position = 400
source_to_sample = np.subtract(sample_positions, wave_positions)
sample_to_detector = np.subtract(detector_position, sample_positions)

# Prepare wave (as in basic.py)
broadcast = Broadcast()
wave0 = broadcast.apply(wave0, position=wave_positions)
wave = Propagate().apply(wave0, distance=source_to_sample)

# Define pipeline for both forward and adjoint
pipeline = Pipeline([
    (Broadcast(), {'position': sample_positions}),
    (Modulate(), {'modulator': wave}),
    (Propagate(), {'distance': sample_to_detector}),
    (Modulate(), {'modulator': detector}),
    (Detect(), {})
])

# Forward: Apply pipeline
intensity = pipeline.apply(sample0)

# Adjoint: Use the same pipeline
reconstructed = pipeline.adjoint(intensity)

# Plot measured intensities
plt.figure(figsize=(20, 5))
num_positions = len(sample_positions)
for i in range(num_positions):
    plt.subplot(1, num_positions, i+1)
    plt.imshow(intensity[i], cmap='gray')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Plot reconstruction
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('Original Sample')
plt.imshow(sample0.amplitude[0], cmap='gray')
plt.colorbar()
plt.subplot(122)
plt.title('Reconstructed Sample')
plt.imshow(reconstructed.amplitude[0], cmap='gray')
plt.colorbar()
plt.tight_layout()
plt.show()