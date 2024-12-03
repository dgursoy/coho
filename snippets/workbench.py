# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho import Wave, Propagate, Modulate, Detect, Broadcast

# Load test images
lena = np.load('./coho/resources/images/lena.npy')
cameraman = np.load('./coho/resources/images/cameraman.npy')

# Initialize reference and sample waves
wave = Wave(lena, energy=10.0, spacing=1e-4, position=0.0).normalize()
sample = Wave(cameraman, energy=10.0, spacing=1e-4, position=0.0).normalize() 
wave += 0.5
sample += 0.5
wave0 = wave.normalize()
sample0 = sample.normalize()

# Initialize detector
detector = wave.ones_like()
detector.position = 400

# Initialize operators
broadcast = Broadcast()
modulate = Modulate()
propagate = Propagate()
detect = Detect()

# Distances
wave_positions = [0, 0, 0, 0]
sample_positions = [0, 100, 200, 300]
detector_position = 400
source_to_sample = np.subtract(sample_positions, wave_positions)
sample_to_detector = np.subtract(detector_position, sample_positions)

# Forward pipeline
sample0 = broadcast.apply(sample0, position=sample_positions)
wave0 = broadcast.apply(wave0, position=wave_positions)
wave = propagate.apply(wave0, distance=source_to_sample)
wave = modulate.apply(wave, sample0)
wave = propagate.apply(wave, distance=sample_to_detector)
wave = modulate.apply(wave, detector)
intensity = detect.apply(wave)

# Plot results
plt.figure(figsize=(20, 5))
num_positions = len(sample_positions)
for i in range(num_positions):
    plt.subplot(1, num_positions, i+1)
    plt.imshow(wave.amplitude[i], cmap='gray')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Backward pipeline
wave = detect.adjoint(intensity)
wave = modulate.adjoint(wave, detector)
wave = propagate.adjoint(wave, distance=sample_to_detector)
sample = modulate.adjoint(wave, wave0) 
sample = broadcast.adjoint(sample, position=sample_positions)
wave = modulate.adjoint(wave, sample0)
wave = propagate.adjoint(wave, distance=source_to_sample)
wave = broadcast.adjoint(wave, position=wave_positions)

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Wave')
plt.imshow(np.abs(wave.amplitude[0]), cmap='gray')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title('Sample')
plt.imshow(sample.amplitude[0], cmap='gray')
plt.colorbar()
plt.tight_layout()
plt.show()
