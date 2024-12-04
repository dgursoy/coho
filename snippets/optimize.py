# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho import Wave, Pipeline, Propagate, Modulate, Detect, Broadcast, GradientDescent, LeastSquares

# Load test images
lena = np.load('./coho/resources/images/lena.npy') / 255.
cameraman = np.load('./coho/resources/images/cameraman.npy') / 255.
ship = np.load('./coho/resources/images/ship.npy') / 255.
barbara = np.load('./coho/resources/images/barbara.npy') / 255.

# Initialize waves
sample = Wave(cameraman * np.exp(ship * 1j), energy=10.0, spacing=1e-4, position=0.0).normalize()
wave = Wave(lena * np.exp(barbara * 1j), energy=10.0, spacing=1e-4, position=0.0).normalize() 


# # Plot results
# plt.figure(figsize=(8, 4))

# # Plot 2: Reconstruction
# plt.subplot(132)
# plt.imshow(sample.amplitude, cmap='gray')
# plt.title('Reconstructed Sample')
# plt.colorbar()
# # Plot 2: Reconstruction
# plt.subplot(133)
# plt.imshow(sample.phase, cmap='gray')
# plt.title('Reconstructed Sample')
# plt.colorbar()

# plt.tight_layout()
# plt.show()

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

# Prepare wave
broadcast = Broadcast('position')
wave0 = broadcast.apply(wave0, values=wave_positions)
wave = Propagate().apply(wave0, distance=source_to_sample)

# Define pipeline
pipeline = Pipeline([
    (Broadcast('position'), {'values': sample_positions}),
    (Modulate(), {'modulator': wave}),
    (Propagate(), {'distance': sample_to_detector}),
    (Modulate(), {'modulator': detector}),
    (Detect(), {})
])

# Forward: Apply pipeline
measurements = pipeline.apply(sample0)

# Create objective with cost monitoring
objective = LeastSquares(target=measurements, operator=pipeline)

# Reconstruct sample with more iterations
initial_guess = sample0.zeros_like()
solver = GradientDescent(
    objective=objective,
    step_size=0.9,
    iterations=110,  
    initial_guess=initial_guess
)

# Run optimization and monitor progress
reconstruction = solver.solve()

# Plot results
plt.figure(figsize=(12, 4))

# Plot 1: Convergence
plt.subplot(131)
plt.semilogy(objective.cost_history, 'b-')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Convergence History')

# Plot 2: Reconstruction
plt.subplot(132)
plt.imshow(reconstruction.amplitude[0], cmap='gray')
plt.title('Reconstructed Sample')
plt.colorbar()
# Plot 2: Reconstruction
plt.subplot(133)
plt.imshow(reconstruction.phase[0], cmap='gray')
plt.title('Reconstructed Sample')
plt.colorbar()

plt.tight_layout()
plt.show()