# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho import Wave, Pipeline, Propagate, Modulate, Detect, Broadcast, GradientDescent, LeastSquares

# Load test images
lena = np.load('./coho/resources/images/lena.npy')
cameraman = np.load('./coho/resources/images/cameraman.npy')

# Initialize waves
sample = Wave(lena, energy=10.0, spacing=1e-4, position=0.0).normalize()
wave = Wave(cameraman, energy=10.0, spacing=1e-4, position=0.0).normalize() 
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
broadcast = Broadcast()
wave0 = broadcast.apply(wave0, position=wave_positions)
wave = Propagate().apply(wave0, distance=source_to_sample)

# Define pipeline
pipeline = Pipeline([
    (Broadcast(), {'position': sample_positions}),
    (Modulate(), {'modulator': wave}),
    (Propagate(), {'distance': sample_to_detector}),
    (Modulate(), {'modulator': detector}),
    (Detect(), {})
])

# Forward: Apply pipeline
initial_guess = sample0.zeros_like()
measurements = pipeline.apply(sample0)

# Create objective with cost monitoring
objective = LeastSquares(target=measurements, operator=pipeline)

# Reconstruct sample with more iterations
solver = GradientDescent(
    objective=objective,
    step_size=0.9,
    iterations=50,  
    initial_guess=initial_guess
)

# Run optimization and monitor progress
reconstruction = solver.solve()

# Plot results
plt.figure(figsize=(12, 4))

# Plot 1: Convergence
plt.subplot(121)
plt.semilogy(objective.cost_history, 'b-')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Convergence History')

# Plot 2: Reconstruction
plt.subplot(122)
plt.imshow(np.abs(reconstruction.amplitude[0]), cmap='gray')
plt.title('Reconstructed Sample')
plt.colorbar()

plt.tight_layout()
plt.show()