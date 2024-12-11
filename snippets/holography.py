# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho.core.component import Wave
from coho.core.pipeline import MultiDistanceHolography
from coho.core.optimization import GradientDescent, LeastSquares


# Load test images
lena = np.load('./coho/resources/images/lena.npy') + 256.
cameraman = np.load('./coho/resources/images/cameraman.npy') + 256.

# Initialize waves
reference = Wave(lena, energy=10.0, spacing=1e-4, position=0.0).normalize()
sample = Wave(cameraman, energy=10.0, spacing=1e-4, position=0.0).normalize()
detector = Wave(np.ones_like(lena), energy=10.0, spacing=1e-4, position=400.0).normalize()

# Create sample wave at multiple positions
sample_positions = np.arange(100, 300, 9)  # Multiple sample positions

# Create and run pipeline
pipeline = MultiDistanceHolography(reference, detector, sample_positions)
measurements = pipeline.apply(sample)

# # Plot measurements
# plt.figure(figsize=(20, 5))
# num_positions = len(sample_positions)
# for i in range(num_positions):
#     plt.subplot(1, num_positions, i+1)
#     plt.imshow(measurements[i], cmap='gray')
#     plt.colorbar()
# plt.tight_layout()
# plt.show()

# Adjoint
sample = pipeline.adjoint(measurements)

# # Plot adjoint
# plt.figure(figsize=(20, 5))
# plt.imshow(sample.amplitude, cmap='gray')
# plt.tight_layout()
# plt.show()

print (reference.position)
print (sample.position)
print (detector.position)

# Create objective with cost monitoring
objective = LeastSquares(target=measurements, operator=pipeline)

# Reconstruct sample with more iterations
initial_guess = sample.zeros_like()
solver = GradientDescent(
    objective=objective,
    step_size=0.9,
    iterations=1,  
    initial_guess=initial_guess
)

# Run optimization and monitor progress
reconstruction = solver.solve()

# Plot results
plt.figure(figsize=(12, 4))

# # Plot 1: Convergence
# plt.subplot(131)
# plt.semilogy(objective.cost_history, 'b-')
# plt.grid(True)
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Convergence History')

# # Plot 2: Reconstruction
# plt.subplot(132)
# plt.imshow(reconstruction.amplitude[0], cmap='gray')
# plt.title('Reconstructed Sample')
# plt.clim([0, 1])
# plt.colorbar()
# # Plot 2: Reconstruction
# plt.subplot(133)
# plt.imshow(reconstruction.phase[0], cmap='gray')
# plt.title('Reconstructed Sample')
# plt.clim([0, 1])
# plt.colorbar()

# plt.tight_layout()
# plt.show()