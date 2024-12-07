# Standard imports
import torch
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho.core.component import Wave
from coho.core.pipeline import MultiDistanceHolography
from coho.core.optimization import GradientDescent, LeastSquares

# Load test images and convert to tensors
lena = torch.from_numpy(np.load('./coho/resources/images/lena.npy')) + 256.
cameraman = torch.from_numpy(np.load('./coho/resources/images/cameraman.npy')) + 256.

# Initialize waves
reference = Wave(lena, energy=10.0, spacing=1e-4, position=0.0).normalize()
sample = Wave(cameraman, energy=10.0, spacing=1e-4, position=0.0).normalize()
detector = Wave(torch.ones_like(lena), energy=10.0, spacing=1e-4, position=800.0).normalize()

# Create sample wave at multiple positions
sample_positions = torch.arange(100., 800., 200, dtype=torch.float64)  # Multiple sample positions

print(repr(reference))
print(repr(sample))
print(repr(detector))
print(repr(sample_positions))

# Create and run pipeline
pipeline = MultiDistanceHolography(reference, detector, sample_positions)

# Forward pipeline
measurements = pipeline.apply(sample)

# Plot measurements
plt.figure(figsize=(20, 5))
num_positions = len(sample_positions)
for i in range(num_positions):
    plt.subplot(1, num_positions, i+1)
    plt.imshow(measurements[i].cpu().numpy(), cmap='gray')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Adjoint pipeline
sample = pipeline.adjoint(measurements)

# Plot adjoint
plt.figure(figsize=(20, 5))
plt.imshow(sample.amplitude.cpu().numpy(), cmap='gray')
plt.tight_layout()
plt.show()

# # Create objective with cost monitoring
# objective = LeastSquares(target=measurements, operator=pipeline)

# # Reconstruct sample with more iterations
# initial_guess = sample.zeros_like()
# solver = GradientDescent(
#     cost=objective,
#     step_size=0.9,
#     iterations=1,  
#     initial_guess=initial_guess
# )

# # Run optimization and monitor progress
# reconstruction = solver.solve()

# # Plot results
# plt.figure(figsize=(12, 4))

# # Plot 1: Convergence
# plt.subplot(131)
# plt.semilogy(objective.cost_history, 'b-')
# plt.grid(True)
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Convergence History')

# # Plot 2: Reconstruction Amplitude
# plt.subplot(132)
# plt.imshow(reconstruction.amplitude[0].cpu().numpy(), cmap='gray')
# plt.title('Reconstructed Amplitude')
# plt.clim([0, 1])
# plt.colorbar()

# # Plot 3: Reconstruction Phase
# plt.subplot(133)
# plt.imshow(reconstruction.phase[0].cpu().numpy(), cmap='gray')
# plt.title('Reconstructed Phase')
# plt.clim([0, 1])
# plt.colorbar()

# plt.tight_layout()
# plt.show()