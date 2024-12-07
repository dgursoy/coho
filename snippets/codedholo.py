# Standard imports
import torch
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho.core.component import Wave
from coho.core.pipeline import CodedHolography
from coho.core.optimization import LeastSquares, GradientDescent

# Load test images and convert to tensors
baboon = torch.from_numpy(np.load('./coho/resources/images/baboon.npy')) + 256.
cameraman = torch.from_numpy(np.load('./coho/resources/images/cameraman.npy')) + 256.
barbara = torch.from_numpy(np.load('./coho/resources/images/barbara.npy')) + 256.

# Initialize waves
reference = Wave(baboon, energy=10.0, spacing=1e-4, position=0.0).normalize()
code = Wave(barbara, energy=10.0, spacing=1e-4, position=0.0).normalize()
sample = Wave(cameraman, energy=10.0, spacing=1e-4, position=400.0).normalize()
detector = Wave(torch.ones_like(baboon), energy=10.0, spacing=1e-4, position=800.0).normalize()

# Initialize initial guess
initial_guess = sample.zeros_like()

# Create sample wave at multiple positions
x = torch.arange(-16, 17, 16, dtype=torch.float64)
y = torch.arange(-16, 17, 16, dtype=torch.float64)
positions_x, positions_y = torch.meshgrid(x, y, indexing='ij')
positions_x = positions_x.flatten()
positions_y = positions_y.flatten()
print(repr(positions_x))
print(repr(positions_y))

# Create and run pipeline
pipeline = CodedHolography(reference, detector, sample, code, positions_x, positions_y)
intensity = pipeline.apply(sample)

# Plot intensity
plt.figure(figsize=(20, 5))
num_positions = len(positions_x) 
for i in range(num_positions):
    plt.subplot(1, num_positions, i+1)
    plt.imshow(intensity[i].cpu().numpy(), cmap='gray')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Adjoint
sample = pipeline.adjoint(intensity)

# Plot adjoint
plt.figure(figsize=(20, 5))
plt.imshow(sample.amplitude.cpu().numpy(), cmap='gray')
plt.tight_layout()
plt.show()

# # Create objective with cost monitoring
# objective = LeastSquares(target=intensity, operator=pipeline)

# # Reconstruct sample with more iterations
# solver = GradientDescent(
#     cost=objective,
#     step_size=0.9,
#     iterations=50,  
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
# plt.colorbar()

# # Plot 3: Reconstruction Phase
# plt.subplot(133)
# plt.imshow(reconstruction.phase[0].cpu().numpy(), cmap='gray')
# plt.title('Reconstructed Phase')
# plt.colorbar()

# plt.tight_layout()
# plt.show()