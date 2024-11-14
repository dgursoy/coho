# mymain.py - Example script to run a simulation from a configuration file

import coho
import matplotlib.pyplot as plt

# Build and initialize the Simulation instance from configuration
simulation = coho.build_simulation_from_config("myconfig.yaml")

# Run and get results
simulation.run()
results = simulation.get_results()

# Plot results (first and only image)
plt.figure(figsize=(8, 6))
plt.imshow(results[0], cmap='gray')
plt.title("Simulated Wavefront Intensity at Detector")
plt.colorbar(label="Intensity")
plt.show()