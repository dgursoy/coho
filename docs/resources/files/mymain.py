# mymain.py - Example script to run a simulation from a configuration file

from coho import load_config, Holography
import matplotlib.pyplot as plt

# Load configuration from file
config = load_config("myconfig.yaml")

# Create and run simulation
forward = Holography(config)
image = forward.run()

# Plot detector image
plt.imshow(image[0], cmap='gray')
plt.title("Image captured by detector")
plt.colorbar()
plt.show()