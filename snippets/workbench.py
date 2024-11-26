# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho.config import load_config

# Components
from coho.core import GaussianWavefront, CodedOptic, LenaSample, StandardDetector

# Operators
from coho.core.operator import FresnelPropagate, Interact, Detect, Pipeline

# Load configuration
config = load_config('snippets/config.yaml')

# Create components
wavefront = GaussianWavefront(config.components.root['Wavefront'])
aperture = CodedOptic(config.components.root['Aperture'])
sample = LenaSample(config.components.root['Sample'])
detector = StandardDetector(config.components.root['Detector'])

# Initialize operators
holography = Pipeline([
    (FresnelPropagate(), {'distance': 1.0}),
    (Interact(), {'component': aperture}),
    (FresnelPropagate(), {'distance': 1.0}),
    (Interact(), {'component': sample}),
    (FresnelPropagate(), {'distance': 100.0}),
    (Detect(), {'component': detector})
])

# Forward propagation
intensity = holography.apply(wavefront)

# Plot wavefront
plt.imshow(intensity)
plt.colorbar()
plt.show()

# Backward propagation
wavefront = holography.adjoint(intensity)

# Plot wavefront
plt.imshow(np.abs(wavefront.phasor))
plt.colorbar()
plt.show()