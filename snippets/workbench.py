# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho.config import load_config

# Components
from coho.core import GaussianWavefront, CodedOptic, LenaSample, StandardDetector
from coho.core.experiment import HolographyScan

# Operators
from coho.core.operator import FresnelPropagate, Interact, Detect, Pipeline, Rotate, Translate, ScanParameter, FourierPropagate

# ######################
# # Sequential operators
# ######################

# # Load configuration
# config = load_config('snippets/config.yaml')

# WavefrontConfig = config.components.wavefront
# OpticConfig = config.components.optic
# SampleConfig = config.components.sample
# DetectorConfig = config.components.detector

# # Initialize components with configurations
# wavefront = GaussianWavefront(WavefrontConfig)
# optic = CodedOptic(OpticConfig)
# sample = LenaSample(SampleConfig)
# detector = StandardDetector(DetectorConfig)

# # Create operator instances once
# fresnel = FresnelPropagate()
# interact = Interact()
# rotate = Rotate()
# translate = Translate()
# detect = Detect()  
# fourier = FourierPropagate()

# # Create a raster scan pattern
# x = np.linspace(-50, 50, 2)  # 3 points in x
# y = np.linspace(-50, 50, 2)    # 2 points in y
# xx, yy = np.meshgrid(x, y)
# translations = np.column_stack((xx.flatten(), yy.flatten()))  # Shape: (6, 2)

# # Create a rotation pattern
# rotations = np.linspace(0.0, 60.0, 10)
# #
# # optic = rotate.apply(optic, parameter=ScanParameter(rotation=rotations))
# optic = translate.apply(optic, parameter=ScanParameter(translation=translations))
# wavefront = interact.apply(wavefront, component=optic)
# wavefront = fresnel.apply(wavefront, distance=1.0)
# wavefront = interact.apply(wavefront, component=sample)
# wavefront = fresnel.apply(wavefront, distance=50.0)
# intensity = detect.apply(wavefront, component=detector)

# plt.figure(figsize=(20, 5))
# num_components = wavefront.complexform.shape[0]
# for m in range(num_components):
#     plt.subplot(1, num_components, m + 1)
#     plt.imshow(intensity[m])
#     # plt.colorbar() 
# plt.tight_layout()
# plt.show()

# wavefront = detect.adjoint(intensity, component=detector)
# wavefront = fresnel.adjoint(wavefront, distance=50.0)
# wavefront = interact.adjoint(wavefront, component=sample)
# wavefront = fresnel.adjoint(wavefront, distance=1.0)
# wavefront = interact.adjoint(wavefront, component=optic)
# # optic = rotate.adjoint(optic, parameter=ScanParameter(rotation=rotations))
# optic = translate.adjoint(optic, parameter=ScanParameter(translation=translations))


# plt.figure(figsize=(20, 5))
# num_components = wavefront.complexform.shape[0]
# for m in range(num_components):
#     plt.subplot(1, num_components, m + 1)
#     plt.imshow(np.abs(wavefront.complexform[m]))
#     # plt.colorbar() 
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(wavefront.complexform[0]))
# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(optic.complexform[0]))
# plt.tight_layout()
# plt.show()

# # ##################
# # # Nested operators
# # ##################

# # Load configuration
# config = load_config('snippets/config.yaml')

# WavefrontConfig = config.components.wavefront
# OpticConfig = config.components.optic
# SampleConfig = config.components.sample
# DetectorConfig = config.components.detector

# # # Create components
# wavefront = GaussianWavefront(WavefrontConfig)
# optic = CodedOptic(OpticConfig)
# sample = LenaSample(SampleConfig)
# detector = StandardDetector(DetectorConfig)

# # Set distances for example
# distances = [1.0, 1.0, 50.0]

# # Create operator instances once
# fresnel = FresnelPropagate()
# interact = Interact()
# detect = Detect()

# # Initialize operators with reused instances
# holography = Pipeline([
#     (fresnel, {'distance': distances[0]}),
#     (interact, {'component': optic}),
#     (fresnel, {'distance': distances[1]}),
#     (interact, {'component': sample}),
#     (fresnel, {'distance': distances[2]}),
#     (detect, {'component': detector})
# ])

# # Forward propagation
# intensity = holography.apply(wavefront)

# # Backward propagation
# wavefront = holography.adjoint(intensity)

# # Plot wavefront
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(intensity[0])
# plt.title('Detected intensity')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(wavefront.complexform[0]))
# plt.title('Backpropagated wavefront')
# plt.colorbar()
# plt.show()

# #########################
# # Scanning experiment
# #########################

# Load configuration
config = load_config('snippets/config.yaml')

WavefrontConfig = config.components.wavefront
OpticConfig = config.components.optic
SampleConfig = config.components.sample
DetectorConfig = config.components.detector

# Create components
wavefront = GaussianWavefront(WavefrontConfig)
optic = CodedOptic(OpticConfig)
sample = LenaSample(SampleConfig)
detector = StandardDetector(DetectorConfig)

# Create a dictionary of components
components = {
    'wavefront': wavefront,
    'optic': optic,
    'sample': sample,
    'detector': detector
}

# Initialize HolographyScan
holography = HolographyScan()

# Run the holography experiment
intensity = holography.run(components)

# Plot the results
plt.figure(figsize=(10, 5))
plt.imshow(intensity[0])
plt.title('Detected intensity')
plt.colorbar()
plt.show()
