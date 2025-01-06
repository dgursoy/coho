# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho import (
    Wave,
    Pipeline,
    Propagate,
    Modulate,
    Detect,
    Broadcast,
    GradientDescent,
    LeastSquares,
    MultiDistanceHolography,
    #
    DiagonalCovariance,
    SparseCovariance,
    SparseCovarianceLocalization,
    GaussianNoise,
    ComplexGaussianNoise,
    real_to_complex_covariances,
    LocalUnscentedKalmanFilter,
)

# Load test images
lena = np.load('./coho/resources/images/lena.npy') / 255.
cameraman = np.load('./coho/resources/images/cameraman.npy') / 255.
ship = np.load('./coho/resources/images/ship.npy') / 255.
barbara = np.load('./coho/resources/images/barbara.npy') / 255.

# Initialize waves
ref_image = cameraman * np.exp(ship * 1j)
sample = Wave(ref_image[:30, :30], energy=10.0, spacing=1e-4, position=0.0).normalize()
ref_wave = lena * np.exp(barbara * 1j)
wave = Wave(ref_wave[:30, :30], energy=10.0, spacing=1e-4, position=0.0).normalize()
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
wave0 = broadcast.apply(wave0, {'position': wave_positions})
wave = Propagate().apply(wave0, distance=source_to_sample)

# Define pipeline
pipeline = Pipeline([
    (Broadcast(), {'values': {'position': sample_positions}}),
    (Modulate(), {'modulator': wave}),
    (Propagate(), {'distance': sample_to_detector}),
    (Modulate(), {'modulator': detector}),
    (Detect(), {})
])

# Forward: Apply pipeline
measurements = pipeline.apply(sample0)
if measurements.ndim == 3:
    measurements = measurements[0, ...]

## Bayesian stuff
# Settings
prior_noise_real = 0.15
prior_noise_imag = 0.15
prior_real_imag_covariance = 0.15
obs_noise = 0.01
neighborhood_size = 1
random_seed = 1011

####################
# Full covariance/relation matrices
# Create Tridiagonal covariances (real-real, imaginary-imaginary, real-imaginary)
####################
# Localization array that defines covariances between pixels and neighbors
covariance_localization = SparseCovarianceLocalization(
    step_size=neighborhood_size,
    waveform_shape=sample.shape[-2: ],
    localization_function=lambda d: np.exp(-d),
).covariance_array

# Now, use the structure above to create covariances (and pseudo-covariances) as needed
Cov, PCov = real_to_complex_covariances(
    C_RR=covariance_localization*prior_noise_real,
    C_II=covariance_localization*prior_noise_imag,
    C_RI=covariance_localization*prior_real_imag_covariance,
)

# Create Covariance matrix (Cov) and pseudo covariance matrix (PCov)
Cov = SparseCovariance(
    waveform_shape=sample.shape[-2: ],
    data=Cov.data,
    coord=Cov.nonzero(),
)
PCov = SparseCovariance(
    waveform_shape=sample.shape[-2: ],
    data=PCov.data,
    coord=PCov.nonzero(),
)

# Finally mean (shift back):
if sample.form.ndim == 3:
    mean = sample.form[0, ...]
else:
    mean = sample

# Cleanup
del covariance_localization
####################

prior = GaussianNoise(
    mean=Wave(np.zeros_like(mean)),
    covariance=Cov,
    random_seed=random_seed,
)

noise_std = np.sqrt(obs_noise) * (measurements.max()+0.01)
observation_noise = GaussianNoise(
    mean=Wave(measurements),
    covariance=DiagonalCovariance(
        waveform_shape=measurements.shape,
        data=np.ones(measurements.size)*noise_std**2,
    ),
    random_seed=random_seed,
)

solver = LocalUnscentedKalmanFilter(
    pom=lambda x: pipeline.apply(x),
    prior=prior,
    observation_noise=observation_noise,
    observation=measurements,
)

posterior = solver.solve()

# Run optimization and monitor progress
reconstruction = posterior.mean

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
