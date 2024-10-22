import fire
import coho
import numpy as np
import xraylib


def main(config_file):
    '''Test sctript for the coho module'''
    # Load configuration
    config = coho.config(config_file)

    # Load sample properties
    material = 'Au'
    density = 19.3 # g/cm^3 for gold
    refractive_index = 1 - xraylib.Refractive_Index(material, config['energy'], density)
    
    # Load sample structure
    sample = np.load('images/lena.npy')
    sample = refractive_index * sample / sample.max()
    
    # Calculate wave at sample
    thickness = config['thickness'] * 1e-4 # Convert thickness to cm
    wavelength = coho.wavelength(config['energy'])
    wave_at_sample = np.exp(1j * 2 * np.pi * sample * thickness / wavelength)

    # Propagate the wave to the detector plane
    wave_at_detector = coho.fresnel(
        wave_at_sample,
        config['distance'],
        config['energy'],
        config['pixelsize'],
        adjoint=False
    )
    # Calculate intensity at the detector plane
    incident_intensity = config['incident_intensity']
    intensity_at_detector = incident_intensity * coho.intensity(wave_at_detector)
    
    # Simulate detector reading with Poisson noise
    detector_reading = coho.add_poisson_noise(intensity_at_detector)
    
    # Backpropagate the wave to the sample plane
    wave_at_sample = coho.fresnel(
        wave_at_detector,
        config['distance'],
        config['energy'],
        config['pixelsize'],
        adjoint=True
    )

    # Visualize the results
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Original sample (real part)
    ax = axes[0, 0]
    im = ax.imshow(np.real(sample), cmap='gray')
    ax.set_title('Original (real)')
    fig.colorbar(im, ax=ax)

    # Original sample (imaginary part)
    ax = axes[0, 1]
    im = ax.imshow(-np.imag(sample), cmap='gray')
    ax.set_title('Original (imag)')
    fig.colorbar(im, ax=ax)

    # Propagated wave (magnitude)
    ax = axes[1, 0]
    im = ax.imshow(intensity_at_detector, cmap='gray')
    ax.set_title('Internsity at detector')
    fig.colorbar(im, ax=ax)

    # Back-propagated wave (magnitude)
    ax = axes[1, 1]
    im = ax.imshow(detector_reading, cmap='gray')
    ax.set_title('Detector reading')
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('test.png')


if __name__ == '__main__':
    fire.Fire(main)