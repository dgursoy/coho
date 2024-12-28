"""
Driver to test noise models (mainly Gaussian).
"""
# Standard imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from coho import (
    Wave,
    DiagonalCovariance,
    SparseCovariance,
    SparseCovarianceLocalization,
    GaussianNoise,
    ComplexGaussianNoise,
    real_to_complex_covariances,
)


def create_plots_from_Gaussian(gm, sample_size=30, saveto=None, return_fig=False, ):
    """Given a Gaussian model, sample it, and create plots of things"""

    ## Plot

    # Create figure
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))

    ## Ground truth) Mean
    # Amplitude
    ax = axes[0, 0]
    im = ax.imshow(gm.mean.amplitude[0], cmap='gray')
    ax.set_title('Mean (Amplitude)')
    plt.colorbar(im, ax=ax)
    # Phase
    ax = axes[0, 1]
    im = ax.imshow(gm.mean.phase[0], cmap='gray')
    ax.set_title('Mean (Phase)')
    plt.colorbar(im, ax=ax)

    ## Covariance snapshot (first 200 entries) (both real and imaginary parts)
    cov_plot_size = 50
    cov = gm.covariance.covariance_array[0: cov_plot_size, 0: cov_plot_size].toarray()
    # Real
    cov_real = cov.real
    cov_real[cov_real==0] = np.nan
    ax = axes[0, 2]
    mat = ax.matshow(cov_real, )
    ax.set_title('Covariance (real)')
    plt.colorbar(mat, ax=ax)

    # Imaginary
    cov_imag = cov.imag
    cov_imag[cov_imag==0] = np.nan
    ax = axes[0, 3]
    mat = ax.matshow(cov_imag, )
    ax.set_title('Covariance (Imaginary)')
    plt.colorbar(mat, ax=ax)

    ## Samples: sample (sample_size), plot the first three, and calculat the average
    average = None
    # print("\n******\n", gm.mean.form, "\n******\n")
    for j in range(sample_size):
        sample = gm.sample()
        # print("Sample {j+1}/{sample_size}: ", sample.form)
        if j == 0:
            ax = axes[1, 0]
            im = ax.imshow(sample.amplitude[0], cmap='gray')
            ax.set_title(f'Sample {j+1}/{sample_size} (Amplitude)')
            plt.colorbar(im, ax=ax)
            ax = axes[1, 1]
            im = ax.imshow(sample.phase[0], cmap='gray')
            ax.set_title(f'Sample {j+1}/{sample_size} (phase)')
            plt.colorbar(im, ax=ax)
        elif j == 1:
            ax = axes[1, 2]
            im = ax.imshow(sample.amplitude[0], cmap='gray')
            ax.set_title(f'Sample {j+1}/{sample_size} (Amplitude)')
            plt.colorbar(im, ax=ax)
            ax = axes[1, 3]
            im = ax.imshow(sample.phase[0], cmap='gray')
            ax.set_title(f'Sample {j+1}/{sample_size} (phase)')
            plt.colorbar(im, ax=ax)

        # Add to mean
        if j == 0:
            average = sample
        else:
            average += sample

    average /= float(sample_size)

    # Plot Average (phase and amplitude)
    # Amplitude
    ax = axes[2, 0]
    im = ax.imshow(average.amplitude[0], cmap='gray')
    ax.set_title('Sample Average (Amplitude)')
    plt.colorbar(im, ax=ax)
    # Phase
    ax = axes[2, 1]
    im = ax.imshow(average.phase[0], cmap='gray')
    ax.set_title('Sample Average (Phase)')
    plt.colorbar(im, ax=ax)

    # Plot Errors (mean-average)
    err = gm.mean - average
    # Amplitude
    ax = axes[2, 2]
    # im = ax.imshow(err.amplitude[0], cmap='gray')
    im = ax.imshow(gm.mean.amplitude[0]-average.amplitude[0], cmap='gray')
    ax.set_title('Error (Amplitude)')
    plt.colorbar(im, ax=ax)
    # Phase
    ax = axes[2, 3]
    # im = ax.imshow(err.phase[0], cmap='gray')
    im = ax.imshow(gm.mean.phase[0]-average.phase[0], cmap='gray')
    ax.set_title('Error (Phase)')
    plt.colorbar(im, ax=ax)

    # Writing figure
    if saveto is not None:
        dirname = os.path.dirname(os.path.abspath(saveto))
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        fig.savefig(saveto, bbox_inches=None)
        print(f"Gaussian Model Plots saved to '{saveto}'")

    # Cleanup or return figure
    if return_fig:
        return fig
    else:
        plt.close(fig)


if __name__ == "__main__":
    ########################################################################
    ## TEST LOGIC
    #############
    # 1- Model a complex-valued Gaussian distribution using
    #    (modeling complex-valued wave) using full variance/relation structure
    # 2- ......
    #
    #
    ########################################################################

    # Load test images and normalize
    lena = np.load('./coho/resources/images/lena.npy') / 255.
    cameraman = np.load('./coho/resources/images/cameraman.npy') / 255.
    ship = np.load('./coho/resources/images/ship.npy') / 255.
    barbara = np.load('./coho/resources/images/barbara.npy') / 255.

    # Settings
    random_seed = 1011
    sample_size = 10
    sample_shift = 0.5
    noise_level = 0.10      # fraction ([0, 1]) of the magitude of the pixel with highest value
    real_imag_split = 0.8  # this mutlplies reference image to get real component; remaining from 1 gives imaginary
    neighborhood_size = 1  # NOTE: Make more to increase neighborhood size
    plot_format = "png"
    cropsize = 512  # Make less than 512 (full size) Just to make things faster...

    # Create a random number generator (rng)
    rng = np.random.default_rng(random_seed)


    ## Choose true image
    reference_image = cameraman

    ## Create reference sample (complex-valued) from reference image
    # Random signs to have positive and negative signs in the complex signal
    if True:
        reference_sample = (cameraman * np.exp(ship * 0.5j))[:cropsize, :cropsize]  # original composite
    else:
        random_signs = rng.choice((1, -1), size=reference_image.size).reshape(reference_image.shape)
        reference_sample = reference_image * real_imag_split + 1j * (1-real_imag_split) * random_signs * reference_image
        reference_sample = reference_sample[: cropsize, :cropsize]  # crop image (if needed)


    # Noise levels (standarde deviations) & correlations
    noise_stdev_real = noise_level * np.abs(reference_sample.real).max() + 0.1
    noise_stdev_imag = noise_level * np.abs(reference_sample.imag).max() + 0.1
    real_imag_correlation = np.corrcoef(reference_sample.real.ravel(), reference_sample.imag.ravel())[0, 1]

    # # Variances and covariances
    noise_variance_real  = noise_stdev_real**2
    noise_variance_imag  = noise_stdev_imag**2
    real_imag_covariance = real_imag_correlation * noise_stdev_real * noise_stdev_imag

    # Initialize sample (complex valued) and extract wave form
    sample = Wave(
        reference_sample,
        energy=10.0,
        spacing=1e-4,
        position=0.0
    ).normalize()
    sample += sample_shift

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
        C_RR=covariance_localization*noise_variance_real,
        C_II=covariance_localization*noise_variance_imag,
        C_RI=covariance_localization*real_imag_covariance,
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
    mean = sample

    # Cleanup
    del covariance_localization
    ####################

    #
    ########################################################
    ##    Create multiple Gaussian distributions with     ##
    ##    the same mean and with different covariances    ##
    ########################################################

    ##
    # Multiple versions of the Gaussian Noise Model
    ##

    # 1- Gaussian Model (Diagonal Covariance with No Pseudo covariances)
    # -------------------------------------------------------------------
    gm = GaussianNoise(
        mean=mean,
        covariance=DiagonalCovariance(
            waveform_shape=sample.shape[-2: ],
            data=Cov.covariance_array.diagonal(),
        ),
        random_seed=random_seed,
    )
    create_plots_from_Gaussian(
        gm,
        sample_size=sample_size,
        saveto=f"_PLOTS/GaussianNoise_NoisePlots_DiagonalCov_OnlyCovariance_SampleSize_{sample_size}.{plot_format}"
    )

    # 2- Gaussian Model with correlations (Ignore pseudo covariance) and create plots
    # -------------------------------------------------------------------
    gm = GaussianNoise(
        mean=mean,
        covariance=Cov,
        random_seed=random_seed,
    )
    create_plots_from_Gaussian(
        gm,
        sample_size=sample_size,
        saveto=f"_PLOTS/GaussianNoise_NoisePlots_TriDiagonalCov_OnlyCovariance_Neighborhood_{neighborhood_size}_SampleSize_{sample_size}.{plot_format}"
    )

    # 3- Gaussian Model with correlations (Ignore pseudo covariance) and create plots
    # -------------------------------------------------------------------
    gm = ComplexGaussianNoise(
        mean=mean,
        covariance=Cov,
        pseudo_covariance=PCov,
        random_seed=random_seed,
    )
    create_plots_from_Gaussian(
        gm,
        sample_size=sample_size,
        saveto=f"_PLOTS/ComplexGaussianNoise_NoisePlots_TriDiagonalCov_TriDiagonalPseudoCov_Neighborhood_{neighborhood_size}_SampleSize_{sample_size}.{plot_format}"
    )

    # Create Gaussian Model and create plots
    gm = ComplexGaussianNoise(
        mean=mean,
        covariance=DiagonalCovariance(
            waveform_shape=sample.shape[-2: ],
            data=Cov.covariance_array.diagonal(),
        ),
        pseudo_covariance=DiagonalCovariance(
            waveform_shape=sample.shape[-2: ],
            data=PCov.covariance_array.diagonal(),
        ),
        random_seed=random_seed,
    )
    create_plots_from_Gaussian(
        gm,
        sample_size=sample_size,
        saveto=f"_PLOTS/ComplexGaussianNoise_NoisePlots_DiagonalCov_DiagonalPseudoCov_SampleSize_{sample_size}.{plot_format}"
    )


