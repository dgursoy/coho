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
    SparseCovarianceLocalization,
    GaussianNoise,
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
    for j in range(sample_size):
        sample = gm.sample()
        if j == 0:
            ax = axes[1, 0]
            ax = fig.add_subplot(345)
            im = ax.imshow(sample.amplitude[0], cmap='gray')
            ax.set_title(f'Sample {j+1}/{sample_size} (Amplitude)')
            plt.colorbar(im, ax=ax)
            ax = axes[1, 1]
            im = ax.imshow(sample.phase[0], cmap='gray')
            ax.set_title(f'Sample {j+1}/{sample_size} (phase)')
            plt.colorbar(im, ax=ax)
        if j == 1:
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
    im = ax.imshow(err.amplitude[0], cmap='gray')
    ax.set_title('Error (Amplitude)')
    plt.colorbar(im, ax=ax)
    # Phase
    ax = axes[2, 3]
    im = ax.imshow(err.phase[0], cmap='gray')
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
    # Load test images and normalize
    lena = np.load('./coho/resources/images/lena.npy') / 255.
    cameraman = np.load('./coho/resources/images/cameraman.npy') / 255.
    ship = np.load('./coho/resources/images/ship.npy') / 255.
    barbara = np.load('./coho/resources/images/barbara.npy') / 255.

    # Settings
    random_seed = 1011
    sample_size = 50
    plot_format = "png"
    cropsize = 512  # Make less than 512 (full size) Just to make things faster...

    # Create a random number generator (rng)
    rng = np.random.default_rng(random_seed)

    # Initialize sample (complex valued) and extract wave form
    sample = Wave(
        (cameraman * np.exp(ship * 1j))[:cropsize, :cropsize],
        energy=10.0,
        spacing=1e-4,
        position=0.0
    ).normalize()
    sample += 0.5

    #
    ########################################################
    ##    Create multiple Gaussian distributions with     ##
    ##    the same mean and with different covariances    ##
    ########################################################

    ####################
    # 1- Diagonal covariance with variances of real and imaginary parts beign equal
    ####################
    # Create covariance operator with random variances [0, 1]
    nx, ny = sample.shape[1: ]
    covariance = DiagonalCovariance(
        waveform_shape=(nx, ny),
        data=rng.random(nx* ny) + 1e-5,
    )

    # Create Gaussian Model and create plots
    gm = GaussianNoise(
        mean=sample,
        covariance=covariance,
        random_seed=random_seed,
    )
    create_plots_from_Gaussian(
        gm,
        sample_size=sample_size,
        saveto=f"NoisePlots_DiagonalCov_EqualRealImag_SampleSize_{sample_size}.{plot_format}"
    )
    ####################


    ####################
    # Tri-diagonal covariance (each pixel correlate with next two) with ...
    ####################
    nx, ny = sample.shape[1: ]
    stdevs = 0.1  # rng.random() + 1e-5 # Random value of the variance

    # Variances random variances [0, 1]
    covariance = SparseCovarianceLocalization(
        step_size=2,
        waveform_shape=(nx, ny),
        localization_function=lambda d: np.exp(-d),
    )
    covariance.covariance_array.data *= stdevs

    # Now, split variance to real and imaginar parts
    coord = covariance.coord
    data = covariance.covariance_array.data
    lt = data[coord[0]<coord[1]]
    ut = data[coord[0]>coord[1]]
    lt = 0.5 * lt + 0.5j * lt
    ut = 0.5 * ut - 0.5j * ut
    covariance.covariance_array.data[coord[0]==coord[1]] += 0.1
    covariance.covariance_array.data[coord[0]<coord[1]] = lt
    covariance.covariance_array.data[coord[0]>coord[1]] = ut


    # Create Gaussian Model and create plots
    gm = GaussianNoise(
        mean=sample,
        covariance=covariance,
        random_seed=random_seed,
    )
    create_plots_from_Gaussian(
        gm,
        sample_size=sample_size,
        saveto=f"NoisePlots_TriDiagonalCov_EqualRealImag_SampleSize_{sample_size}.{plot_format}"
    )
    ####################

    # Create complex wave with random values


