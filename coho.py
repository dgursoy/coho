"""
coho.py

This module contains core functions for the Coho project.
"""

# Import necessary modules
import numpy as np
import yaml


# Set up random number generator
from numpy.random import RandomState
prng = RandomState(1234567890)


# Constants
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


def config(config_file):
    stream = open(config_file, 'r')
    pars = yaml.safe_load(stream)
    return pars


# Functions
def wavelength(energy):
    """
    Calculate the wavelength from energy.

    Parameters:
    energy (float): The energy value in keV.

    Returns:
    float: The calculated wavelength in cm.
    """
    '''Calculate wavelength from energy'''
    return 2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy 


def fresnel(wave, distance, energy, pixelsize, adjoint=False):
    """
    Propagate wave with Fresnel propagator at a given distance.

    Parameters:
    wave (ndarray): The input plane wave to be propagated.
    distance (float): The propagation distance in cm.
    energy (float): The energy of the wave in keV.
    pixelsize (float): The size of the voxel in microns.
    adjoint (bool, optional): If True, propagates the wave in the reverse direction. Default is False.

    Returns:
    ndarray: The propagated plane wave.
    """
    if adjoint is True:
        distance *= -1
    kernel = fresnel_kernel(wave, distance, energy, pixelsize)
    return fourier_convolution(wave, kernel)


def fourier_convolution(wave, kernel):
    """
    Perform convolution on a given plane wave with a Fresnel kernel.

    Parameters:
    wave (ndarray): The input wave array.
    kernel (ndarray): The Fresnel kernel to be applied.

    Returns:
    ndarray: The resulting wave after applying the Fresnel convolution.
    """
    n = wave.shape[-1]
    wave = np.pad(wave, ((n//2, n//2), (n//2, n//2)), 'symmetric')
    wave = np.fft.ifft2(np.fft.fft2(wave) * kernel)
    return wave[n//2:-n//2, n//2:-n//2]


def fresnel_kernel(wave, distance, energy, pixelsize):
    """
    Calculate the Fresnel diffraction kernel.

    Parameters:
    wave (ndarray): The input plane wave.
    distance (float): The propagation distance in cm.
    energy (float): The energy of the wave in keV.
    pixelsize (float): The size of each voxel in the plane wave in microns.

    Returns:
    ndarray: The Fresnel diffraction kernel.
    """
    '''Calculate Fresnel kernel'''
    n = wave.shape[-1]
    fx = np.fft.fftfreq(2*n, d=pixelsize*1e-4).astype('float32')
    [fx, fy] = np.meshgrid(fx, fx)
    return np.exp(-1j * np.pi * wavelength(energy) * distance * (fx**2 + fy**2))


def intensity(wave):
    """
    Calculate the intensity of a wave.

    The intensity is computed as the square of the absolute value of the wave.

    Parameters:
    wave (numpy.ndarray): The input plane wave.

    Returns:
    numpy.ndarray: The intensity of the wave.
    """
    return np.power(np.abs(wave), 2)

def add_poisson_noise(intensity, normalize=False, level=1000):
    """
    Add noise to a given intensity.

    This function normalizes the input intensity and adds Poisson-distributed noise based on a specified incident intensity level.

    Parameters:
    intensity (ndarray): The input intensity to which noise is to be added.
    normalize (bool): Whether to normalize the input intensity before adding noise. Default is False.
    level (float): The incident intensity to be used for generating noise. Default is 1000.

    Returns:
    ndarray: The intensity with added noise.
    """
    if normalize:
        intensity = intensity / intensity.max()
        intensity *= level
    return np.random.poisson(intensity)