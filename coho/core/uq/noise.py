"""
UQ-type Objects & algorithms.
This module is experimental and will be moved once mature enough

.. note::
    Some of the functionality here is ported from the sister package
    PyOED (https://web.cels.anl.gov/~aattia/pyoed/).
"""

# Standard imports
import time
from numbers import Number
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from typing import Union  # Type, Tuple, Callable,

# Local imports
from .covariance import Covariance
from ..component import Wave

__all__ = [
    'Noise',
    'GaussianNoise',
]

_DEBUG = False

class Noise(ABC):
    """Base class for all Noise models (Guassian, Poisson, etc.)"""

    @abstractmethod
    def generate_noise(self):
        """
        Generate a random noise vector sampled from the underlying distribution
        """
        ...

    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        Generate a random noise vector sampled from the underlying distribution
        """
        ...

    @abstractproperty
    def size(self):
        """
        Dimension of the underlying probability distribution
        """
        ...

    @abstractproperty
    def random_seed(self):
        """
        Registered random seed
        """
        ...

    @random_seed.setter
    def random_seed(self, val):
        """Update the random seed."""
        raise NotImplementedError(
            f"Random seed setter is not implemented in {self.__class__}."
        )

    def pdf(self, *args, **kwargs):
        """
        Evaluate the value of the density function (normalized or upto a *fixed* scaling
        constant) at the passed state/vector.
        """
        raise NotImplementedError(
            f"The PDF evaluation function is not implemented for this model {self.__class__}.\n"
            "This needs to be implemented for each error model individually"
        )

    def pdf_gradient(self, *args, **kwargs):
        """
        Evaluate the gradient of the density function at the passed state/vector.
        """
        raise NotImplementedError(
            f"The PDF evaluation function is not implemented for this model {self.__class__}.\n"
            "This needs to be implemented for each error model individually"
        )

    def log_density(self, *args, **kwargs):
        """
        Evaluate the logarithm of the density function at the passed state `x`.
        """
        raise NotImplementedError(
            f"The PDF evaluation function is not implemented for this model {self.__class__}.\n"
            "This needs to be implemented for each error model individually"
        )

    def log_density_gradient(self, *args, **kwargs):
        """
        Evaluate the gradient of the logarithm of the density function at the passed state.
        """
        raise NotImplementedError(
            f"The PDF evaluation function is not implemented for this model {self.__class__}.\n"
            "This needs to be implemented for each error model individually"
        )


class RandomNumberGenerationMixin:
    """
    Mixin class that provides a unified interface to random number generation
    functionality through Numpy Random Number Generator (RNG).

    This Mixin introduces the following to the object inheriting this Mixin class:

        - **Attributes**:
            * `_RNG`: a random number generator created by calling
            :py:meth:`numpy.random.default_rng()`.

        - **Properties**:
            * `random_number_generator`: a reference to the underlying random number generator `_RNG`.

        - **Methods**:
            * `update_random_number_generator`: reset/Update the underlying random_number generator
                by resetting it's `random_seed`

    .. note::
        When you use this Mixin, make sure you put it as the last class in the inheritance tuple.
    """
    def __init__(self, random_seed=None, ):
        """Initialize the random number generator"""
        self._RNG = np.random.default_rng(random_seed)

    def update_random_number_generator(
        self,
        random_seed,
    ):
        """
        Reset/Update the underlying random_number generator by resetting it's `random_seed`.
        This actually replaces the current random number generator with a new one created from
        the given seed

        :param int|None random_seed: an integer (or None) to be used to reset the random sequence

        :remarks: In the future we may provide an argument to enable changing the generator/algorithm
        """
        self._RNG = np.random.default_rng(random_seed)

    @property
    def random_number_generator(self):
        """Return a handle to the underlying random number generator"""
        return self._RNG


class GaussianNoise(Noise, RandomNumberGenerationMixin):

    def __init__(self, mean: Wave, covariance: Covariance, create_copies: bool = False, random_seed: Union[None, int] = None, ):
        """
        A simple Gaussian noise model with a give `mean` and `covariance` matrix.

        :param mean: Wave object which form is used as the mean.
            The shape of the waveform must be equal to the waveform shape associated with the
            covariance matrix (extra dimension (e.g., for multiplication/replication) will be
            iterated over.
        :param covariance: the covariance matrix/operator.
        :param create_copies: copy the mean and covariance operators to detach from passed arguments
        :param random_seed: the random seed to be used with the underlying random number generator

        .. note::
            The passed mean and covariance are not copied locally to prevent space duplication
            unless `create_copies` is set to `True`.
        """
        # Check dimensionality and data types
        if not isinstance(mean, Wave):
            raise TypeError(
                f"The mean is expected to be a Wave instance not {type(mean)}"
            )
        if not isinstance(covariance, Covariance):
            raise TypeError(
                f"The covariance is expected to be a Covariance instance not {type(mean)}"
            )
        if covariance.waveform_shape not in [mean.shape, mean.shape[1: ]]:
            raise TypeError(
                f"Unconformable sizes/shapes of the mean and the covariance.\n"
                f"{mean.shape=}; {covariance.waveform_shape=}"
            )

        ## Keep track of mean and covariance and provide property-based access
        self._MEAN = mean.copy() if create_copies else mean
        self._COVARIANCE = covariance.copy() if create_copies else covariance

        # Lazy evaluation of the cholesky factor (for sampling)
        self._STDEV = None

        ## Set the random number seed (and if valid keep track of it.)
        self.update_random_number_generator(random_seed=random_seed)
        self._RANDOM_SEED = random_seed
        ## Initialization Done.

    def generate_white_noise(self, truncate : bool = False, truncate_threshold=3, ):
        """
        Safely generate white noise: normal random noise with zero mean and variance 1
        for each of the real and the imaginary parts. Note that scaling is taken care
        of by the (augmented) covariance matrices.

        This function returns two entries, noise_re, noise_im providing white noise
        for the real and the imaginary parts consecutively.

        :param bool truncate: if `True`, truncate the samples at -/+3, that is any sample
            point/entry above 3 is set to 3, and any value below -3 is set to -3.
            Truncation is applied to the real and imaginary parts individually.
            The value 3 here is set (and can be modified) by `truncate_threshold`.

        :param truncate_threshold: positive number (default 3) at which to do trunction
            if `truncate` is set to `True`.

        :returns: complex-valued white noise (composed of white noise of
            the real and the imaginary parts assuming they are independent which noises)
        """
        assert isinstance(truncate_threshold, Number) and truncate_threshold > 0, \
            (
                f"`truncate_threshold` must be positive number; "
                f"{truncate_theshold=} of {type(truncate_threshold)=}"
            )

        # Generate white noise with variance 1 (for both real and imaginary parts)
        white_noise = self.random_number_generator.standard_normal(self.size*2)

        # Split the noise vector
        noise_re = white_noise[ :self.size]
        noise_imag = white_noise[self.size: ]

        # cleanup
        del white_noise

        # Truncate (real and imaginary parts) if requested
        if truncate:
            noise_re[noise_re > truncate_threshold] = truncate_threshold
            noise_re[noise_re < -truncate_threshold] = -truncate_threshold

            noise_imag[noise_imag > truncate_threshold] = truncate_threshold
            noise_imag[noise_imag < -truncate_threshold] = -truncate_threshold

        # return results
        return (noise_re + 1.0j *noise_imag).reshape(self.waveform_shape)

    def generate_noise(self):
        """
        Generate a random noise vector sampled from the underlying Complex-Valued
        Gaussian distribution.
        This is a noise vector/array that is produced by multiplying a standard normal
        random vector by the lower Cholesky factor of the covariance matrix without
        adding the mean of the distribution.
        """
        # complex-valued white noise
        white_noise = self.generate_white_noise()

        # Apply STDEV
        noise = self.stdev.apply(white_noise, in_place=True, )

        return noise

    def sample(self):
        """
        Sample a random vector from the underlying Complex-Valued Gaussian distribution
        """
        if _DEBUG: t = time.time()
        # Add a scaled random noise (with the underlying covariance matrix) to the underlying mean
        sample = self.mean.copy()
        if sample.form.ndim == 3:
            # Generate noise (once) for all replicas
            noise = self.generate_noise()
            for j in range(sample.form.shape[0]):
                sample.form[j, ...] += noise
        elif sample.form.ndim == 2:
            sample.form += self.generate_noise()
        else:
            raise TypeError(
                f"Unexpected waveform shape {sample.form.shape}"
            )

        if _DEBUG:
            # TIME
            t = time.time() - t
            # Hours, minutes, seconds
            h, m, s = t//3600, (t-((t//3600)*3600))//60, t-((t-((t//3600)*3600))//60)*60
            print(f"Sampling took {h}:{m}:{s}")
        return sample

    @property
    def waveform_shape(self):
        """Shape of the underlying waveform (nx, ny)"""
        return self.covariance.waveform_shape

    @property
    def size(self):
        """Dimension of the underlying probability distribution"""
        return self.covariance.size

    @property
    def random_seed(self):
        """Registered random seed"""
        return self._RANDOM_SEED
    @random_seed.setter
    def random_seed(self, val):
        """Update the random seed."""
        out = self.update_random_number_generator(val)
        self._RANDOM_SEED = val
        return out

    @property
    def mean(self):
        """Reference to the distribution mean (underlying wave)"""
        return self._MEAN

    @property
    def covariance(self):
        """Reference to the distribution covariance (underlying Covariance operator)"""
        return self._COVARIANCE

    @property
    def stdev(self):
        """The Lower Cholesky factor of the underlying covariance matrix"""
        if self._STDEV is None:
            # Construct the Cholesky factor
            if _DEBUG: t = time.time()
            self._STDEV = self.covariance.cholesky(in_place=False, lower=True, )

            # Hours, minutes, seconds
            if _DEBUG:
                t = time.time() - t
                h, m, s = (
                    int(t//3600),
                    int((t-((t//3600)*3600))//60),
                    np.round(t-((t-((t//3600)*3600))//60)*60, 2)
                )
                print(f"Cholesky Factorization took {h}:{m}:{s}")

        return self._STDEV



