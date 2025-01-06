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
import scipy.sparse as sp
from abc import ABC, abstractmethod, abstractproperty
from typing import Union  # Type, Tuple, Callable,

# Local imports
from .covariance import (
    Covariance,
    cholesky,
)
from ..component import Wave

__all__ = [
    'Noise',
    'GaussianNoise',
    'ComplexGaussianNoise',
    'real_to_complex_covariances',
    'complex_to_real_covariances',
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
    """
    A simple Gaussian noise model with a give `mean` and `covariance` matrix.
    This models only mean and covariances of the random variable.

    .. info::
        While this model has a default `dtype` set to `complex` for storing mean and covaariances,
        it can only handle mean and ``Hermitian`` covariances but it ignores any pseudo covariances.
        Full capture of such relations can be achieved by employing :py:class:`ComplexGaussianNoise`.

    :param mean: Wave object which form is used as the mean.
        The shape of the waveform must be equal to the waveform shape associated with the
        covariance matrix (extra dimension (e.g., for multiplication/replication) will be
        iterated over.
    :param covariance: the covariance matrix/operator.
    :param create_copies: copy the mean and covariance operators to detach from passed arguments
    :param random_seed: the random seed to be used with the underlying random number generator
    :param verbose: screen verbosity

    .. note::
        The passed mean and covariance are not copied locally to prevent space duplication
        unless `create_copies` is set to `True`.
    """

    def __init__(
        self,
        mean: Union[Wave, np.ndarray],
        covariance: Covariance,
        create_copies: bool = False,
        random_seed: Union[None, int] = None,
        verbose: bool = False,
    ):
        # Check dimensionality and data types
        if not isinstance(mean, (Wave, np.ndarray)):
            raise TypeError(
                f"The mean is expected to be a numpy array or a Wave instance not {type(mean)=}"
            )
        if not isinstance(covariance, Covariance):
            raise TypeError(
                f"The covariance is expected to be a Covariance instance not {type(covariance)=}"
            )
        if not (
            covariance.waveform_shape in [mean.shape, mean.shape[-2: ]] or
            (
                isinstance(mean, np.ndarray) and
                mean.ndim == 1 and
                covariance.waveform_shape[-2]*covariance.waveform_shape[-1] == mean.size
            )
        ):
            raise TypeError(
                f"Unconformable types/sizes/shapes of the mean and the covariance.\n"
                f"{type(mean)=}; {type(covariance)=}\n"
                f"{mean.shape=}; {covariance.waveform_shape=}"
            )

        ## Keep track of mean and covariance and provide property-based access
        self._MEAN = mean.copy() if create_copies else mean
        self._COVARIANCE = covariance.copy() if create_copies else covariance

        # Verbosity
        self._VERBOSE = bool(verbose)

        # Lazy evaluation of the cholesky factor (for sampling) & inverse if needed
        self._STDEV = None
        self._STDEV_INV = None

        ## Set the random number seed (and if valid keep track of it.)
        self.update_random_number_generator(random_seed=random_seed)
        self._RANDOM_SEED = random_seed
        ## Initialization Done.

    def copy(self):
        return GaussianNoise(
            mean=self.mean,
            covariance=self.covariance,
            create_copies=True,
            random_seed=self.random_seed,
            verbose=self.verbose,
        )

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
        return (noise_re + 1j *noise_imag).reshape(self.waveform_shape[-2: ])

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
        if isinstance(sample, Wave):
            sample_form = sample.form
        elif isinstance(sample, np.ndarray):
            sample_form = sample
        else:
            raise TypeError(
                f"Unexpected {type(self.mean)=}"
            )
        if sample_form.ndim == 3:
            # Generate noise (once) for all replicas
            noise = self.generate_noise()
            for j in range(sample.form.shape[0]):
                sample_form[j, ...] += noise
        elif sample_form.ndim == 2:
            sample_form += self.generate_noise()
        elif sample_form.ndim == 1:
            sample_form += self.generate_noise().ravel()
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
            self._STDEV = self.covariance.cholesky(lower=True, )

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

    @property
    def stdev_inv(self):
        """
        Inverse of the lower Cholesky factor of the covariance matrix.

        .. warning::
            This is expensive and should not be used for real settings.
            Expected only for testing...
        """
        if self._STDEV_INV is None:
            # Construct the Inverse of the lower cholesky factor
            stdev = self.stdev
            if _DEBUG: t = time.time()
            self._STDEV_INV = sp.linalg.inv(stdev)

            # Hours, minutes, seconds
            if _DEBUG:
                t = time.time() - t
                h, m, s = (
                    int(t//3600),
                    int((t-((t//3600)*3600))//60),
                    np.round(t-((t-((t//3600)*3600))//60)*60, 2)
                )
                print(f"Inversion of the Lower Cholesky Factorization took {h}:{m}:{s}")
        return self._STDEV

    @property
    def precision(self, ):
        r"""
        Calculate the precision matrix. This is just the inverse of the
        covariance matrix.

        .. warning::
            This is expensive and should not be used for real settings.
            Expected only for testing...

        :returns: 2D array representing inverse of the covariances of the complex variable.
        """
        return self.stdev_inv.conjugate().T @ self.stdev_inv

    @property
    def dtype(self):
        """Default data type to use (inferred from mean data type)"""
        return self.covariance.dtype
        # return self.covariance.dtype

    @property
    def verbose(self):
        return self._VERBOSE
    @verbose.setter
    def verbose(self, val):
        self._VERBOSE = bool(val)


class ComplexGaussianNoise(GaussianNoise):
    """
    A simple Numpy-based (or scipy sparse) Complex-Valued Gaussian error model.
    A complex-valued random variable/vector ::math:`\\mathbf{Z}` is said to follow
    a Gaussian distribution ::math:`\\mathbf{Z}\\sim \\mathcal{CN}(\\mu, \\Gamma, C)` iff
    its real and imaginary parts are normally distributed. ::math:`\\Gamma` is
    a positive (semi-) definite covariance (defining total covariances of both real
    and imaginary parts) matrix with real-valued diagonals, and ::math:`C` is the
    relation matrix defining covariances between real and imaginary parts.

    This differs from :py:class:`GaussianErrorModel` in that it provides
    covariance matrix as well as a relation (pseudo-covariance) matrix.

    .. info::
        This is copied from PyOED; see
        (https://gitlab.com/ahmedattia/pyoed/-/blob/main/pyoed/models/error_models/Gaussian/complexGaussian.py).

    .. note::
        This model creates lazy objects under the hood. These are lazy in the sense that
        they are constructed/calculated only when they are needed, and are kept in memory
        unless the design changes. These objects are accessible through the following
        properties:

        1. `complex_augmented_stdev`: this refers to a lazy evaluation of the lower
           Cholesky factor of the complex augmented covariance matrix.

        2. `complex_augmented_stdev_inv`: this refers to a lazy evaluation of the inverse
           of the lower Cholesky factor of the complex augmented covariance matrix.

        3. `real_composite_stdev`: this refers to a lazy evaluation of the lower
           Cholesky factor of the real composite covariance matrix.

        4. `real_composite_stdev_inv`: this refers to a lazy evaluation of the inverse
           of the lower Cholesky factor of the real composite covariance matrix.


    :param mean: Wave object which form is used as the mean.
        The shape of the waveform must be equal to the waveform shape associated with the
        covariance matrix (extra dimension (e.g., for multiplication/replication) will be
        iterated over.
    :param covariance: the covariance matrix/operator.
    :param pseudo_covariance: the pseudo-covariance (relation) matrix/operator.
    :param create_copies: copy the mean and covariance operators to detach from passed arguments
    :param map_to_real: apply computations (e.g., pdf, etc.) by mapping to the real domain
        using the duality between with the composite real vector
    :param random_seed: the random seed to be used with the underlying random number generator
    :param verbose: screen verbosity

    .. note::
        The passed mean and covariance are not copied locally to prevent space duplication
        unless `create_copies` is set to `True`.
    """

    def __init__(self, mean: Wave, covariance: Covariance, pseudo_covariance: Covariance,
                 create_copies: bool = False, map_to_real: bool = True,
                 random_seed: Union[None, int] = None, verbose: bool = False ):

        # Validate everything and instantiate (except for pseudo covariances)
        super().__init__(
            mean=mean,
            covariance=covariance,
            create_copies=create_copies,
            random_seed=random_seed,
            verbose=verbose,
        )

        # Cleanup Unneeded attributes
        del self._STDEV

        ## Now, validate pseudo covariances
        if pseudo_covariance.waveform_shape not in [mean.shape, mean.shape[1: ]]:
            raise TypeError(
                f"Unconformable sizes/shapes of the pseudo_covariance.\n"
                f"{mean.shape=}; {pseudo_covariance.waveform_shape=}"
            )

        ## Keep track of mean and covariance and provide property-based access
        self._PSEUDO_COVARIANCE = pseudo_covariance.copy() if create_copies else pseudo_covariance

        # Lazy initialization of crucial matrices and constants:
        self._COMPLEX_AUGMENTED_STDEV = None
        self._REAL_COMPOSITE_STDEV = None

        self._COMPLEX_AUGMENTED_STDEV_INV = None
        self._REAL_COMPOSITE_STDEV_INV = None

        # Approach used for carrying out operations (augmented vs composite)
        self._MAP_TO_REAL = bool(map_to_real)

        ## Initialization Done.

    def generate_noise(self):
        """
        Generate a random noise vector sampled from the underlying Complex-Valued
        Gaussian distribution.
        This sampling is generic and does not assume circular symmetry
        """
        # complex-valued white noise
        white_noise = self.generate_white_noise().ravel()

        # Scale the aggregated vector by standard deviation (in the real space)
        if self.map_to_real:
            # Convert the complex noise to a composite real vector
            noise_composite = np.concatenate([white_noise.real, white_noise.imag])
            noise_composite[:] = self.real_composite_stdev @ noise_composite

            # Map to the complex space
            noise = noise_composite[: self.size] + 1j * noise_composite[self.size :]

        else:
            # Convert the complex noise to a complex augmented vector
            noise_augmented = np.concatenate([white_noise, white_noise.conjugate()])
            noise_augmented[:] = self.complex_augmented_stdev @ noise_augmented
            noise_augmented *= 1.0 / np.sqrt(2)  # NOTE: This is because we use `z` and conj

            # Extract first half (samples of `z` and drop samples of the conjugate)
            noise = noise_augmented[: self.size] + noise_augmented[self.size: ].conjugate()
            noise /= 2.0

        # Reshape and return
        noise = noise.reshape(self.waveform_shape[-2: ])
        return noise

    def real_composite_covariance(self, ):
        """
        Calculate the real composite covariance matrix. This is a matrix with
        four blocks defining covariances of the real and imaginary parts and
        the cross-covariances between real and imaginary parts.

        :returns: 2D array representing covariances of the composite (real
            and imaginary) parts of the random variable/vector.
        """
        # Cov: The covariance matrix (of the complex random variable)
        Cov = self.covariance.covariance_array

        # PCov: The Pseudo-covariance (Relation) matrix (of the complex random variable)
        PCov = self.pseudo_covariance.covariance_array

        ## Extract covariance (and cross-covariances) of the real and imaginary parts
        # Covariance of the real components
        C_RR = 0.5 * (Cov.real + PCov.real)

        # Covariance of the imaginary components
        C_II = 0.5 * (Cov.real - PCov.real)

        # Cross covariances of real and imaginary components
        C_RI = 0.5 * (PCov.imag - Cov.imag)

        # The real composite covariance matrix (Block matrix)
        composite_covariances = sp.block_array(
            [
                [ C_RR,  C_RI],
                [C_RI.T, C_II]
            ],
            format=self.covariance.format,
            dtype=float,
        )
        return composite_covariances

    def complex_augmented_covariance(self, ):
        """
        Calculate the augmented covariance matrix and return it (full).

        :returns: 2D array representing covariances of the augmented
            (complex variable and its conjugate) of the random variable/vector.
        """
        # Cov: The covariance matrix (of the complex random variable)
        Cov = self.covariance.covariance_array

        # PCov: The Pseudo-covariance (Relation) matrix (of the complex random variable)
        PCov = self.pseudo_covariance.covariance_array

        # The complex augmented covariance matrix (Block matrix)
        augmented_covariances = sp.block_array(
            [
                [ Cov,  PCov],
                [PCov.conjugate(), Cov.conjugate()]
            ],
            format=Cov.format,
            dtype=self.dtype,
        )
        return augmented_covariances

    def complex_augmented_covariance_from_sample(
        self,
        sample,
        ddof=1,
        localize=True,
    ):
        """
        Evaluate the complex augmented covariance matrix from a sample.
        This method calculates the sample based covariances and pseudo covariances
        and use them to construct the augmented covariance matrix.

        :parm sample: ArrayLike iterable of two dimensions. Each row represetns
            a sample of the underlying distribution
        :param int ddof: `ddof=1` will return unbiased estimate since in the
            covariance formula, we divide by sample_size-ddof.
        :param localize: if `True`, only entries corresponding to active entries in
            the covariance matrix are stored; all others are ignored

        :returns:
            - `covariances`: sample-based covariance matrix
            - `pseudo_covariances`: sample-based pseudo covariance matrix
        """
        # Extract sample size and probability space dimension
        sample = np.asarray(sample, dtype=complex)
        if np.ndim(sample) == 1:
            np.reshape(sample, (sample.size, 1))

        # Extract sample size
        sample_size = len(sample)

        # Innovation (sample shifted to center around the mean)
        innov = sample - np.mean(sample, axis=0)

        # Covariances and Pseudo Covariances (Relations)
        scl  = 1.0 / (sample_size - ddof)

        if localize:
            raise NotImplementedError(
                "TODO: Localized sample calculator is not impelemented yet."
            )
            # TODO: Only evaluate entries corresponding to self.covariance.coord
            # Cov = ...
            # PCov = ...
        else:
            Cov  = scl * (innov.T @ innov.conjugate()).T
            PCov = scl * (innov.T @ innov).T

        # The complex augmented covariance matrix
        augmented_covariances = sp.block_array(
            [
                [ Cov,  PCov],
                [PCov.conjugate(), Cov.conjugate()]
            ],
            format=self.covariance.format,
            dtype=self.dtype,
        )
        return augmented_covariances

    def real_composite_covariance_from_sample(
        self,
        sample,
        ddof=1,
        localize=True,
    ):
        """
        Evaluate the real composite covariance matrix from a sample.
        This method calculates the covariance matrix of the composite real vector
        composed of the real and the imaginary parts of the complex vector.

        :parm sample: ArrayLike iterable of two dimensions. Each row represetns
            a sample of the underlying distribution
        :param int ddof: `ddof=1` will return unbiased estimate since in the
            covariance formula, we divide by sample_size-ddof.
        :param localize: if `True`, only entries corresponding to active entries in
            the covariance matrix are stored; all others are ignored

        :returns:
            - `covariances`: sample-based covariance matrix
            - `pseudo_covariances`: sample-based pseudo covariance matrix
        """
        # Extract sample size and probability space dimension
        sample = np.asarray(sample, dtype=complex)
        if np.ndim(sample) == 1:
            np.reshape(sample, (sample.size, 1))

        # Extract sample size
        sample_size = len(sample)

        # Innovation (sample shifted to center around the mean)
        innov = sample - np.mean(sample, axis=0)

        innov_real = innov.real
        innov_imag = innov.imag

        # Scaling factor
        scl  = 1.0 / (sample_size - ddof)



        # Covariance of parts
        if localize:
            raise NotImplementedError(
                "TODO: Localized sample calculator is not impelemented yet."
            )
            # TODO: Only evaluate entries corresponding to self.covariance.coord
            # C_RR = ...
            # C_II = ...
            # C_RI = ...
        else:
            C_RR = scl * (innov_real.T @ innov_real)
            C_II = scl * (innov_imag.T @ innov_imag)
            C_RI = scl * (innov_real.T @ innov_imag)

        # Covariances of the real composite vector (Block matrix)
        composite_covariances = sp.block_array(
            [
                [C_RR, C_RI],
                [C_RI.T, C_II]
            ],
            format=float,
            dtype=self.dtype,
        )

        return composite_covariances

    @property
    def pseudo_covariance(self, ):
        r"""
        Calculate the relation (pseudo-covariance) matrix and return it.
        The relation matrix is defined as:

        .. math::
            \mathbb{E} (x - \mu ) ( x - \mu )^{T}
            = \mathbb{E} \left(x - \mathbb{E}(x) \right)
                \left( x - \mathbb{E}(x) \right)^{T} \,,

            where ::math:`x` is the random variable, ::math:`\mu = \mathbb{E}[x]` is the
            distribution mean, and ::math:`x^{T}` is the transpose of ::math:`x`.

        :returns: 2D array representing relation pseudo-covariances of the complex variable.
        """
        # Copy the covariance matrix (of the complex random variable)
        return self._PSEUDO_COVARIANCE

    @property
    def real_composite_stdev(self, ):
        """
        Construct and return the lower Cholesky factor (if not available) in the
        projected space (after applying the design) of the real composite
        covariance matrix
        """
        if self._REAL_COMPOSITE_STDEV is None:
            if self.verbose:
                print(
                    f"Generating lower Cholesky factor of the real composite "
                    f"covariance matrix"
                )

            self._REAL_COMPOSITE_STDEV = cholesky(
                self.real_composite_covariance(),
                lower=True,
            )

        # Return it
        return self._REAL_COMPOSITE_STDEV

    @property
    def real_composite_stdev_inv(self, ):
        """
        Construct and return the inverse of the lower Cholesky factor (if not
        available) in the projected space (after applying the design)
        of the real composite covariance matrix
        """
        if self._REAL_COMPOSITE_STDEV_INV is None:
            if self.verbose:
                print(
                    f"Generating the inverse of the lower Cholesky factor of "
                    f"the real composite covariance matrix"
                )
            # Calculate the inverse
            self._REAL_COMPOSITE_STDEV_INV = sp.linalg.inv(self.real_composite_stdev)

        # Return it
        return self._REAL_COMPOSITE_STDEV

    @property
    def complex_augmented_stdev(self, ):
        """
        Construct and return the lower Cholesky factor (if not available) in the
        projected space (after applying the design) of the complex augmented
        covariance matrix
        """
        if self._COMPLEX_AUGMENTED_STDEV is None:
            if self.verbose:
                print(
                    f"Generating lower Cholesky factor of the complex "
                    f"augmented covariance matrix"
                )

            self._COMPLEX_AUGMENTED_STDEV = cholesky(
                self.complex_augmented_covariance(),
                lower=True,
            )
        return self._COMPLEX_AUGMENTED_STDEV

    @property
    def complex_augmented_stdev_inv(self, ):
        """
        Construct and return the inverse of the lower Cholesky factor (if not
        available) of the complex augmented covariance matrix
        in the projected space (after applying the design)
        """
        if self._COMPLEX_AUGMENTED_STDEV_INV is None:
            if self.verbose:
                print(
                    f"Generating the inverse of the lower Cholesky factor of "
                    f"the complex augmented covariance matrix"
                )

            # Calculate the inverse
            self._COMPLEX_AUGMENTED_STDEV_INV = sp.linalg.inv(self.complex_augmented_stdev)

        # Return it
        return self._COMPLEX_AUGMENTED_STDEV_INV

    @property
    def map_to_real(self, ):
        """
        Whether to map to real domain (of `True`) for calculations
        (i.e., use the real-composite formulation)
        or to use the complex augmented formulation
        """
        return self._MAP_TO_REAL
    @map_to_real.setter
    def map_to_real(self, val):
        """Update `map_to_real`"""
        self._MAP_TO_REAL=bool(val)


##################################################################################
##                               Helper Functions                               ##
##################################################################################
def real_to_complex_covariances(
    C_RR,
    C_II,
    C_RI,
):
    """
    Convert covariances and cross covariances or real and imaginary parts
    of a complex random variable into covariances and pseudo-covariances
    (relations) of the complex vector

    :param C_RR: covariances of the real part of a complex random vector/variable
    :param C_II: covariances of the imaginary part of a complex random vector/variable
    :param C_RI: cross-covariances of the real with the imaginar part of
        a complex random vector/variable

    :returns: Two matrices in the following order:
        - Cov: the covariances of the complex variable
        - PCov: the pseudo covariances
    """
    # Check data types and shapes
    if not (
        sp.issparse(C_RR) and
        sp.issparse(C_II) and
        sp.issparse(C_RI)
    ):
        raise TypeError(
            f"All arguments must be scipy-based sparse arrays;\n"
            f"{type(C_RR)=}\n"
            f"{type(C_II)=}\n"
            f"{type(C_RI)=}"
        )
    if not (
        C_RR.shape == C_II.shape == C_RI.shape and
        C_RR.ndim==2 and
        C_RR.shape[0] == C_RR.shape[1] > 0
    ):
        raise TypeError(
            f"Invalid or inconsistent covariance matrices; "
            f"expected square matrices of equal shpaes;\n"
            f"Received: \n"
            f"{C_RR.shape=}\n"
            f"{C_II.shape=}\n"
            f"{C_RI.shape=}"
        )

    # Calculate covariance and pseudo covariances
    Cov = C_RR + C_II + 1j * (C_RI.T - C_RI)
    PCov = C_RR - C_II + 1j * (C_RI.T + C_RI)
    return Cov, PCov

def complex_to_real_covariances(
    Cov,
    PCov,
):
    """
    Convert covariances and cross covariances or real and imaginary parts to
    covariances and pseudo-covariances (relations) of the complex vector

    :param Cov: the covariances of the complex variable
    :param PCov: the pseudo covariances

    :returns: Three matrices in the following order:
        - C_RR: covariances of the real part of a complex random vector/variable
        - C_II: covariances of the imaginary part of a complex random vector/variable
        - C_RI: cross-covariances of the real with the imaginary part of
          a complex random vector/variable
    """
    # Check data types and shapes
    if isinstance(Cov, Covariance) and isinstance(PCov, Covariance):
        Cov = Cov.covariance_array
        PCov = PCov.covariance_array
    if not (
        sp.issparse(Cov) and
        sp.issparse(PCov)
    ):
        raise TypeError(
            f"All arguments must be scipy-based sparse arrays;\n"
            f"{type(Cov)=}\n"
            f"{type(PCov)=}"
        )
    if not (
        Cov.shape == PCov.shape and
        Cov.ndim==2 and
        Cov.shape[0] == Cov.shape[1] > 0
    ):
        raise TypeError(
            f"Invalid or inconsistent covariance matrices; "
            f"expected square matrices of equal shpaes;\n"
            f"Received: \n"
            f"{Cov.shape=}\n"
            f"{PCov.shape=}"
        )

    ## Extract covariance (and cross-covariances) of the real and imaginary parts
    # Covariance of the real components
    C_RR = 0.5 * (Cov.real + PCov.real)

    # Covariance of the imaginary components
    C_II = 0.5 * (Cov.real - PCov.real)

    # Cross covariances of real and imaginary components
    C_RI = 0.5 * (PCov.imag - Cov.imag)

    return C_RR, C_II, C_RI


