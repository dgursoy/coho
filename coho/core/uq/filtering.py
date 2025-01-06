"""Filtering classes for solving Bayesian inverse problems."""

# Standard imports
from abc import ABC, abstractmethod
from typing import Callable
from numbers import Number
import numpy as np
import scipy.sparse as sp

# Local imports
from ..component import Wave
from .covariance import (
    Covariance,
    DenseCovariance,
    SparseCovariance,
    cholesky,
)
from .noise import (
    GaussianNoise,
    ComplexGaussianNoise,
)

__all__ = [
    'Filter',
    'UnscentedKalmanFilter',
    'LocalUnscentedKalmanFilter'
]


class Filter(ABC):
    """Base class for all Filters."""

    def __init__(self, *args, **kwargs ) -> None:
        ...

    @abstractmethod
    def solve(self) -> GaussianNoise:
        """Solve the Bayesian inverse (filtering) problem and return the posterior."""
        pass


class UnscentedKalmanFilter(Filter):
    """
    Unscented Kalman Filter (UKF) applied to holography
    with :py:class:`GaussianNoise` prior.

    :param pom: The parameter-to-observable-map (POM) that maps the
        inference parameter to the observation space.
    :param prior: The Gaussian prior
    :param observation_noise: Observation noise/error model
    :param observation: The observational data
    :param alpha: determines the spread of the sigma points around the mean
        (first sigma point)
    :param beta: incorporates knowledge of the distribution of the inference
        variable. For Gaussian models, 2 is optimal and is so chosen here.
    :param kappa: secondary scaling parameter

    .. note::
        This implementation assumes time-independent formulation, and thus the
        prediction step of the filter (simulation model) is assumed identity.
        The parameter-to-observable-map is thus absorbed into the observation
        operator, and the observation error covariance should model model
        imperfection as well as observation noise.

    References
    ----------

    .. [1] Wan, Eric A., and Rudolph Van Der Merwe.
       "The unscented Kalman filter for nonlinear estimation."
       In Proceedings of the IEEE 2000 adaptive systems for signal processing,
       communications, and control symposium (Cat. No. 00EX373), pp. 153-158. Ieee, 2000.

    """

    def __init__(self, pom : Callable, prior: GaussianNoise,
                 observation_noise: GaussianNoise,
                 observation: np.ndarray,
                 alpha: float = 1e-3, beta: float = 2, kappa: float = 0) -> None:
        # Check types and instantiate
        if callable(pom):
            # We can try it on the prior mean if needed
            self._POM = pom
        else:
            raise TypeError(
                f"The parameter-to-observable map must be a callable; "
                f"received {pom=} of {type(pom)=}"
            )

        if isinstance(prior, GaussianNoise) and type(prior) is GaussianNoise:
            self._PRIOR = prior
        else:
            raise TypeError(
                f"Expected prior to be GaussianNoise not {type(prior)=}"
            )

        if isinstance(observation_noise, GaussianNoise) and type(prior) is GaussianNoise:
            self._OBSERVATION_NOISE = observation_noise
        else:
            raise TypeError(
                f"Expected observation_noise to be GaussianNoise "
                f"not {type(observation_noise)=}"
            )

        if isinstance(observation, np.ndarray):
            self._OBSERVATION = observation
        else:
            raise TypeError(
                f"Expected observation to be numpy array not {type(observation)=}"
            )

        if isinstance(alpha, Number):
            self._ALPHA = alpha
        else:
            raise TypeError(
                "alpha must be number; received {alpha=} of {type(alpha)=}"
            )
        if isinstance(beta, Number):
            self._BETA = beta
        else:
            raise TypeError(
                "beta must be number; received {beta=} of {type(beta)=}"
            )
        if isinstance(kappa, Number):
            self._KAPPA = kappa
        else:
            raise TypeError(
                "kappa must be number; received {kappa=} of {type(kappa)=}"
            )

        # Place-holders for sigma points and their weights
        self._SIGMA_POINTS = self._SIGMA_POINTS_WEIGHTS = None

    def solve(self, return_np=False, ) -> GaussianNoise:
        """
        Solve the Bayesian inverse (filtering) problem and return the posterior.
        This currently returns the output of :py:meth:`_analysis`.

        .. note:
            We might need to change that so it returns
            a :py:class:`GaussianNoise` model instead!
        """
        # TODO: Fix mean update...
        x, P = self._analysis()
        x = x.reshape(covariance.waveform_shape[-2: ])
        if return_np:
            return x, P
        else:
            mean = self.prior.mean.copy()
            mean.form = x[np.newaxis, ...]

            covariance = self.prior.covariance.copy()
            covariance.covariance_array = P

            posterior = GaussianNoise(
                mean=mean,
                covariance=covariance,
                create_copies=False,
            )
            return posterior

    def _create_sigma_points(self, alpha=1e-3, beta=2, kappa=0):
        """
        Create and return the sigma points and their weights to be
        used for evaluating means and covariances, respectively.
        The sigma points are stored as an array (two-dimensional) with
        each sigma point (sample) stored as a row.
        Thus, sigma points can be iterated over using list-style.
        The weights are divided into `mean_weights`, and `covariance_weights`.
        Because, the and the covariance are assigned two sets of weights (respectively)
        for the first sigma point (mean) and the others, the returned weights are
        two tuples.

        .. note::
            We can store long array containing all weights, but there will be redundancy.
            This might change if the weights evaluation scheme changes.

        :param alpha: determines the spread of the sigma points around the mean
            (first sigma point)
        :param beta: incorporates knowledge of the distribution of the inference
            variable. For Gaussian models, 2 is optimal and is so chosen here.
        :param kappa: secondary scaling parameter

        :returns:
            - `sigma_points`: 2d array holding sigma points (as rows)
            - `mean_weights`: a tuple `(w0, wi)` where `w0` is the weight given to the first
                sigma point (the mean), and `wi` is associated with all other sigma points.
            - `covariance_weights`: same as `mean_weights`; this is a tuple `(w0, wi)` where
                `w0` is the weight given to the first sigma point (the mean), and
                `wi` is associated with all other sigma points.
        """
        ## Parameters and Configurations
        L = self.size
        _lambda = alpha**2 * (L + kappa) - L
        scl = np.sqrt(L+_lambda)  # scalar multiplied by Cholesky factor of the covariance

        # Number of sigma points
        num_sigma_points = 2 * L + 1

        ## Generate the sigma points

        # Extract prior mean (as numpy array)
        prior_mean = self.prior.mean
        prior_mean_np = prior_mean.form.ravel()
        if prior_mean_np.size != L:
            raise TypeError(
                f"Sigma points creator is invalid for this prior mean shape! "
                f"Found {self.mean.shape=}"
            )

        # Placeholder for all sigma points (with mean set in all column)
        sigma_points = np.vstack([prior_mean_np]*num_sigma_points).T
        if sigma_points.ndim == 1:
            sigma_points = sigma_points.reshape((sigma_points.size, 1))

        # First L sigma points (add to mean)
        sigma_points[:, 1: L+1] += scl * self.prior.stdev.covariance_array

        # Second L sigma points (subtract from mean)
        sigma_points[:, 1: L+1] -= scl * self.prior.stdev.covariance_array

        ## Evaluate the weights of the sigma points
        # Weights for evaluating the mean
        w0 = _lambda / (L + _lambda)
        wi = 1 / ( 2 * (L+_lambda) )
        mean_weights = (w0, wi)

        # Covariance weights
        w0 = _lambda / (L + _lambda) + (1 - alpha**2 + beta)
        wi = mean_weights[-1]
        covariance_weights = (w0, wi)

        return sigma_points, (mean_weights, covariance_weights)

    def _analysis(self, ):
        """
        Carry out the analysis/correction step of the filter.
        This method is called internally by the :py:meth:`solve` method
        to solve the Bayesian inverse problem.

        .. note::
            This implementation is global and DOES NOT ASSUME LOCALIZATION.
            Thus, efficient implementation would apply this filter locally for
            a moving window over connected pixels.

        .. warning::
            Calling this method on a global scale is inefficient and should
            be avoided as it will assume spurious correlations and will be
            extremely expensive. See the note above!
        """
        # Prediction (prior) mean and covariance
        xp = self.prior.mean
        P = self.prior.covariance

        # Observation error (noise) covariance matrix
        R = self.observation_noise.covariance

        # Observations
        Z = np.vstack(
            [
                self.pom(x)
                for x in self.sigma_points
            ]
        )
        z_mean = np.mean(Z, axis=0)
        for i in range(Z.shape[0]):
            Z[i, :] -= z_mean

        # Theoretical observation error covariance
        Zw = Z.copy()
        w0 = self.sigma_points_weights[-1][0]
        wi = self.sigma_points_weights[-1][-1]
        Zw[0, :] *= w0
        for i in range(1, Zw.shape[0]):
            Zw[i, :] *= wi
        S = Z @ Z.conjugate().T + R

        # Mean of the sigma points
        w0 = self.sigma_points_weights[0][0]
        wi = self.sigma_points_weights[0][-1]
        sigma_points_mean = self.sigma_points[0] * w0
        for s in self.sigma_points[1: ]:
            sigma_points_mean += wi * s

        # Cross-covariance
        Pxz = np.vstack(
            [
                sigma_point - sigma_points_mean
                for sigma_point in self.sigma_points
            ]
        )
        w0 = self.sigma_points_weights[-1][0]
        wi = self.sigma_points_weights[-1][-1]
        Pxz[0, :] *= w0
        for i in range(1, Pxz.shape[0]):
            Pxz[i, :] *= wi
        Pxz @= Z.conjugate().T

        inverse = sp.linalg.inv if sp.issparse(S) else np.linalg.inv

        # Kalman gain
        K = Pxz @ inverse(S)

        # Posterior
        x = xp + K @ (self.observation - z_mean)
        P = P - K @ S @ K.conjugate().T

        return x, P

    @property
    def parameter_to_observable_map(self):
        """The registered parameter-to-observable map (forward operator)"""
        return self._POM
    pom = parameter_to_observable_map

    @property
    def alpha(self):
        return self._ALPHA
    @property
    def beta(self):
        return self._BETA
    @property
    def kappa(self):
        return self._KAPPA

    @property
    def prior(self):
        """
        The registered prior.
        """
        return self._PRIOR

    @property
    def observation_noise(self):
        return self._OBSERVATION_NOISE

    @property
    def observation(self):
        return self._OBSERVATION

    @property
    def size(self):
        """
        Dimension of the state space (inference variable).
        This is extracted from the registered prior
        """
        return self.prior.size

    @property
    def sigma_points(self):
        """
        Sigma points. If not created, they are instantiated
        """
        if self._SIGMA_POINTS is None or self._SIGMA_POINTS is None:
            self._SIGMA_POINTS, self._SIGMA_POINTS_WEIGHTS = self._create_sigma_points(
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa,
            )
        return self._SIGMA_POINTS

    @property
    def sigma_points_weights(self):
        """
        Sigma points weights. If not created, they are instantiated
        """
        if self._SIGMA_POINTS is None or self._SIGMA_POINTS is None:
            self._SIGMA_POINTS, self._SIGMA_POINTS_WEIGHTS = self._create_sigma_points(
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa,
            )
        return self._SIGMA_POINTS_WEIGHTS


class LocalUnscentedKalmanFilter(Filter):
    """
    This an implementation that applies the UnscentedKalmanFilter
    locally to connected pixels to construct posterior only for locally connected
    pixels.
    This basically operates by updating 1) mean entries one at a time, and
    2) covariance rows/columns one at a time.
    Note that those updates are completely parallelizable.
    """

    def __init__(self, pom : Callable, prior: GaussianNoise,
                 observation_noise: GaussianNoise,
                 observation: np.ndarray,
                 alpha: float = 1e-3, beta: float = 2, kappa: float = 0) -> None:
        # Check types and instantiate
        if callable(pom):
            # We can try it on the prior mean if needed
            self._POM = pom
        else:
            raise TypeError(
                f"The parameter-to-observable map must be a callable; "
                f"received {pom=} of {type(pom)=}"
            )

        if (
            isinstance(prior, GaussianNoise) and type(prior) is GaussianNoise and
            isinstance(observation_noise, GaussianNoise) and type(prior) is GaussianNoise
        ):

            if (
                isinstance(prior.covariance, SparseCovariance) and
                isinstance(observation_noise.covariance, SparseCovariance)
            ):
                # Now make sure covariances are sparse
                self._PRIOR = prior
                self._OBSERVATION_NOISE = observation_noise
            else:
                raise TypeError(
                    f"Expected prior and noise models to be associated with ;"
                    f"sparse covariances. received {type(prior.covariance)=} "
                    f"{type(observation_noise.covariance)=}"
                )

        else:
            raise TypeError(
                f"Expected prior and noise models to be GaussianNoise;"
                f"received {type(prior)=} {type(observation_noise)=}"
            )

        if isinstance(observation, np.ndarray):
            self._OBSERVATION = observation
        else:
            raise TypeError(
                f"Expected observation to be numpy array not {type(observation)=}"
            )

        if isinstance(alpha, Number):
            self._ALPHA = alpha
        else:
            raise TypeError(
                "alpha must be number; received {alpha=} of {type(alpha)=}"
            )
        if isinstance(beta, Number):
            self._BETA = beta
        else:
            raise TypeError(
                "beta must be number; received {beta=} of {type(beta)=}"
            )
        if isinstance(kappa, Number):
            self._KAPPA = kappa
        else:
            raise TypeError(
                "kappa must be number; received {kappa=} of {type(kappa)=}"
            )

        # Place-holders for sigma points and their weights
        self._SIGMA_POINTS = self._SIGMA_POINTS_WEIGHTS = None

    def solve(self, return_np=False, ) -> GaussianNoise:
        """
        Solve the Bayesian inverse (filtering) problem and return the posterior.
        This currently returns the output of :py:meth:`_analysis`.

        .. note:
            We might need to change that so it returns
            a :py:class:`GaussianNoise` model instead!
        """
        # Apply UKF locally for each entry of the mean
        posterior = self.prior.copy()

        # Mean as 1d numpy array
        global_mean = self.prior.mean.form.flatten()
        global_observation = self.observation.flatten()

        # Coordinates with non-zero covariances
        row_coord, col_coord = self.prior.covariance.coord
        covariance_data = self.prior.covariance.data

        # For each entry, get the corresponding pixel and the ones correlated to it
        for i in range(posterior.size):
            nonzero_locs = col_coord[np.where(row_coord==i)[0]]
            nonzero_locs = nonzero_locs[nonzero_locs>=i]

            mean = global_mean[nonzero_locs]
            prior_covariance = self.prior.covariance.covariance_array[nonzero_locs, :][:, nonzero_locs].toarray()

            local_prior = GaussianNoise(
                mean=Wave(
                    form=mean.reshape((nonzero_locs.size, 1)),
                    energy=self.prior.mean.energy,
                    spacing=self.prior.mean.spacing,
                    position=self.prior.mean.position,
                    x=self.prior.mean.x,
                    y=self.prior.mean.y,
                ),
                covariance=DenseCovariance(
                    waveform_shape=(nonzero_locs.size, 1),
                    covariance=prior_covariance,
                ),
            )

            noise_covariance = self.observation_noise.covariance.covariance_array[nonzero_locs, :][:, nonzero_locs].toarray()
            local_noise = GaussianNoise(
                mean=Wave(
                    form=np.zeros((nonzero_locs.size, 1)),
                ),
                covariance=DenseCovariance(
                    waveform_shape=(nonzero_locs.size, 1),
                    covariance=noise_covariance,
                ),
            )

            local_observation = global_observation[nonzero_locs]

            solver = UnscentedKalmanFilter(
                pom=self.pom,
                prior=local_prior,
                observation_noise=local_noise,
                observation=local_observation,
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa,
            )
            x, P = solver._analysis()

            # TODO: Fix mean update index (for 3d...)
            # Update ith mean entry
            x_ind = i / posterior.covariance.waveform_shape[-1]
            y_ind = i % posterior.covariance.waveform_shape[-1]
            posterior.mean.form[x_ind, y_ind] = x[0]

            # Update ith row column
            posterior.covariance.covariance_array[[i], nonzero_locs] = P[0, :]
            posterior.covariance.covariance_array[nonzero_locs, [i]] = P[:, 0]

        return posterior

    @property
    def parameter_to_observable_map(self):
        """The registered parameter-to-observable map (forward operator)"""
        return self._POM
    pom = parameter_to_observable_map

    @property
    def alpha(self):
        return self._ALPHA
    @property
    def beta(self):
        return self._BETA
    @property
    def kappa(self):
        return self._KAPPA

    @property
    def prior(self):
        """
        The registered prior.
        """
        return self._PRIOR

    @property
    def observation_noise(self):
        return self._OBSERVATION_NOISE

    @property
    def observation(self):
        return self._OBSERVATION

    @property
    def size(self):
        """
        Dimension of the state space (inference variable).
        This is extracted from the registered prior
        """
        return self.prior.size

