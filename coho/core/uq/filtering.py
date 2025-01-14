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

    .. note::
        This implementation assumes only numpy arrays as data structures.
        Thus, this implementation should not be used for high-dimensional inference.
        It can be employed for local application of unscented kalman filtering.

    :param pom: The parameter-to-observable-map (POM) that maps the
        inference parameter to the observation space.
        This function takes a one-dimensional array (same as inference parameter)
        and returns a one-dimensional array (same as observation size).
    :param xb: The mean of the Gaussian Prior
    :param P: The covariance matrix of the Gaussian prior
    :param R: Observation noise covariance (assuming zero-mean Gaussian noise)
    :param y: The observational data
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
    def __init__(
        self,
        pom : Callable,
        xb: np.ndarray,
        P: np.ndarray,
        y: np.ndarray,
        R: np.ndarray,
        #
        alpha: float = 1e-3,
        beta: float = 2,
        kappa: float = 0,
    ) -> None:
        ## All assertions
        # Check types and instantiate
        if not callable(pom):
            # We can try it on the prior mean if needed
            raise TypeError(
                f"The parameter-to-observable map must be a callable; "
                f"received {pom=} of {type(pom)=}"
            )

        for _ in [xb, y]:
            if not isinstance(_, np.ndarray):
                raise TypeError(f"Expected numpy array; received{type(_)}")
            if not np.ndim(_) == 1 or _.size < 1:
                raise TypeError(f"Expected 1d numpy array; received array of shape {_.shape}")

        for _ in [P, R]:
            if not isinstance(_, np.ndarray):
                raise TypeError(f"Expected numpy array; received{type(_)}")

        if P.shape != (xb.size, xb.size):
            raise TypeError(
                f"Expected P of shape ({xb.size}, {xb.size})"
                f"received array of shape {P.shape=}"
            )
        if R.shape != (y.size, y.size):
            raise TypeError(
                f"Expected R of shape ({y.size}, {y.size})"
                f"received array of shape {R.shape=}"
            )

        if not isinstance(alpha, Number):
            raise TypeError(
                "alpha must be number; received {alpha=} of {type(alpha)=}"
            )
        if not isinstance(beta, Number):
            raise TypeError(
                "beta must be number; received {beta=} of {type(beta)=}"
            )
        if not isinstance(kappa, Number):
            raise TypeError(
                "kappa must be number; received {kappa=} of {type(kappa)=}"
            )

        ## Assign to attributes
        #
        self.pom = pom
        #
        self.xb = xb
        self.P = P
        self._P_SQRT= None
        self.R = R
        self.y = y
        #
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Place-holders for sigma points and their weights
        self._SIGMA_POINTS = self._SIGMA_POINTS_WEIGHTS = None

    def _create_sigma_points(self, ):
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

        :returns:
            - `sigma_points`: 2d array holding sigma points (as rows)
            - `mean_weights`: a tuple `(w0, wi)` where `w0` is the weight given to the first
                sigma point (the mean), and `wi` is associated with all other sigma points.
            - `covariance_weights`: same as `mean_weights`; this is a tuple `(w0, wi)` where
                `w0` is the weight given to the first sigma point (the mean), and
                `wi` is associated with all other sigma points.
        """
        ## Parameters and Configurations
        alpha = self.alpha
        beta = self.beta
        kappa = self.kappa

        # Dimension and number of sigma points
        L = self.xb.size
        _lambda = alpha**2 * (L + kappa) - L
        scl = np.sqrt(L+_lambda)  # scalar multiplied by Cholesky factor of the covariance
        num_sigma_points = 2 * L + 1

        ## Generate the sigma points

        # Extract prior mean (as numpy array)
        xb= self.xb.copy()

        # Placeholder for all sigma points (with mean set in all column)
        sigma_points = np.vstack(
            [xb]*num_sigma_points
        ).T
        if sigma_points.ndim == 1:
            sigma_points = sigma_points.reshape((sigma_points.size, 1))

        # First L sigma points (add to mean)
        sigma_points[:, 1: L+1] += scl * self.P_Stdev.T

        # Second L sigma points (subtract from mean)
        sigma_points[:, L+1: ] -= scl * self.P_Stdev.T

        # Transpose so that each row corresponds to a sigma point
        sigma_points = sigma_points.T

        ## Evaluate the weights of the sigma points
        # Weights for evaluating the mean
        # mean_weights = (
        #     _lambda / (L + _lambda),
        #     1.0 / ( 2.0 * (L+_lambda) )
        # )
        a2k = alpha ** 2 * kappa
        mean_weights = (
            (a2k - L) / a2k,
            (2*L + 1) / (2 * a2k)
        )

        # Covariance weights
        # covariance_weights = (
        #     _lambda / (L + _lambda) + (1 - alpha**2 + beta),
        #     1.0 / ( 2.0 * (L+_lambda) )
        # )
        covariance_weights = (
            (a2k - L) / a2k + 1 - alpha**2 + beta,
            (2*L + 1) / (2 * a2k)
        )

        print(f"{mean_weights=} \n {covariance_weights=}\n")

        # Verify mean and covariance from sigma points
        v_mean = mean_weights[0] * sigma_points[0]
        for s in sigma_points[1: ]:
            v_mean += mean_weights[1] * s
        s_mean = np.mean(sigma_points, axis=0)
        v_cov = np.zeros((s_mean.size, s_mean.size), dtype=sigma_points.dtype)
        for s in sigma_points[1: ]:
            innov = s - s_mean
            v_cov += covariance_weights[1] * np.outer(innov,  innov)

        print("Mean Validation: ", xb, "\n**\n", v_mean, "\n\n")
        print("Covariance Validation: ", self.P, "\n**\n", v_cov, "\n\n")

        print(
            f"{scl=}\n"
            f"{_lambda=}; {L=}; {alpha=}; {beta=}; {kappa=}\n"
            f"{self.P_Stdev=}\n"
            f"{sigma_points=}\n"
            f"{mean_weights=}\n"
            f"{mean_weights=}\n"
            f"{covariance_weights=}\n"
            f"{covariance_weights=}\n********\n"
        )

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
        # Observations equivalent to the sigma points (and their mean)
        Z = np.vstack([self.pom(x) for x in self.sigma_points]).astype(self.xb.dtype)
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
        S = Z.T @ Z.conjugate() + self.R

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
        Pxz = Pxz.T @ Z.conjugate()

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # Posterior
        x = self.xb + K @ (self.y- z_mean)
        P = self.P - K @ S @ K.conjugate().T

        print(
            f"{self.sigma_points=}\n"
            f"{w0=}\n{wi=}\n"
            f"{self.y=} \n"
            f"{z_mean=} \n"
            f"{(self.y-z_mean)=} \n\n"
            f"{S=}\n"
            f"{Pxz=}\n"
            f"{K=}\n"
            f"{self.xb=} \n"
            f"xa={x} \n"
            f"{self.P=} \n"
            f"Pa={P} \n"
        )
        raise IOError
        return x, P

    def solve(self, ) -> (np.ndarray, np.ndarray):
        """
        Solve the Bayesian inverse (filtering) problem and return the
        posterior mean and covariance (Gaussian approximation).

        .. note::
            This currently returns the output of :py:meth:`_analysis`.
        """
        return self._analysis()

    @property
    def P_Stdev(self):
        if self._P_SQRT is None:
            self._P_SQRT = np.linalg.cholesky(self.P, upper=False)
        return self._P_SQRT

    @property
    def parameter_to_observable_map(self):
        """The registered parameter-to-observable map (forward operator)"""
        return self.pom

    @property
    def size(self):
        """
        Dimension of the state space (inference variable).
        This is extracted from the registered prior mean
        """
        return self.xb.size

    @property
    def sigma_points(self):
        """
        Sigma points. If not created, they are instantiated
        """
        if self._SIGMA_POINTS is None or self._SIGMA_POINTS is None:
            self._SIGMA_POINTS, self._SIGMA_POINTS_WEIGHTS = self._create_sigma_points()
        return self._SIGMA_POINTS

    @property
    def sigma_points_weights(self):
        """
        Sigma points weights. If not created, they are instantiated
        """
        if self._SIGMA_POINTS is None or self._SIGMA_POINTS is None:
            self._SIGMA_POINTS, self._SIGMA_POINTS_WEIGHTS = self._create_sigma_points()
        return self._SIGMA_POINTS_WEIGHTS



class _UnscentedKalmanFilter(Filter):
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

    def __init__(
        self,
        pom : Callable, prior: GaussianNoise,
        observation_noise: GaussianNoise,
        observation: np.ndarray,
        alpha: float = 1e-3, beta: float = 2, kappa: float = 0
    ) -> None:
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
        elif isinstance(observation, Wave):
            self._OBSERVATION = observation.form
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
                f"Found {self.prior.mean.shape=}"
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

        # TODO: After discussing localizing the POM, proceed with this...
        # TODO: Figure out eneregy mismatch since `x` could basically be a numpy array!
        # TODO: Define a function to apply POM given that the energy need to match POM


        # Observations
        Z = np.vstack(
            [
                self.parameter_to_observable_map(Wave(x))
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

    def __init__(
        self, pom : Callable,
        prior: GaussianNoise,
        observation_noise: GaussianNoise,
        observation: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2,
        kappa: float = 0,
    ) -> None:
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

    def solve(self, ) -> GaussianNoise:
        """
        Solve the Bayesian inverse (filtering) problem and return the posterior.
        """
        ## Initiate the posterior (copy from the prior)
        posterior = self.prior.copy()

        ## Extract prior mean
        prior_mean = self.prior.mean.form
        if prior_mean.ndim == 3:
            prior_mean = prior_mean[0, :]
        if prior_mean.ndim != 2:
            raise TypeError(f"Unexpected {prior_mean.shape=}")

        # prior mean and observations as 1d numpy arrays
        global_mean = prior_mean.flatten()
        global_observation = self.observation.flatten()

        # Coordinates with non-zero covariances
        PCov = self.prior.covariance.covariance_array
        row_coord, col_coord = self.prior.covariance.coord
        covariance_data = self.prior.covariance.data

        ## Local POM based on the full POM
        def local_pom(x, nonzero_locs):
            moving_x = self.prior.mean.zeros_like()
            for i, ind in enumerate(nonzero_locs):
                x_ind = ind // moving_x.shape[-1]
                y_ind = i % moving_x.shape[-1]
                moving_x.form[:, x_ind, y_ind] = x[i]

            y = self.pom(moving_x)
            if isinstance(y, Wave): y = y.form
            if y.ndim == 3:
                y = y[0, ...].ravel()[nonzero_locs]
            elif y.ndim == 2:
                y = y.ravel()[nonzero_locs]
            else:
                raise TypeError(f"Unexpected {y.shape=}")
            return y

        # For each entry, get the corresponding pixel and the ones correlated to it
        for i in range(posterior.size):
            # Indexes which cell i is correlated with
            nonzero_locs = col_coord[np.where(row_coord==i)[0]]
            nonzero_locs = nonzero_locs[nonzero_locs>=i]

            ## Define Gaussian prior for this nonzero locations
            # Extract mean and covariance of the nonzero locations
            xb = global_mean[nonzero_locs]
            P = PCov[nonzero_locs, :][:, nonzero_locs].toarray()

            ## Observation information for nonzero locations
            y = global_observation[nonzero_locs]
            R = self.observation_noise.covariance.covariance_array[nonzero_locs, :][:, nonzero_locs].toarray()

            ## Apply UKF locally for each entry of the mean
            xa, Pa = UnscentedKalmanFilter(
                pom=lambda x: local_pom(x, nonzero_locs),
                xb=xb,
                P=P,
                y=y,
                R=R,
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa,
            ).solve()

            # TODO: Fix mean update index (for 3d...)
            # Update ith mean entry
            x_ind = i // posterior.covariance.waveform_shape[-1]
            y_ind = i % posterior.covariance.waveform_shape[-1]
            print(i, xb, y, xa[0], Pa[0, :])
            if posterior.mean.form.ndim==2:
                posterior.mean.form[x_ind, y_ind] = xa[0]
            elif posterior.mean.form.ndim==3:
                posterior.mean.form[:, x_ind, y_ind] = xa[0]
            else:
                raise TypeError(
                    f"Unexpected {posterior.mean.form.shape=}"
                )

            # Update ith row column
            posterior.covariance.covariance_array[[i], nonzero_locs] = Pa[0, :]
            posterior.covariance.covariance_array[nonzero_locs, [i]] = Pa[:, 0]

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


