"""
UQ-type Objects & algorithms.
This module is experimental and will be moved once mature enough
"""

# Standard imports
import numpy as np
import scipy.sparse as sp
from abc import ABC, abstractmethod
from typing import Tuple, Callable, Union
from itertools import product

# Local imports
from ..operator import Operator
from ..component import Wave

__all__ = [
    'Covariance',
    'SparseCovariance',
    'SparseCovarianceLocalization',
]


class Covariance(ABC):
    """Base class for Covariance Operators"""

    @abstractmethod
    def apply(self, wave: Wave | np.ndarray) -> Wave | np.ndarray:
        """
        Appy a covariance operator (matrix-vector product) to 2d array or
        a wave form.
        The return type is the same as the input type.

        .. note::
            This results in applying a convolution mask centered at each
            pixel with weights equal to covariances with other neighboring pixels.
        """
        ...


class SparseCovariance(Covariance):
    """Scipy-based sparse implementation of a covariance operator"""

    def __init__(self, waveform_shape: Tuple[int, int],
                 data: np.ndarray, coord: Tuple[np.ndarray, np.ndarray]):
        """
        Create a covariance operator where each image pixel is only correlated
        with neighboring pixels.

        :param waveform_shape: wave form shape (nx, ny)
        :param data: 1d data array holding nonzero elements.
        :param coord: coordinates `(row, col)` to be filled with with `data`.
        """
        nx, ny = waveform_shape
        assert int(nx)==nx and nx>0 and int(ny)==ny and ny>0, f"Unexpected shape {waveform_shape=} of {type(waveform_shape)=}"
        assert isinstance(data, np.ndarray), f"Expected np.ndarray; received {type(data)=}"
        assert isinstance(coord, tuple), f"Expected tuple; received {type(coord)=}"

        self._SIZE = nx * ny
        self._WAVEFORM_SHAPE = waveform_shape
        self._DATA = sp.csc_array(
            (data, coord),
            shape=(self.size, self.size),
            dtype=float,
        )

    def apply(self, wave: Wave | np.ndarray) -> Wave | np.ndarray:
        """
        Appy a covariance operator (matrix-vector product) to 2d array or
        a wave form.
        The return type is the same as the input type.
        """
        if isinstance(wave, np.ndarray):
            form = wave
        elif isinstance(wave, Wave):
            form = wave.form
        else:
            raise TypeError(
                f"Expected Wave or 2D array. Received {wave=} of {type(wave)=}"
            )
        assert wave.shape == self.waveform_shape, f"Invalid wave front {wave.shape=}; expected ({self.waveform_shape=})"

        # Write in place matrix-vector product
        form[...] = (
            self.data @ form.ravel()
        ).reshape(form.shape)

        return wave

    @property
    def size(self):
        return self._SIZE

    @property
    def data(self):
        return self._DATA

    @property
    def shape(self):
        return (self.size, self.size)

    @property
    def waveform_shape(self):
        return self._WAVEFORM_SHAPE


class SparseCovarianceLocalization(SparseCovariance):
    """
    Scipy-based sparse implementation of a covariance localization operator
    """

    def __init__(self, waveform_shape: Tuple[int, int],
                 step_size: int, localization_function: Union[None, Callable]=None):
        """
        Create a covariance operator where each image pixel is only correlated
        with neighboring pixels.

        :param waveform_shape: wave form shape (nx, ny)
        :param step_size: size of the neighborhood in which correlations are
            preserved.
        :param localization_function: if not `None`, a callable that takes a distance
            (int representing distance between pixels here) and calculates
            localization factor (should be number in ``[0, 1]``).
        """
        ## Size (square matrix of shpae (size, size)) & Localization step size
        nx, ny = waveform_shape
        assert int(nx)==nx and nx>0 and int(ny)==ny and ny>0, f"Unexpected shape {waveform_shape=} of {type(waveform_shape)=}"
        self._SIZE = nx * ny
        self._WAVEFORM_SHAPE = waveform_shape

        assert int(step_size)==step_size and 0<=step_size<=self.size, f"Expected int; received {self.size=} of {type(step_size)=}"
        self._STEP_SIZE = step_size

        # Check localization function
        if callable(localization_function):
            self._LOCALIZATION_FUNCTION = localization_function
        elif localization_function is None:
            self._LOCALIZATION_FUNCTION = lambda x: np.atleast_1d(np.ones_like(x))
        else:
            raise TypeError(
                f"Expected None or callable; unexpected {type(localization_function)=}"
            )

        ## Generate coordinates and data arrays; two columns with x/y coordinates
        coordinates = []
        distances = []
        # Loop over indexes of the waveform pixels;
        for i in range(nx):
            for j in range(ny):
                # flattened index: Index in covariance matrix
                cov_i = i * nx + j
                _neighbors = self.neighborhood(i, j)
                _distances = self.distance(
                    first=[(i, j)]*len(_neighbors),
                    second=_neighbors
                )
                coordinates += [(cov_i, cov_j[0]*nx+cov_j[1]) for cov_j in _neighbors]
                distances += _distances

        coordinates = np.asarray(coordinates)
        distances = np.asarray(distances)

        data = self.localization_function(distances)
        return super().__init__(
            waveform_shape=waveform_shape,
            data=data,
            coord=(coordinates[:, 0].flatten(), coordinates[:, 1].flatten()),
        )

    def distance(self, first, second):
        """Calculate distance (euclidean) between two coordinate points"""
        return [
            np.sqrt(
                (i1-i2)**2 + (j1-j2)**2
            ) for (i1, j1), (i2, j2) in zip(first, second)
        ]

    def neighborhood(self, i, j):
        """find points in valid neighborhood of given coordinates based on step size"""
        neighborhood = []
        nx, ny = self.waveform_shape
        for x_coord in range(max(0, i-self.step_size), min(nx, i+self.step_size+1)):
            for y_coord in range(max(0, j-self.step_size), min(ny, j+self.step_size+1)):
                neighborhood.append((x_coord, y_coord))
        return neighborhood

    @property
    def step_size(self):
        return self._STEP_SIZE

    @property
    def localization_function(self):
        return self._LOCALIZATION_FUNCTION

