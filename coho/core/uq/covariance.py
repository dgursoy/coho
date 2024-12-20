"""
UQ-type Objects & algorithms.
This module is experimental and will be moved once mature enough
"""

# Standard imports
from numbers import Number
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from abc import ABC, abstractmethod, abstractproperty
from typing import Type, Tuple, Callable, Union
from itertools import product

try:
    from sksparse.cholmod import cholesky as sp_cholesky
except ImportError:
    sp_cholesky = None

# Local imports
from ..operator import Operator
from ..component import Wave

__all__ = [
    'Covariance',
    'SparseCovariance',
    'SparseCovarianceLocalization',
    'DiagonalCovariance',
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
                 data: np.ndarray, coord: Tuple[np.ndarray, np.ndarray],
                 dtype: Type = np.complex128,
                 format: str = "csc", ):
        """
        Create a covariance operator where each image pixel is only correlated
        with neighboring pixels.

        :param waveform_shape: wave form shape (nx, ny)
        :param data: 1d data array holding nonzero elements.
        :param coord: coordinates `(row, col)` to be filled with with `data`.
        :param dtype: data type of the stored covariances data
        :param format: Matrix format of the result.
            By default "csc" is chosen to allow easy matrix operations.
            Allowed formats are:
            - "csc": calls scipy.sparse.csc_array
            - "csr": calls scipy.sparse.csr_array
            - "coo": calls scipy.sparse.coo_array

        .. note::
            Other formats such as "bsr" for Block sparse row format,
            "lil" for Row-based LIst of list sparse array, or
            "dia" for DIAgonal storage can be provided by other derived classes
            since their interface might be a little different.
        """
        # Check dimensionality and data types
        nx, ny = waveform_shape
        assert int(nx)==nx and nx>0 and int(ny)==ny and ny>0, \
            f"Unexpected shape {waveform_shape=} of {type(waveform_shape)=}"
        size = nx * ny
        assert isinstance(data, np.ndarray), f"Expected np.ndarray; received {type(data)=}"
        assert isinstance(coord, (tuple, np.ndarray)), f"Expected tuple (or np.ndarray for dia); received {type(coord)=}"

        # Create the *sparse* covariance array based on the passed format
        sparse_generator = self._map_format_to_sparse_generator(format)
        self._COVARIANCE_ARRAY = sparse_generator(
            (data, coord),
            shape=(size, size),
            dtype=dtype,
        )

        # Save waveform shape (image dimensions nx, ny)
        self._WAVEFORM_SHAPE = waveform_shape
        ## Initialization done...

    def _map_format_to_sparse_generator(self, format):
        """
        Map the passed format the right function/creator.
        This is not intended to be called by the user.
        """
        if isinstance(format, str):
            match format.lower().strip():
                case "csc":
                    return sp.csc_array

                case "csr":
                    return sp.csr_array

                case "coo":
                    return sp.coo_array

                case _:
                    raise ValueError(
                        f"Unsupported format '{format}'! "
                        f"Only 'csc', 'csr', 'coo' are currently supported"
                    )

        else:
            raise TypeError(
                f"Expected format to be string; received '{type(format)}'"
            )

    def apply(self, wave: Wave | np.ndarray, in_place:bool=True, ) -> Wave | np.ndarray:
        """
        Appy a covariance operator (matrix-vector product) to 2d array or
        a wave form.

        :param wave: the wave (or waveform/image) to apply the covariance operator to.
            The shape of the waveform is either 2D or 3D.
            If it is 2D, the shape must be equal to the registered `waveform_shape`.
            If it is 3D, the shape must by `(n, nx, ny)` where `(nx, ny)`
            is the registered `waveform_shape`.is the method method first axis is expected
        :param in_place: either overwrite the passed object, or return a copy
            holding the results
        """
        # Create a copy if not `in_place`
        wave = wave if in_place else wave.copy()

        # Extract a reference to the waveform (numpy array) to work with
        if isinstance(wave, np.ndarray):
            form = wave
        elif isinstance(wave, Wave):
            form = wave.form
        else:
            raise TypeError(
                f"Expected Wave or 2D array. Received {wave=} of {type(wave)=}"
            )

        if form.shape == self.waveform_shape:
            # Write in place matrix-vector product
            form[...] = (
                self.covariance_array @ form.ravel()
            ).reshape(form.shape)

        elif len(form.shape) == 3 and form.shape[1: ] == self.waveform_shape:
            for j in range(form.shape[0]):
                form[j, ...] = (
                self.covariance_array @ form[j, ...].ravel()
            ).reshape(form.shape)

        else:
            raise TypeError(
                f"Invalid wave front shape {form.shape}; "
                f"expected ({self.waveform_shape}) or "
                f"(n, {self.waveform_shape[0], self.waveform_shape[1]}) "
                f"for some positive integer n"
            )

        return wave

    @property
    def shape(self):
        """Shape of the covariance matrix"""
        return (self.size, self.size)

    @property
    def waveform_shape(self):
        """Shape of the waveform (image dimensions nx x ny ) """
        return self._WAVEFORM_SHAPE

    @property
    def size(self):
        """
        Dimension of the probability space.
        This is the total number of pixels in an image (waveform)
        """
        nx, ny = self.waveform_shape
        return nx * ny

    def nonzero(self):
        """
        Nonzero indices of the array/matrix.

        Returns a tuple of arrays (row,col) containing the indices
        of the non-zero elements of the array.
        """
        return self.covariance_array.nonzero()

    @property
    def coord(self):
        """
        Coordinates of nonzero entries.
        Wrapper around :py:meth:`nonzero`
        """
        return self.nonzero()

    @property
    def format(self):
        """
        Format used to store the covariance matrix.
        """
        return self.covariance_array.format

    @property
    def dtype(self):
        """
        Data type of entris of the covariance matrix.
        """
        return self.covariance_array.dtype

    @property
    def data(self):
        """
        Data held in nonzero entris of the arrays. This is a 1D array that stores values at coordinates
        retured by :py:meth:`nonzero`.
        """
        return self.covariance_array.data

    @property
    def covariance_array(self):
        """Reference to the sparse covariance array/matrix."""
        return self._COVARIANCE_ARRAY
    @covariance_array.setter
    def covariance_array(self, val):
        """Update the sparse covariance array/matrix."""
        assert sp.issparse(val) and val.shape == self.shape, f"Invalid {val=} of {type(val)=}"
        self._COVARIANCE_ARRAY = val

    def __str__(self) -> str:
        """Simple string representation."""
        return (
            f"Sparse (Scipy-Based) covariance matrix/kernel "
            f"'SparseCovariance' of shape {self.shape} "
            f"for waveform of shape {self.waveform_shape}"
        )

    def copy(self) -> 'SparseCovariance':
        """Create a copy of the SparseCovariance operator with same properties and data."""
        return SparseCovariance(
            waveform_shape=self.waveform_shape,
            data=self.data,
            coord=self.coord,
            format=self.format,
            dtype=self.dtype,
        )

    def cholesky(self, in_place : bool = False,
                 lower : bool = True, ) -> 'SparseCovariance':
        """
        Evaluate the Cholesky factor of the covariance matrix where
        :math:`A=LL*`.
        Since scipy.sparse.linalg does not provide Cholesky factorization for
        positive semi definite matrices, we wither have to rely on LU decomposition,
        or use alternative packages such as scikit-sparse.
        Here, I am following the former.

        :param in_place: either overwrite the underlying of the covariance matrix
            or return a copy.
        :param lower: if `True` return the lower Cholesky factor, otherwise
        return the upper factor.
            or return a copy.
        """
        # Copy if needed
        cov = self if in_place else self.copy()

        # Sparse LU factorization
        if sp_cholesky is not None:
            factor = sp_cholesky(cov.covariance_array)
            std = factor.L()
            std @= std.conjugate().T
            pass
        else:
            try:
                LU = splinalg.splu(cov.covariance_array, diag_pivot_thresh=0, permc_spec="NATURAL")
            except Exception as err:
                print(
                    f"Cholesky factorization failed!\n"
                    f"Failed to use efficient LU factorization for sparse matrices; \n"
                    f"Unexpected {err=} of {type(err)=}"
                )
                raise

            else:
                # Check the matrix is positive semi definite:
                if any(LU.perm_r != np.arange(cov.size)) or any(
                    LU.U.diagonal() < 0
                ):
                    # Compose error message and raise
                    msg = f"Cholesky factorization failed!\n"
                    msg += f"Failed to use efficient LU factorization for sparse matrices; \n"
                    msg += f"{LU.perm_r=}; \n"
                    msg += f"{LU.U.diagonal()=}; \n"
                    msg += f"{any(LU.perm_r != np.arange(cov.size))=};\n"
                    msg += f"{any(LU.U.diagonal() < 0)=}"
                    raise TypeError(msg)

                else:
                    # Calculate the lower Cholesky factor
                    std = LU.L @ (sp.diags(LU.U.diagonal() ** 0.5))

                    # Transpose if upper factor is needed
                    if not lower: std = std.T

                    # Convert format if needed
                    if std.format != cov.format:
                        if cov.format == "csc":
                            std = std.tocsc()
                        elif cov.format == "csr":
                            std = std.tocsc()
                        elif cov.format == "coo":
                            std = std.tocoo()
                        else:
                            raise TypeError(
                                f"Unexpected format of the covariance {cov.format=}"
                            )

        # Apply the factorization
        cov._COVARIANCE_ARRAY = std

        return cov

    def __repr__(self) -> str:
        """String representation of the covariance operator."""
        return (f"SparseCovariance"
            f"waveform_shape={self.waveform_shape!r}, "
            f"dtype={self.covariance_array.dtype!r}, "
            f"with data stored as {repr(self.covariance_array)!r})"
        )

    def __matmul__(self, other: Union['SparseCovariance', float, int, complex, np.ndarray, sp.sparray]) -> 'SparseCovariance':
        """Matrix (left) multiplication of covariance operators; self @ other """
        if isinstance(other, Number):
            # Left multiplication (self @ other) where other is a number
            cov = self.data * other  # Scipy doesn't allow it but we do
            return SparseCovariance(
                waveform_shape=self.waveform_shape,
                data=cov,
                coord=self.coord,
                dtype=cov.dtype,
            )

        elif type(other) is type(self):
            if self.shape[1] !=other.shape[0]:
                raise TypeError(
                    f"Inconsistent shapes {self.shape} != {other.shape}"
                )
            if not (self.waveform_shape == other.waveform_shape or self.waveform_shape[::-1] == other.waveform_shape):
                warnings.warn(
                    f"multiplication of covariances associated with waveforms of different shapes might be unpredictible."
                )
            cov = self.covariance_array @ other.covariance_array
            return SparseCovariance(
                waveform_shape=self.waveform_shape,
                data=cov.data,
                coord=cov.nonzero(),
                dtype=cov.dtype,
            )

        elif isinstance(other, np.ndarray) or sp.issparse(other):
            return self.covariance_array @ other

        else:
            raise TypeError(
                f"Expected scalar, numpy/scipy array, or SparseCovariance isntance; received {type(other)}"
            )

    def __rmatmul__(self, other: Union['SparseCovariance', Number, np.ndarray, sp.sparray]) -> 'SparseCovariance':
        """Matrix (right) multiplication of covariance operators; other @ self """
        if isinstance(other, Number):
            # Left multiplication (self @ other) where other is a number
            cov = self.data * other  # Scipy doesn't allow it but we do
            return SparseCovariance(
                waveform_shape=self.waveform_shape,
                data=cov,
                coord=self.coord,
                dtype=cov.dtype,
            )

        elif type(other) is type(self):
            return other.__matmul__(self)

        elif isinstance(other, np.ndarray) or sp.issparse(other):
            return other @ self.covariance_array

        else:
            raise TypeError(
                f"Expected scalar, numpy/scipy array, or SparseCovariance isntance; received {type(other)}"
            )

    def __mul__(self, other: Union['SparseCovariance', float, int, complex]) -> 'SparseCovariance':
        """Pointwise (left) multiplication of covariance operators; self * other """

        if isinstance(other, Number):
            cov = self.data * other  # Let scipy handle this
            # Left multiplication (self * other) where other is a number
            return SparseCovariance(
                waveform_shape=self.waveform_shape,
                data=cov,
                coord=self.coord,
                dtype=cov.dtype,
            )

        elif type(other) is type(self):
            if self.shape !=other.shape:
                raise TypeError(
                    f"Inconsistent shapes {self.shape} != {other.shape}"
                )
            if self.waveform_shape != other.waveform_shape:
                warnings.warn(
                    f"Pointwise multiplication of two covariances associated with waveforms of different shapes."
                )
            cov = self.covariance_array * other.covariance_array
            return SparseCovariance(
                waveform_shape=self.waveform_shape,
                data=cov.data,
                coord=cov.nonzero(),
                dtype=cov.dtype,
            )

        else:
            raise TypeError(
                f"Expected scalar or SparseCovariance isntance; received {type(other)}"
            )

    def __rmul__(self, other: Union['SparseCovariance', float, int, complex]) -> 'SparseCovariance':
        """Pointwise (right) multiplication of covariance operators; other * self """
        return self.__mul__(other)

    def __imul__(self, other: Union['SparseCovariance', float, int, complex]) -> 'SparseCovariance':
        """In-place pointwise multiplication."""
        if isinstance(other, Number):
            self.covariance_array.data *= other

        elif type(other) is type(self):

            if self.shape !=other.shape:
                raise TypeError(
                    f"Inconsistent shapes {self.shape} != {other.shape}"
                )
            if self.waveform_shape != other.waveform_shape:
                warnings.warn(
                    f"Pointwise multiplication of two covariances associated with waveforms of different shapes."
                )

            # In place multiplication of underlying covariance array
            self.covariance_array *= other.covariance_array


        else:
            raise TypeError(
                f"Expected scalar or SparseCovariance isntance; received {type(other)}"
            )

        return self

    def __add__(self, other: Union['SparseCovariance', float, int, complex]) -> 'SparseCovariance':
        """Left addition of covariance operator with other covariance operator or with scalar; self + other """

        if isinstance(other, Number):
            raise NotImplementedError(
                f"Adding a nonzero scalar to a sparse covariance array is not supported"
            )

        elif type(other) is type(self):
            if self.shape !=other.shape:
                raise TypeError(
                    f"Inconsistent shapes {self.shape} != {other.shape}"
                )
            if self.waveform_shape != other.waveform_shape:
                warnings.warn(
                    f"Addition of two covariances associated with waveforms of different shapes."
                )
            cov = self.covariance_array + other.covariance_array
            return SparseCovariance(
                waveform_shape=self.waveform_shape,
                data=cov.data,
                coord=cov.nonzero(),
                dtype=cov.dtype,
            )

        else:
            raise TypeError(
                f"Expected scalar or SparseCovariance isntance; received {type(other)}"
            )

    def __radd__(self, other: Union['SparseCovariance', float, int, complex]) -> 'SparseCovariance':
        """Right addition of covariance operator with other covariance operator or with scalar; other + self"""
        return self.__add__(other)

    def __iadd__(self, other: Union['SparseCovariance', float, int, complex]) -> 'SparseCovariance':
        """In-place addition."""
        if isinstance(other, Number):
            raise NotImplementedError(
                f"Adding a nonzero scalar to a sparse covariance array is not supported"
            )

        elif type(other) is type(self):
            if self.shape !=other.shape:
                raise TypeError(
                    f"Inconsistent shapes {self.shape} != {other.shape}"
                )
            if self.waveform_shape != other.waveform_shape:
                warnings.warn(
                    f"Addition of two covariances associated with waveforms of different shapes."
                )

            # In place addition of underlying covariance array
            self._COVARIANCE_ARRAY += other.covariance_array

        else:
            raise TypeError(
                f"Expected scalar or SparseCovariance isntance; received {type(other)}"
            )
        return self

    def __sub__(self, other: Union['SparseCovariance', float, int, complex]) -> 'SparseCovariance':
        """Subtraction of covariance operators self - other """

        if type(other) is type(self):
            if self.shape !=other.shape:
                raise TypeError(
                    f"Inconsistent shapes {self.shape} != {other.shape}"
                )
            if self.waveform_shape != other.waveform_shape:
                warnings.warn(
                    f"Addition of two covariances associated with waveforms of different shapes."
                )
            cov = self.covariance_array - other.covariance_array
            return SparseCovariance(
                waveform_shape=self.waveform_shape,
                data=cov.data,
                coord=cov.nonzero(),
                dtype=cov.dtype,
            )

        else:
            raise TypeError(
                f"Expected SparseCovariance isntance; received {type(other)}"
            )

    def __isub__(self, other: Union['SparseCovariance', float, int, complex]) -> 'SparseCovariance':
        """In-place subtraction."""
        if type(other) is type(self):
            if self.shape !=other.shape:
                raise TypeError(
                    f"Inconsistent shapes {self.shape} != {other.shape}"
                )
            if self.waveform_shape != other.waveform_shape:
                warnings.warn(
                    f"Subtraction of two covariances associated with waveforms of different shapes."
                )

            # In place addition of underlying covariance array
            self._COVARIANCE_ARRAY -= other.covariance_array

        else:
            raise TypeError(
                f"Expected SparseCovariance isntance; received {type(other)}"
            )

        return self


class DiagonalCovariance(SparseCovariance):
    """Covariance matrix with only diagonal entries (no correlations)"""

    def __init__(self, waveform_shape: Tuple[int, int],
                 data: np.ndarray,
                 dtype: Type = np.complex128,
                 format: str = "csc"):
        """
        Create a covariance operator where each image pixel is not correlated with any other pixel.

        :param waveform_shape: wave form shape (nx, ny)
        :param data: scalar or 1d data array holding non-negative elements elements
            to be added to the diagonal of the covariance matrix.
            This typically represents the variances of all pixels.
        :param dtype: data type of the stored covariances data
        :param format: Matrix format of the result (default "csc").
            See the formats supported by :py:class:`SparseCovariance`
        """
        nx, ny = waveform_shape
        size = nx * ny
        assert int(nx)==nx and nx>0 and int(ny)==ny and ny>0, \
            f"Unexpected shape {waveform_shape=} of {type(waveform_shape)=}"
        if isinstance(data, Number):
            data = np.atleast_1d(data)
        elif isinstance(data, np.ndarray):
            if (np.ndim(data) == 1 and data.size in [1, size]):
                data = np.atleast_1d(data)
            else:
                raise TypeError(
                    f"Expected np.ndarray of size 1 (scalar) or {size}; "
                    f"received array with dimensions {np.ndim(data)} "
                    f"and size {data.size}"
                )
        else:
            raise TypeError(
                f"Expected scalar or np.ndarray; received {type(data)=}"
            )

        # Convert scalar data to 1d array
        if data.size == 1 < size:
            data = data.repeat(size)

        return super().__init__(
            waveform_shape=waveform_shape,
            coord=(range(size), range(size)),
            data=data,
            format=format,
            dtype=dtype,
        )


class SparseCovarianceLocalization(SparseCovariance):
    """
    Scipy-based sparse implementation of a covariance localization operator.
    An operator like this is block diagonal by nature as it correlates each pixel
    only to neighboring pixels.
    """

    def __init__(self, waveform_shape: Tuple[int, int],
                 step_size: int, localization_function: Union[None, Callable]=None,
                 dtype: Type = np.complex128,
                 format: str = "csc"):
        """
        Create a covariance operator where each image pixel is only correlated
        with neighboring pixels.

        :param waveform_shape: wave form shape (nx, ny)
        :param step_size: size of the neighborhood in which correlations are
            preserved.
        :param localization_function: if not `None`, a callable that takes a distance
            (int representing distance between pixels here) and calculates
            localization factor (should be number in ``[0, 1]``).
        :param format: Matrix format of the result (default "csc").
            See the formats supported by :py:class:`SparseCovariance`
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
            format=format,
            dtype=dtype,
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

