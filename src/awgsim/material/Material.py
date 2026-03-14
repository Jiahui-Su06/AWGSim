import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable
from scipy.interpolate import Akima1DInterpolator, RegularGridInterpolator
import inspect


def _dispersion_curve(
    func: Callable[[ArrayLike], ArrayLike],
    lambda1: float,
    lambda2: float,
    points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a function over a wavelength range.
    Returns (values, wavelength).
    """
    wavelength = np.linspace(lambda1, lambda2, points)
    values = np.asarray(func(wavelength))
    return values, wavelength


class Material:
    """
    Material chromatic dispersion model.

    If refractive index is complex, the imaginary part is the extinction
    coefficient kappa such that n = n + 1j*kappa produces a power loss
    coefficient alpha in exp(-alpha*z), where alpha = 4*pi*kappa/lambda0.

    Accepted model types
    --------------------
    - Material: another Material object
    - callable: function like f(lambda0) or f(lambda0, T)
    - scalar: constant index
    - 1D array: polynomial coefficients (numpy.polyval convention)
    - 2D array: lookup table with wavelength in column 1, index in column 2
    - dict: with keys {'wavelength', 'index'} and optional 'temperature'
    """

    def __init__(self, model):
        if model is None:
            raise ValueError("A material model must be provided to Material(model).")

        if isinstance(model, Material):
            self._type = model._type
            self._model = model._model
            return

        if callable(model):
            sig = inspect.signature(model)
            n_in = len(sig.parameters)
            if n_in < 1:
                raise ValueError(
                    "Invalid model: callable must accept at least 1 input argument."
                )
            self._type = "function"
            self._model = model
            return

        if isinstance(model, dict):
            if "wavelength" not in model:
                raise ValueError('Data model must contain a field named "wavelength".')
            if "index" not in model:
                raise ValueError('Data model must contain a field named "index".')

            wavelength = np.asarray(model["wavelength"])
            index = np.asarray(model["index"])

            if "temperature" in model:
                temperature = np.asarray(model["temperature"])
                expected_shape = (len(wavelength), len(temperature))
                if index.shape != expected_shape:
                    raise ValueError(
                        "Data provided is of the wrong dimensions for interpolation."
                    )
            else:
                if index.shape[0] != len(wavelength):
                    raise ValueError("Data provided must be the same length.")

            self._type = "lookup"
            self._model = model
            return

        arr = np.asarray(model)

        if arr.ndim == 0:
            self._type = "constant"
            self._model = arr.item()
            return

        if arr.ndim == 1:
            self._type = "polynomial"
            self._model = arr
            return

        if arr.ndim == 2:
            nr, nc = arr.shape
            if nc > nr:
                arr = arr.T
            if arr.shape[1] != 2:
                raise ValueError(
                    "Invalid model: lookup data must be a 2-column matrix "
                    "with wavelength in column 1 and index in column 2."
                )
            self._type = "lookup"
            self._model = arr
            return

        raise ValueError(
            "Invalid model argument provided for Material(model); "
            "see documentation for acceptable inputs."
        )

    def index(self, lmbda: ArrayLike, T: float = 295) -> np.ndarray | complex | float:
        """
        Calculate refractive index at given wavelength and temperature.
        """
        lmbda_arr = np.asarray(lmbda)

        if self._type == "constant":
            if lmbda_arr.ndim == 0:
                return self._model
            return np.full_like(lmbda_arr, self._model, dtype=np.asarray(self._model).dtype)

        if self._type == "function":
            sig = inspect.signature(self._model)
            if len(sig.parameters) > 1:
                return self._model(lmbda, T)
            return self._model(lmbda)

        if self._type == "polynomial":
            return np.polyval(self._model, lmbda)

        # lookup cases
        if isinstance(self._model, dict):
            wavelength = np.asarray(self._model["wavelength"])
            index = np.asarray(self._model["index"])

            if "temperature" in self._model:
                temperature = np.asarray(self._model["temperature"])

                # RegularGridInterpolator handles 2D interpolation.
                interp = RegularGridInterpolator(
                    (wavelength, temperature),
                    index,
                    bounds_error=False,
                    fill_value=None,
                )

                lam_eval = np.atleast_1d(lmbda_arr)
                temp_eval = np.full_like(lam_eval, T, dtype=float)
                pts = np.column_stack((lam_eval, temp_eval))
                out = interp(pts)

                if np.asarray(lmbda).ndim == 0:
                    return out[0]
                return out

            interp = Akima1DInterpolator(wavelength, index)
            out = interp(lmbda_arr)

            if np.asarray(lmbda).ndim == 0:
                return np.asarray(out).item()
            return np.asarray(out)

        wavelength = np.asarray(self._model[:, 0])
        index = np.asarray(self._model[:, 1])

        interp = Akima1DInterpolator(wavelength, index)
        out = interp(lmbda_arr)

        if np.asarray(lmbda).ndim == 0:
            return np.asarray(out).item()
        return np.asarray(out)

    def dispersion(
        self,
        lambda1: float,
        lambda2: float,
        points: int = 1000,
        T: float = 295,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute chromatic dispersion curve over a wavelength range.
        Returns (n, wavelength).
        """
        return _dispersion_curve(lambda x: self.index(x, T), lambda1, lambda2, points)

    def groupindex(
        self,
        lmbda: ArrayLike,
        T: float = 295,
    ) -> np.ndarray | complex | float:
        """
        Compute average group index around wavelength lmbda using
        a finite-difference approximation.
        """
        lmbda_arr = np.asarray(lmbda)

        n0 = self.index(lmbda_arr, T)
        n1 = self.index(lmbda_arr - 0.1, T)
        n2 = self.index(lmbda_arr + 0.1, T)

        ng = n0 - lmbda_arr * (n2 - n1) / 0.2

        if np.asarray(lmbda).ndim == 0:
            return np.asarray(ng).item()
        return np.asarray(ng)

    def group_dispersion(
        self,
        lambda1: float,
        lambda2: float,
        points: int = 1000,
        T: float = 295,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute group index curve over a wavelength range.
        Returns (Ng, wavelength).
        """
        return _dispersion_curve(lambda x: self.groupindex(x, T), lambda1, lambda2, points)
    