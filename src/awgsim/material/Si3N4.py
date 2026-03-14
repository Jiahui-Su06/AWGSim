import warnings
import numpy as np
from numpy.typing import ArrayLike


def Si3N4(
    x: ArrayLike
) -> np.ndarray | float:
    """
    Sellmeier material model for Si3N4 at 20 C.

    Valid nominal wavelength range:
    - 0.31 um to 5.504 um

    Parameters
    ----------
    x : ArrayLike
        Wavelength in um.

    Returns
    -------
    np.ndarray | float
        Refractive index of Si3N4.
    """
    x_arr = np.asarray(x, dtype=float)

    if np.any((x_arr < 0.31) | (x_arr > 5.504)):
        warnings.warn(
            "Extrapolating Sellmeier equation for Si3N4 beyond range of 0.31 um - 5.504 um",
            stacklevel=2,
        )

    n = np.sqrt(
        1
        + 3.0249 / (1 - (0.1353406 / x_arr) ** 2)
        + 40314 / (1 - (1239.842 / x_arr) ** 2)
    )

    if np.asarray(x).ndim == 0:
        return float(np.asarray(n))
    return n
