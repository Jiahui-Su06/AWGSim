import warnings
import numpy as np
from numpy.typing import ArrayLike


def SiO2(
    x: ArrayLike
) -> np.ndarray | float:
    """
    Sellmeier material model for SiO2 at 20 C.

    Valid nominal wavelength range:
    - 0.21 um to 6.7 um

    Parameters
    ----------
    x : ArrayLike
        Wavelength in um.

    Returns
    -------
    np.ndarray | float
        Refractive index of SiO2.
    """
    x_arr = np.asarray(x, dtype=float)

    if np.any((x_arr < 0.21) | (x_arr > 6.7)):
        warnings.warn(
            "Extrapolating Sellmeier equation for SiO2 beyond range of 0.21 um - 6.7 um",
            stacklevel=2,
        )

    n = np.sqrt(
        1
        + 0.6961663 / (1 - (0.0684043 / x_arr) ** 2)
        + 0.4079426 / (1 - (0.1162414 / x_arr) ** 2)
        + 0.8974794 / (1 - (9.8961610 / x_arr) ** 2)
    )

    if np.asarray(x).ndim == 0:
        return float(np.asarray(n))
    return n
