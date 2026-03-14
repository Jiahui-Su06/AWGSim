import numpy as np
from numpy.typing import ArrayLike


def Air(
    x: ArrayLike
) -> np.ndarray:
    """
    Material model for air.

    Parameters
    ----------
    x : ArrayLike
        Wavelength array.

    Returns
    -------
    np.ndarray
        Refractive index array, all ones.
    """
    x = np.asarray(x)
    return np.ones_like(x, dtype=float)
