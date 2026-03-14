import warnings
import numpy as np
from numpy.typing import ArrayLike


def Si(
    x: ArrayLike, 
    T: float = 295
) -> np.ndarray | float:
    """
    Sellmeier material model for Si.

    Valid nominal range:
    - temperature: 20 K to 300 K
    - wavelength: 1.1 um to 5.6 um

    Parameters
    ----------
    x : ArrayLike
        Wavelength in um.
    T : float, optional
        Temperature in K. Default is 295.

    Returns
    -------
    np.ndarray | float
        Refractive index of Si.
    """
    x_arr = np.asarray(x, dtype=float)

    if np.any((x_arr < 1.1) | (x_arr > 5.6)):
        warnings.warn(
            "Extrapolating model equation for Si beyond range of 1.1 um - 5.6 um",
            stacklevel=2,
        )

    if (T < 20) or (T > 300):
        warnings.warn(
            "Extrapolating model equation for Si beyond temperature range of 20 K - 300 K",
            stacklevel=2,
        )

    S1 = np.polyval([3.4469e-12, -5.823e-09, 4.2169e-06, -0.00020802, 10.491], T)
    S2 = np.polyval([-1.3509e-06, 0.0010594, -0.27872, 29.166, -1346.6], T)
    S3 = np.polyval([103.24, 678.41, -76158, -1.7621e6, 4.4283e7], T)

    x1 = np.polyval([2.3248e-14, -2.5105e-10, 1.6713e-07, -1.1423e-05, 0.29971], T)
    x2 = np.polyval([-1.1321e-06, 0.001175, -0.35796, 42.389, -3517.1], T)
    x3 = np.polyval([23.577, -39.37, -6907.4, -1.4498e5, 1.714e6], T)

    n = np.sqrt(
        1
        + S1 * x_arr**2 / (x_arr**2 - x1**2)
        + S2 * x_arr**2 / (x_arr**2 - x2**2)
        + S3 * x_arr**2 / (x_arr**2 - x3**2)
    )

    if np.asarray(x).ndim == 0:
        return float(np.asarray(n))
    return n
