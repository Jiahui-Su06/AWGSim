import warnings
import numpy as np
from numpy.typing import ArrayLike


def Ge(
    x: ArrayLike, 
    T: float = 295
) -> np.ndarray | float:
    """
    Sellmeier material model for Ge.

    Valid nominal range:
    - temperature: 20 K to 300 K
    - wavelength: 1.9 um to 5.5 um

    Parameters
    ----------
    x : ArrayLike
        Wavelength in um.
    T : float, optional
        Temperature in K. Default is 295.

    Returns
    -------
    np.ndarray | float
        Refractive index of Ge.
    """
    x_arr = np.asarray(x, dtype=float)

    if np.any((x_arr < 1.9) | (x_arr > 5.5)):
        warnings.warn(
            "Extrapolating Sellmeier equation for Ge beyond range of 1.9 um - 5.5 um",
            stacklevel=2,
        )

    if (T < 20) or (T > 300):
        warnings.warn(
            "Extrapolating Sellmeier equation for Ge beyond temperature range of 20 K - 300 K",
            stacklevel=2,
        )

    S1 = np.polyval([-4.8624e-12, 2.226e-08, -5.022e-06, 0.0025281, 13.972], T)
    S2 = np.polyval([4.1204e-11, -6.0229e-08, 2.1689e-05, -0.003092, 0.4521], T)
    S3 = np.polyval([-7.7345e-06, 0.0029605, -0.23809, -14.284, 751.45], T)

    x1 = np.polyval([5.3742e-12, -2.2792e-10, -5.9345e-07, 0.00020187, 0.38637], T)
    x2 = np.polyval([9.402e-12, 1.1236e-08, -4.9728e-06, 0.0011651, 1.0884], T)
    x3 = np.polyval([-1.9516e-05, 0.0064936, -0.52702, -0.96795, -2893.2], T)

    n = np.sqrt(
        1
        + S1 * x_arr**2 / (x_arr**2 - x1**2)
        + S2 * x_arr**2 / (x_arr**2 - x2**2)
        + S3 * x_arr**2 / (x_arr**2 - x3**2)
    )

    if np.asarray(x).ndim == 0:
        return float(np.asarray(n))
    return n
