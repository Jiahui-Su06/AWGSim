import numpy as np
from numpy.typing import ArrayLike
from typing import Any


def gmode(
    lmbda: float,
    W: float,
    H: float,
    nclad: float,
    ncore: float,
    x: ArrayLike | None = None,
    limits: tuple[float, float] | list[float] | None = None,
    points: int = 100,
    vcoef: tuple[float, float] | list[float] = (0.337, 0.650),
) -> tuple[list[Any], list[Any], np.ndarray]:
    """
    Generate a fake 1D TE-like mode profile.

    Parameters
    ----------
    lmbda : float
        Wavelength.
    W : float
        Waveguide width.
    H : float
        Waveguide height.
    nclad : float
        Cladding refractive index.
    ncore : float
        Core refractive index.
    x : ArrayLike | None, optional
        Coordinate vector. If None, generated from `limits` and `points`.
    limits : tuple[float, float] | list[float] | None, optional
        Range used to generate x when x is None. Default is (-3*W, 3*W).
    points : int, optional
        Number of points in x when x is None.
    vcoef : tuple[float, float] | list[float], optional
        Empirical coefficients. Default is (0.337, 0.650).

    Returns
    -------
    tuple[list[Any], list[Any], np.ndarray]
        (E, Hfield, x), where:
        - E = [Ex, Ey, Ez]
        - H = [Hx, Hy, Hz]
        Only Ex and Hy are populated; others are None.
    """
    if limits is None:
        limits = (-3 * W, 3 * W)

    if x is None:
        x = np.linspace(limits[0], limits[1], points).reshape(-1)
    else:
        x = np.asarray(x).reshape(-1)

    vcoef = np.asarray(vcoef, dtype=float)
    if vcoef.shape[0] != 2:
        raise ValueError("vcoef must contain exactly two elements.")

    V = 2 * np.pi / lmbda * np.sqrt(ncore**2 - nclad**2)

    w = (vcoef[0] * W**1.5 + vcoef[1] / V**1.5) / np.sqrt(W)
    h = (vcoef[0] * H**1.5 + vcoef[1] / V**1.5) / np.sqrt(H)

    n = (nclad + ncore) / 2

    e_amp = (2 / (np.pi * w**2)) ** 0.25
    h_amp = (2 / (np.pi * h**2)) ** 0.25

    Ex = e_amp * np.exp(-(x**2) / w**2)
    Hy = n / (120 * np.pi) * h_amp * np.exp(-(x**2) / h**2)

    # E = {Ex, [], []}, H = {[], Hy, []}
    E = [Ex, None, None]
    H = [None, Hy, None]

    return E, H, x
