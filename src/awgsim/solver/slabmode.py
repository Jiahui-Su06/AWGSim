import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

from .slabindex import slabindex


def slabmode(
    lambda0: float,
    t: float,
    na: float | Callable[[float], float],
    nc: float | Callable[[float], float],
    ns: float | Callable[[float], float],
    y: ArrayLike | None = None,
    modes: int | float = np.inf,
    polarisation: str = "te",
    limits: tuple[float, float] | None = None,
    points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve guided-mode electromagnetic fields of a 3-layer planar waveguide.

    Parameters
    ----------
    lambda0 : float
        Free-space wavelength.
    t : float
        Core thickness.
    na : float or callable
        Top cladding index, or function na(lambda0).
    nc : float or callable
        Core index, or function nc(lambda0).
    ns : float or callable
        Substrate index, or function ns(lambda0).
    y : ArrayLike | None, optional
        Coordinate vector. If None, generated from `limits` and `points`.
    modes : int or float, optional
        Maximum number of modes to solve. Default is np.inf.
    polarisation : str, optional
        'TE' or 'TM'. Default is 'te'.
    limits : tuple[float, float] | None, optional
        Coordinate range used when y is None. Default is (-3*t, 3*t).
    points : int, optional
        Number of coordinate points if y is None.

    Returns
    -------
    E : np.ndarray
        Electric field array with shape (len(y), n_modes, 3).
    H : np.ndarray
        Magnetic field array with shape (len(y), n_modes, 3).
    y : np.ndarray
        Coordinate vector.
    neff : np.ndarray
        Effective indices of supported modes.
    """
    n0 = 120 * np.pi  # free-space impedance in ohms

    if limits is None:
        limits = (-3 * t, 3 * t)

    # Evaluate dispersive refractive indices if callables are provided
    if callable(ns):
        ns = ns(lambda0)
    if callable(nc):
        nc = nc(lambda0)
    if callable(na):
        na = na(lambda0)

    ns = float(ns)
    nc = float(nc)
    na = float(na)

    # Build coordinate vector
    if y is None:
        y = np.linspace(limits[0], limits[1], points)
    else:
        y = np.asarray(y).reshape(-1)

    # Region masks
    i1 = y < -t / 2
    i2 = (y >= -t / 2) & (y <= t / 2)
    i3 = y > t / 2

    # Solve mode effective indices
    neff = slabindex(
        lambda0=lambda0,
        t=t,
        na=ns,
        nc=nc,
        ns=na,
        modes=modes,
        polarisation=polarisation,
    )

    # Initialize fields: shape = (len(y), n_modes, 3)
    E = np.zeros((len(y), len(neff), 3), dtype=complex)
    H = np.zeros((len(y), len(neff), 3), dtype=complex)

    k0 = 2 * np.pi / lambda0
    pol = polarisation.upper()

    for m in range(len(neff)):
        neff_m = neff[m]

        p = k0 * np.sqrt(neff_m**2 - ns**2 + 0j)  # bottom
        k = k0 * np.sqrt(nc**2 - neff_m**2 + 0j)  # core transverse
        q = k0 * np.sqrt(neff_m**2 - na**2 + 0j)  # top

        if pol == "TE":
            # phase match condition
            f = 0.5 * np.arctan2(
                np.real(k * (p - q)),
                np.real(k**2 + p * q),
            )

            # normalization
            C = np.sqrt(n0 / neff_m / (t + 1 / p + 1 / q))

            # modal field
            Em = np.zeros_like(y, dtype=complex)
            Em[i1] = np.cos(k * t / 2 + f) * np.exp(p * (t / 2 + y[i1]))
            Em[i2] = np.cos(k * y[i2] - f)
            Em[i3] = np.cos(k * t / 2 - f) * np.exp(q * (t / 2 - y[i3]))
            Em = C * Em

            # field components
            H[:, m, 1] = 0
            H[:, m, 2] = neff_m / n0 * Em
            H[:, m, 3] = 1j / (k0 * n0) * np.concatenate(([0], np.diff(Em)))
            E[:, m, 0] = Em
            E[:, m, 1] = 0
            E[:, m, 2] = 0
        elif pol == "TM":
            n = np.ones_like(y, dtype=float)
            n[i1] = ns
            n[i2] = nc
            n[i3] = na

            # phase match condition
            f = 0.5 * np.arctan2(
                np.real((k / nc**2) * (p / ns**2 - q / na**2)),
                np.real((k / nc**2) ** 2 + (p / ns**2) * (q / na**2)),
            )

            # normalization
            p2 = neff_m**2 / nc**2 + neff_m**2 / ns**2 - 1
            q2 = neff_m**2 / nc**2 + neff_m**2 / na**2 - 1
            C = -np.sqrt(nc**2 / n0 / neff_m / (t + 1 / (p * p2) + 1 / (q * q2)))

            # modal field
            Hm = np.zeros_like(y, dtype=complex)
            Hm[i1] = np.cos(k * t / 2 + f) * np.exp(p * (t / 2 + y[i1]))
            Hm[i2] = np.cos(k * y[i2] - f)
            Hm[i3] = np.cos(k * t / 2 - f) * np.exp(q * (t / 2 - y[i3]))
            Hm = C * Hm

            # field components
            E[:, m, 0] = 0
            E[:, m, 1] = -neff_m * n0 / (n**2) * Hm
            E[:, m, 2] = -1j * n0 / (k0 * nc**2) * np.concatenate(([0], np.diff(Hm)))
            H[:, m, 0] = Hm
            H[:, m, 1] = 0
            H[:, m, 2] = 0
        else:
            raise ValueError("polarisation must be 'TE' or 'TM'")

    return E, H, y, neff
