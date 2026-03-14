import numpy as np
from scipy.optimize import brentq
from collections.abc import Callable


def slabindex(
    lambda0: float,
    t: float,
    na: float | Callable[[float], float],
    nc: float | Callable[[float], float],
    ns: float | Callable[[float], float],
    modes: int | float = np.inf, # default modes nember is inf, all of them.
    polarisation: str = "te",
) -> np.ndarray:
    """
    Solve the effective indices of guided modes in a 3-layer slab waveguide.

              na          y
       ^   ----------      |
       t       nc          x -- z
       v   ----------
              ns

    Propagation is along +z.

    Parameters
    ----------
    lambda0 : float
        Free-space wavelength.
    t : float
        Core (guiding layer) thickness.
    na : float or callable
        Cladding index, or a function na(lambda0).
    nc : float or callable
        Core index, or a function nc(lambda0).
    ns : float or callable
        Substrate index, or a function ns(lambda0).
    modes : int or float, optional
        Maximum number of modes to solve. Default is np.inf.
    polarisation : str, optional
        One of {"te", "tm", "TE", "TM"}. Default is "te".

    Returns
    -------
    np.ndarray
        Effective index of each supported mode.
    """
    # Evaluate dispersive refractive indices if callables were provided
    if callable(ns):
        ns = ns(lambda0)
    if callable(nc):
        nc = nc(lambda0)
    if callable(na):
        na = na(lambda0)

    na = float(na)
    nc = float(nc)
    ns = float(ns)

    # TIR critical angle
    a0 = np.max(np.arcsin(ns / nc), np.arcsin(na / nc))
    if not np.isreal(a0):
        return np.array([], dtype=float)

    pol = polarisation.upper()
    if pol not in {"TE", "TM"}:
        raise ValueError("polarisation must be one of 'TE' or 'TM'.")

    if pol == "TE":
        # Fresnel reflection coefficients (E-mode)
        def B1(a):
            return np.sqrt((ns / nc) ** 2 - np.sin(a) ** 2 + 0j)
        def r1(a):
            return (np.cos(a) - B1(a)) / (np.cos(a) + B1(a))  # lower interface
        def B2(a):
            return np.sqrt((na / nc) ** 2 - np.sin(a) ** 2 + 0j)
        def r2(a):
            return (np.cos(a) - B2(a)) / (np.cos(a) + B2(a))  # upper interface
    else:
        # Fresnel reflection coefficients (H-mode)
        def B1(a):
            return (nc / ns) ** 2 * np.sqrt((ns / nc) ** 2 - np.sin(a) ** 2 + 0j)
        def r1(a):
            return (np.cos(a) - B1(a)) / (np.cos(a) + B1(a))  # lower interface
        def B2(a):
            return (nc / na) ** 2 * np.sqrt((na / nc) ** 2 - np.sin(a) ** 2 + 0j)
        def r2(a):
            return (np.cos(a) - B2(a)) / (np.cos(a) + B2(a))  # upper interface

    # Reflection phase shifts
    def phi1(a):
        return np.angle(r1(a))
    def phi2(a):
        return np.angle(r2(a))

    # Number of supported modes
    M = int(np.floor((4 * np.pi * t * nc / lambda0 * np.cos(a0) + phi1(a0) + phi2(a0)) / (2 * np.pi)))

    if M < 0:
        return np.array([], dtype=float)

    supported_modes = np.arange(1, M + 2)

    if np.isfinite(modes):
        supported_modes = supported_modes[supported_modes <= int(modes)]

    neff = []

    # Avoid the exact endpoints to keep the root finder stable
    eps = 1e-12
    a_left = float(np.real(a0)) + eps
    a_right = np.pi / 2 - eps

    for m in supported_modes:
        def f(a):
            return (
                4 * np.pi * t * nc / lambda0 * np.cos(a)
                + phi1(a)
                + phi2(a)
                - 2 * (m - 1) * np.pi
            )

        # Sample to find a sign-change bracket for brentq
        grid = np.linspace(a_left, a_right, 2000)
        vals = np.real(np.array([f(a) for a in grid]))

        root_found = False
        for i in range(len(grid) - 1):
            f1 = vals[i]
            f2 = vals[i + 1]

            if not (np.isfinite(f1) and np.isfinite(f2)):
                continue

            if f1 == 0:
                a_root = grid[i]
                root_found = True
                break

            if f1 * f2 < 0:
                a_root = brentq(lambda a: np.real(f(a)), grid[i], grid[i + 1])
                root_found = True
                break

        if root_found:
            neff.append(nc * np.sin(a_root))

    return np.asarray(neff, dtype=float)
