import numpy as np
from collections.abc import Callable

from .slabindex import slabindex


def wgindex(
    lmbda: float,
    w: float,
    h: float,
    t: float,
    na: float | Callable[[float], float],
    nc: float | Callable[[float], float],
    ns: float | Callable[[float], float],
    modes: int | float = np.inf,
    polarisation: str = "te",
) -> np.ndarray:
    """
    Effective index method for guided modes in an arbitrary etched waveguide.

    Parameters
    ----------
    lmbda : float
        Free-space wavelength.
    w : float
        Core width.
    h : float
        Total core height.
    t : float
        Slab thickness.
        - t < h : rib waveguide
        - t == 0: rectangular waveguide w x h
        - t == h: uniform slab of thickness h
    na : float or callable
        Top cladding refractive index, or function na(lmbda).
    nc : float or callable
        Core refractive index, or function nc(lmbda).
    ns : float or callable
        Bottom substrate refractive index, or function ns(lmbda).
    modes : int or float, optional
        Maximum number of modes to solve. Default is np.inf.
    polarisation : str, optional
        'TE' or 'TM'. Default is 'te'.

    Returns
    -------
    np.ndarray
        Effective indices of the supported guided modes.
    """
    if not (0 <= t <= h):
        raise ValueError("The slab thickness parameter must be within [0, h].")

    if callable(ns):
        ns = ns(lmbda)
    if callable(nc):
        nc = nc(lmbda)
    if callable(na):
        na = na(lmbda)

    na = float(na)
    nc = float(nc)
    ns = float(ns)

    pol = polarisation.upper()
    if pol not in {"TE", "TM"}:
        raise ValueError("polarisation must be 'TE' or 'TM'")

    # Solve region I
    neff_I = slabindex(
        lambda0=lmbda,
        t=h,
        na=na,
        nc=nc,
        ns=ns,
        modes=modes,
        polarisation=polarisation,
    )

    if t == h:
        return neff_I

    # Solve region II
    if t > 0:
        neff_II = slabindex(
            lambda0=lmbda,
            t=t,
            na=na,
            nc=nc,
            ns=ns,
            modes=modes,
            polarisation=polarisation,
        )
    else:
        neff_II = np.array([na], dtype=float)

    neff = []

    n_pairs = min(len(neff_I), len(neff_II))

    if pol == "TE":
        for m in range(n_pairs):
            n = slabindex(
                lambda0=lmbda,
                t=w,
                na=neff_II[m],
                nc=neff_I[m],
                ns=neff_II[m],
                modes=modes,
                polarisation="tm",
            )
            neff.extend(n[n > max(na, ns)])
    else:
        for m in range(n_pairs):
            n = slabindex(
                lambda0=lmbda,
                t=w,
                na=neff_II[m],
                nc=neff_I[m],
                ns=neff_II[m],
                modes=modes,
                polarisation="te",
            )
            neff.extend(n[n > max(na, ns)])

    return np.asarray(neff, dtype=float)
