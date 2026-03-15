import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable

from .slabindex import slabindex
from .slabmode import slabmode
from .wgindex import wgindex


def wgmode(
    lmbda: float,
    w: float,
    h: float,
    t: float,
    na: float | Callable[[float], float],
    nc: float | Callable[[float], float],
    ns: float | Callable[[float], float],
    x: ArrayLike | None = None,
    polarisation: str = "te",
    limits: tuple[float, float] | None = None,
    points: int = 100,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Solve 2D waveguide cross section by the effective index method.

    This function solves the fundamental TE- or TM-like mode fields.

    Parameters
    ----------
    lmbda : float
        Free-space wavelength.
    w : float
        Core width.
    h : float
        Core thickness.
    t : float
        Slab thickness.
        - t < h  : rib waveguide
        - t == 0 : rectangular waveguide
        - t == h : uniform slab
    na : float or callable
        Top cladding refractive index, or function na(lmbda).
    nc : float or callable
        Core refractive index, or function nc(lmbda).
    ns : float or callable
        Bottom substrate refractive index, or function ns(lmbda).
    x : ArrayLike | None, optional
        Coordinate vector. If None, generated from `limits` and `points`.
    polarisation : str, optional
        'TE' or 'TM'. Default is 'te'.
    limits : tuple[float, float] | None, optional
        Coordinate limits when x is None. Default is (-3*w, 3*w).
    points : int, optional
        Number of coordinate points when x is None.

    Returns
    -------
    E : list[np.ndarray]
        Electric field components [Ex, Ey, Ez].
    H : list[np.ndarray]
        Magnetic field components [Hx, Hy, Hz].
    x : np.ndarray
        Coordinate vector.
    neff : np.ndarray
        Effective index array of solved modes.
    """
    if not (0 <= t <= h):
        raise ValueError("The slab thickness parameter must be within [0, h].")

    if limits is None:
        limits = (-3 * w, 3 * w)

    if x is None:
        x = np.linspace(limits[0], limits[1], points).reshape(-1)
    else:
        x = np.asarray(x).reshape(-1)

    nil = np.zeros_like(x, dtype=complex)
    pol = polarisation.upper()

    if pol == "TE":
        neff = wgindex(
            lmbda=lmbda,
            w=w,
            h=h,
            t=t,
            na=na,
            nc=nc,
            ns=ns,
            modes=1,
            polarisation="te",
        )

        # solve slab mode in section I
        n_I = slabindex(
            lambda0=lmbda,
            t=h,
            na=na,
            nc=nc,
            ns=ns,
            modes=1,
            polarisation="te",
        )

        # solve slab mode in section II
        if t > 0:
            n_II = slabindex(
                lambda0=lmbda,
                t=t,
                na=na,
                nc=nc,
                ns=ns,
                modes=1,
                polarisation="te",
            )
        else:
            n_II = np.array([na(lmbda) if callable(na) else na], dtype=float)

        # equivalent slab mode
        Ek, Hk, _, _ = slabmode(
            lambda0=lmbda,
            t=w,
            na=float(n_II[0]),
            nc=float(n_I[0]),
            ns=float(n_II[0]),
            y=x,
            modes=1,
            polarisation="tm",
        )

        f = np.max(np.real(Ek[:, 0, 1]))

        # assemble field components
        Ex = Ek[:, 0, 1] / f
        Hy = -Hk[:, 0, 0] / f
        Hz = -Hk[:, 0, 2] / f

        E = [Ex, nil, nil]
        H = [nil, Hy, Hz]

    elif pol == "TM":
        neff = wgindex(
            lmbda=lmbda,
            w=w,
            h=h,
            t=t,
            na=na,
            nc=nc,
            ns=ns,
            modes=1,
            polarisation="tm",
        )

        # solve slab mode in section I
        n_I = slabindex(
            lambda0=lmbda,
            t=h,
            na=na,
            nc=nc,
            ns=ns,
            modes=1,
            polarisation="tm",
        )

        # solve slab mode in section II
        if t > 0:
            n_II = slabindex(
                lambda0=lmbda,
                t=t,
                na=na,
                nc=nc,
                ns=ns,
                modes=1,
                polarisation="tm",
            )
        else:
            n_II = np.array([na(lmbda) if callable(na) else na], dtype=float)

        # equivalent slab mode
        Ek, Hk, _, _ = slabmode(
            lambda0=lmbda,
            t=w,
            na=float(n_II[0]),
            nc=float(n_I[0]),
            ns=float(n_II[0]),
            y=x,
            modes=1,
            polarisation="te",
        )

        # assemble field components
        Ey = Ek[:, 0, 0]
        Ez = Ek[:, 0, 2]
        Hx = -Hk[:, 0, 1]

        E = [nil, Ey, Ez]
        H = [Hx, nil, nil]

    else:
        raise ValueError("polarisation must be 'TE' or 'TM'")

    return E, H, x, neff
