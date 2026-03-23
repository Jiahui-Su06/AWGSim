import numpy as np
from typing import Callable

from slabindex import slabindex

def wgindex(
    lambda0: float,
    w: float,
    h: float,
    t: float,
    na: float | Callable,
    nc: float | Callable,
    ns: float | Callable,
    modes : int = 1000,
    polarisation: str = "TE"
) -> list:
    """Effective index method for guided modes in arbitrary waveguide

    Description:
        solves for the TE (or TM) effective index of an etched waveguide 
        structure using the effectice index method.

                    |<   w   >|
                     _________           _____
                    |         |            ^
        ___    _____|         |_____ 
         ^                                 h
         t                                  
        _v_    _____________________     __v__
        
                II  |    I    |  II

    Args:
        lambda0 (float): free-space wavelength
        w (float): core width
        h (float): slab thickness
        t (float): slab thickness
            t < h  : rib waveguide
            t == 0 : rectangular waveguide w x h
            t == h : uniform slab of thickness t
        na (float | Callable): (top) oxide cladding layer material index
        nc (float | Callable): (middle) core layer material index
        ns (float | Callable): (bottom) substrate layer material index
        modes (int, optional): number of modes to solve
        polarisation (str, optional): one of 'TE' or 'TM'

    Returns:
        list: TE (or TM) mode index (array of index if multimode)
    
    NOTE:
        it is possible to provide a function of the form n = lambda lambda0: func(lambda0) for 
        the refractive index which will be called using lambda0.
    """
    if modes <= 0:
        raise ValueError("modes must be positive integer")
    if polarisation not in ("TE", "TM"):
        raise ValueError("polarisation must be 'TE' or 'TM'")
    
    na = na(lambda0) if callable(na) else na
    nc = nc(lambda0) if callable(nc) else nc
    ns = ns(lambda0) if callable(ns) else ns

    t = min(max(t, 0), h)

    neff_I = slabindex(
        lambda0=lambda0,
        t=h,
        na=na,
        nc=nc,
        ns=ns,
        modes=modes,
        polarisation=polarisation
    )

    if t == h:
        return neff_I
    if t > 0:
        neff_II = slabindex(
            lambda0=lambda0,
            t=t,
            na=na,
            nc=nc,
            ns=ns,
            modes=modes,
            polarisation=polarisation
        )
    else:
        neff_II = na

    neff = []

    if polarisation == "TE":
        for m in range(min(len(neff_I), len(neff_II))):
            n = slabindex(
                lambda0=lambda0,
                t=w,
                na=neff_II[m],
                nc=neff_I[m],
                ns=neff_II[m],
                modes=modes,
                polarisation="TM"
            )
            neff.extend(i for i in n if i > max(ns, na))
    elif polarisation == "TM":
        for m in range(min(len(neff_I), len(neff_II))):
            n = slabindex(
                lambda0=lambda0,
                t=w,
                na=neff_II[m],
                nc=neff_I[m],
                ns=neff_II[m],
                modes=modes,
                polarisation="TE"
            )
            neff.extend(i for i in n if i > max(ns, na))
    
    return neff


def test():
    pass


if __name__ == '__main__':
    test()