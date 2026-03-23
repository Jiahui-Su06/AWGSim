import numpy as np
import math
from typing import Callable
from scipy.optimize import root


def slabindex(
    lambda0: float,
    t: float,
    na: float | Callable,
    nc: float | Callable,
    ns: float | Callable,
    modes: int = 1000,
    polarisation: str = "TE"
) -> np.ndarray:
    """Solves for the TE (or TM) effective index of a 3-layer slab waveguide.

    Description:
        Solves for the TE (or TM) effective index of a 3-layer slab waveguide
                na          y
        ^   ----------      |
        t       nc          x -- z
        v   ----------     
                ns
        
        with propagation in the +z direction

    Args:
        lambda0 (float): freespace wavelength
        t (float): core (guiding layer) thickness
        na (float | Callable): cladding index
        nc (float | Callable): core index
        ns (float | Callable): substrate index
        modes (int, optional): max number of modes to solve
        polarisation (str, optional): one of 'TE' or 'TM'
    
    Returns:
        np.ndarray: vector of indexes of each supported mode
    
    NOTE:
        it is possible to provide a function of the form n = lambda lambda0: func(lambda0) 
        for the refractive index which will be called using lambda0.
    """
    neff = []

    if modes <= 0:
        raise ValueError("modes must be positive integer")
    if polarisation not in ("TE", "TM"):
        raise ValueError("polarisation must be one of 'TE' or 'TM'")
    na = na(lambda0) if callable(na) else na
    nc = nc(lambda0) if callable(nc) else nc
    ns = ns(lambda0) if callable(ns) else ns
    
    if na >= nc or ns >= nc:
        return neff
    
    # TIR critical angle
    a0 = max(np.arcsin(ns/nc), np.arcsin(na/nc))
    
    if polarisation == "TE":
        # Fresnel reflection coefficients (E-mode)
        def B1(a):
            return np.sqrt(((ns/nc)**2 - np.sin(a)**2) + 0j)
        def r1(a):
            return (np.cos(a) - B1(a)) / (np.cos(a) + B1(a))
        def B2(a):
            return np.sqrt(((na/nc)**2 - np.sin(a)**2) + 0j)
    elif polarisation == "TM":
        # Fresnel reflection coefficients (H-mode)
        def B1(a):
            return (nc/ns)**2 * np.sqrt(((ns/nc)**2 - np.sin(a)**2) + 0j)
        def r1(a):
            return (np.cos(a) - B1(a)) / (np.cos(a) + B1(a))
        def B2(a):
            return (nc/na)**2 * np.sqrt(((na/nc)**2 - np.sin(a)**2) + 0j)
    
    def r2(a):
            return (np.cos(a) - B2(a)) / (np.cos(a) + B2(a))
    
    # reflection phase shifts
    def phi1(a):
        return np.angle(r1(a))
    def phi2(a):
        return np.angle(r2(a))
    
    # number of supported modes
    M = math.floor((4*np.pi*t*nc/lambda0*np.cos(a0) + phi1(a0) + phi2(a0)) / (2*np.pi))

    for m in range(min(modes, M+1)):
        a = root(lambda a : 4*np.pi*t*nc/lambda0*np.cos(a)+phi1(a)+phi2(a)-2*m*np.pi, 1)
        neff.append((np.sin(a.x) * nc)[0])
    return neff


def test():
    neff = slabindex(lambda0=1.55, t=0.6, na=1, nc=2.12, ns=1.44, modes=4, polarisation="TE")
    print(neff)
    # result = np.sqrt(-1)
    # print(result)


if __name__ == '__main__':
    test()
