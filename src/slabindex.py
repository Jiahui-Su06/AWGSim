import numpy as np
from scipy.optimize import root_scalar
from typing import Callable


def slabindex(
    lambda0: float,
    t: float,
    na: float | Callable[[float], float],
    nc: float | Callable[[float], float],
    ns: float | Callable[[float], float],
    modes: int = np.inf,
    polarisation: str = "TE"
) -> np.ndarray:
    """Solves for the TE (or TM) effective index of a 3-layer slab waveguide.

    Args:
        lambda0 (float): freespace wavelength
        t (float): core (guiding layer) thickness
        na (float | Callable[[float], float]): cladding index
        nc (float | Callable[[float], float]): core index
        ns (float | Callable[[float], float]): substrate index
        modes (int, optional): max number of modes to solve
        polarisation (str, optional): one of 'TE' or 'TM'
    
    Returns:
        np.ndarray: vector of indexes of each supported mode
    """
    if modes <= 0:
        raise ValueError("modes must be positive integer")
    if polarisation not in ("TE", "TM"):
        raise ValueError("polarisation must be one of 'TE' or 'TM'")
    
    if callable(na):
        na = float(na(lambda0))
    if callable(nc):
        nc = float(nc(lambda0))
    if callable(ns):
        ns = float(ns(lambda0))
    
    # No guided mode if core index is not highest
    if na >= nc or ns >= nc:
        return np.array([])
    
    # TIR critical angle
    theta_TIR = max(np.asin(ns / nc), np.asin(na / nc))
    
    if polarisation == "TE":
        # Fresnel reflection coefficients (E-mode)
        def B1(theta):
            return np.sqrt((ns / nc) ** 2 - np.sin(theta) ** 2 + 0j)
        def r1(theta):
            return (np.cos(theta) - B1(theta)) / (np.cos(theta) + B1(theta))
        def B2(theta):
            return np.sqrt((na / nc) ** 2 - np.sin(theta) ** 2 + 0j)
        def r2(theta):
            return (np.cos(theta) - B2(theta)) / (np.cos(theta) + B2(theta))
    else: # TM
        # Fresnel reflection coefficients (H-mode)
        def B1(theta):
            return (nc / ns) ** 2 * np.sqrt((ns / nc) ** 2 - np.sin(theta) ** 2 + 0j)
        def r1(theta):
            return (np.cos(theta) - B1(theta)) / (np.cos(theta) + B1(theta))
        def B2(theta):
            return (nc / na) ** 2 * np.sqrt((na / nc) ** 2 - np.sin(theta) ** 2 + 0j)
        def r2(theta):
            return (np.cos(theta) - B2(theta)) / (np.cos(theta) + B2(theta))
    
    # reflection phase shifts
    def phi_1(theta):
        return np.angle(r1(theta))
    def phi_2(theta):
        return np.angle(r2(theta))
    def char_eq(theta, m): # characteristic equation
        return (
            4 * np.pi * t * nc / lambda0 * np.cos(theta)
            + phi_1(theta)
            + phi_2(theta)
            - 2 * m * np.pi
        )
    
    # number of supported modes
    M = int(np.floor(
        (4 * np.pi * t * nc / lambda0 * np.cos(theta_TIR) 
         + phi_1(theta_TIR) 
         + phi_2(theta_TIR)) 
        / (2 * np.pi)
    ))

    nmodes = max(0, int(min(modes, M + 1)))
    if nmodes == 0:
        return np.array([])
    
    neff = np.empty(nmodes)
    eps = 1e-10
    a = theta_TIR + eps
    b = np.pi / 2 - eps

    # solve the characteristic equation
    for m in range(nmodes):
        fa = char_eq(a, m)
        fb = char_eq(b, m)

        if np.real(fa) * np.real(fb) > 0:
            # fallback: scan for a sign change
            grid = np.linspace(a, b, 2000)
            vals = np.real([char_eq(x, m) for x in grid])
            idx = None
            for i in range(len(vals) - 1):
                if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
                    idx = i
                    break
            if idx is None:
                raise RuntimeError(f"Failed to bracket root for mode m={m}")
            left, right = grid[idx], grid[idx + 1]
        else:
            left, right = a, b
        
        sol = root_scalar(
            lambda th: np.real(char_eq(th, m)),
            bracket=[left, right],
            method="brentq"
        )
        theta_sol = sol.root
        neff[m] = nc * np.sin(theta_sol)
    
    return neff


def test():
    neff = slabindex(lambda0=1.55, t=0.6, na=1, nc=2.12, ns=1.44, modes=4, polarisation="TE")
    print(neff)
    # result = np.sqrt(-1)
    # print(result)


if __name__ == '__main__':
    test()
