import numpy as np
from typing import Callable

from slabindex import slabindex


def slabmode(
    lambda0: float,
    t: float,
    na: float | Callable,
    nc: float | Callable,
    ns: float | Callable,
    y: list | None = None,
    modes: int = 1000,
    polarisation: str = "TE",
    limits: list[float, float] | None = None,
    points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Guided mode electromagnetic fields of a 3-layer planar waveguide.

    Args:
        lambda0 (float): simulation wavelength (freespace)
        t (float): core (guiding layer) thickness
        na (float | Callable): top cladding index
        nc (float | Callable): core layer index
        ns (float | Callable): substrate layer index
        y (list | None, optional): provide the coordinate vector to use
        modes (int): max number of modes to solve
        polarisation (str): one of 'TE' or 'TM'
        limits (list[float, float] | None, optional): coordinate range [min, max] (if y was not provided)
        points (int, optional): number of coordinate points (if y was not provided)
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, list]: 
            E, H (np.ndarray): all x,y,z field components, ex. E(<y>,<m>,<i>), where m is the mode 
            number, i is the field component index such that 1: x, 2: y, 3:z
            y (np.ndarray): coordinate vector of shape
            neff (list): effective indices of supported modes
    
    NOTE: 
        it is possible to provide a function of the form n = lambda lambda0: func(lambda0) for 
        the refractive index which will be called using lambda0.
    """
    # permittivity of free space
    n0 = 120*np.pi 

    if modes <= 0:
        raise ValueError("modes must be positive integer")
    if polarisation not in ("TE", "TM"):
        raise ValueError("polarisation must be 'TE' or 'TM'")
    if limits is None:
        limits = [-3*t, 3*t]
    
    na = na(lambda0) if callable(na) else na
    nc = nc(lambda0) if callable(nc) else nc
    ns = ns(lambda0) if callable(ns) else ns

    y = np.linspace(limits[0], limits[1], points) if y is None else y

    i1 = []
    i2 = []
    i3 = []
    for i, e in enumerate(y):
        if e < -t/2:
            i1.append(i)
        elif e <= t/2 and e >= -t/2:
            i2.append(i)
        else:
            i3.append(i)

    # solve for the mode effective indexes
    neff = slabindex(
        lambda0=lambda0,
        t=t,
        na=na,
        nc=nc,
        ns=ns,
        modes=modes,
        polarisation=polarisation,
    )

    # intialize the fields
    E = np.zeros((len(y), len(neff), 3), dtype=complex)
    H = np.zeros((len(y), len(neff), 3), dtype=complex)
    k0 = 2*np.pi / lambda0

    for m in range(len(neff)):
        p = k0 * np.sqrt(neff[m]**2 - ns**2)
        k = k0 * np.sqrt(nc**2 - neff[m]**2)
        q = k0 * np.sqrt(neff[m]**2 - na**2)

        if polarisation == "TE":
            # phase match condition
            f = 0.5 * np.arctan2(k*(p-q), (k**2 + p*q))
            # normalization
            C = np.sqrt(n0/neff[m]/(t + 1/p + 1/q))
            # E-mode
            Em1 = np.cos(k*t/2 + f) * np.exp(p * (t/2 + y[i1]))
            Em2 = np.cos(k*y[i2] - f)
            Em3 = np.cos(k*t/2 - f) * np.exp(q * (t/2 - y[i3]))
            Em = np.concatenate((Em1, Em2, Em3)) * C
            # E and H components
            H[:, m, 1] = neff[m] / n0 * Em
            H[:, m, 2] = 1j / (k0*n0) * np.concatenate((np.zeros(1), np.diff(Em)))
            E[:, m, 0] = Em
        elif polarisation == "TM":
            n = np.ones(len(y))
            n[i1] = ns
            n[i2] = nc
            n[i3] = na

            # phase match condition
            f = 0.5 * np.arctan2(
                (k/nc**2) * (p/ns**2 - q/na**2),
                ((k/nc**2)**2 + p/ns**2 * q/na**2)
            )
            # normalization
            p2 = neff[m]**2/nc**2 + neff[m]**2/ns**2 - 1
            q2 = neff[m]**2/nc**2 + neff[m]**2/na**2 - 1
            C = - np.sqrt(nc**2/n0/neff[m]/(t + 1/(p*p2) + 1/(q*q2)))
            # H-mode
            Hm1 = np.cos(k*t/2 + f) * np.exp(p * (t/2 + y[i1]))
            Hm2 = np.cos(k*y[i2] - f)
            Hm3 = np.cos(k*t/2 - f) * np.exp(q * (t/2 - y[i3]))
            Hm = np.concatenate((Hm1, Hm2, Hm3)) * C
            # E and H components
            E[:, m, 1] = - neff[m] * n0 / n**2 * Hm
            E[:, m, 2] = - 1j * n0 / (k0*nc**2) * np.concatenate((np.zeros(1), np.diff(Hm)))
            H[:, m, 0] = Hm
    
    return E, H, y, neff


def test():
    import matplotlib.pyplot as plt
    wavelengh = np.arange(1.5, 1.6, 0.001)
    Neff = np.zeros(len(wavelengh))
    

    for i in range(len(wavelengh)):
        (E, H, y, neff) = slabmode(lambda0=wavelengh[i], t=0.8, na=1, nc=2.12, ns=1.44, modes=4, polarisation="TE")
        # print(type(E))
        # print(type(H))
        # print(type(y))
        # print(type(neff))
        # print(type(slabmode(lambda0=wavelengh[i], t=0.8, na=1, nc=2.12, ns=1.44, modes=4, polarisation="TE")))
        Neff[i] = neff[0]
    
    plt.plot(wavelengh, Neff)
    plt.xlim([1.5, 1.6])
    plt.show()



if __name__ == '__main__':
    test()
