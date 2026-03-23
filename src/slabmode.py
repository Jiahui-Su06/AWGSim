import numpy as np
from typing import Callable

from slabindex import slabindex


def slabmode(
    lambda0: float,
    t: float,
    na: float | Callable[[float], float],
    nc: float | Callable[[float], float],
    ns: float | Callable[[float], float],
    y: np.ndarray | None = None,
    modes: int | float = np.inf,
    polarisation: str = "TE",
    limits: tuple[float, float] | None = None,
    points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Guided mode electromagnetic fields of a 3-layer planar waveguide.

    Args:
        lambda0 (float): Free-space wavelength.
        t (float): Core thickness.
        na (float | Callable[[float], float]): Top cladding index or dispersion function n(lambda0).
        nc (float | Callable[[float], float]): Core index or dispersion function n(lambda0).
        ns (float | Callable[[float], float]): Substrate index or dispersion function n(lambda0).
        y (np.ndarray | None, optional): Coordinate vector. If None, generated from limits and points. Defaults to None.
        modes (int | float, optional): Maximum number of modes to solve. Defaults to np.inf.
        polarisation (str, optional): "TE" or "TM". Defaults to "TE".
        limits (tuple[float, float] | None, optional): Coordinate range if y is not provided. Default is (-3*t, 3*t).
        points (int, optional): Number of coordinate points if y is not provided. Defaults to 100.
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            E (np.ndarray): Electric field array of shape (Ny, Nm, 3).
            H (np.ndarray): Magnetic field array of shape (Ny, Nm, 3).
            y (np.ndarray): Coordinate vector of shape (Ny,).
            neff (np.ndarray): Effective indices of supported modes.
    """
    n0 = 120 * np.pi # permittivity of free space

    if polarisation not in ("TE", "TM"):
        raise ValueError("polarisation must be 'TE' or 'TM'")
    if modes <= 0:
        raise ValueError("modes must be positive integer")
    if limits is None:
        limits = (-3 * t, 3 * t)
    
    # Evaluate dispersion functions if needed
    if callable(ns):
        ns = float(ns(lambda0))
    if callable(nc):
        nc = float(nc(lambda0))
    if callable(na):
        na = float(na(lambda0))
    
    # build coordinate vector
    if y is None:
        y = np.linspace(limits[0], limits[1], points)
    else:
        y = np.asarray(y).reshape(-1)
    
    # Region indices
    i1 = np.where(y < -t / 2)[0]
    i2 = np.where((y >= -t / 2) & (y <= t / 2))[0]
    i3 = np.where(y > t / 2)[0]

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

    k0 = 2 * np.pi / lambda0

    for m in range(len(neff)):
        ne = neff[m]

        if not (ne > ns and ne > na and ne < nc):
            continue

        p = k0 * np.sqrt(max(ne**2 - ns**2, 0.0))
        k = k0 * np.sqrt(max(nc**2 - ne**2, 0.0))
        q = k0 * np.sqrt(max(ne**2 - na**2, 0.0))

        if polarisation == "TE":
            # phase match condition
            f = 0.5 * np.arctan2(k * (p - q), (k**2 + p * q))

            # normalization
            C = np.sqrt(n0 / ne / (t + 1 / p + 1 / q))

            # modal field
            Em = np.zeros(len(y), dtype=complex)
            Em[i1] = C * np.cos(k * t / 2 + f) * np.exp(p * (t / 2 + y[i1]))
            Em[i2] = C * np.cos(k * y[i2] - f)
            Em[i3] = C * np.cos(k * t / 2 - f) * np.exp(q * (t / 2 - y[i3]))

            # field components
            H[:, m, 0] = 0
            H[:, m, 1] = ne / n0 * Em
            H[:, m, 2] = 1j / (k0 * n0) * np.concatenate(([0], np.diff(Em)))
            E[:, m, 0] = Em
            E[:, m, 1] = 0
            E[:, m, 2] = 0
        else:  # TM
            n = np.ones(len(y), dtype=float)
            n[i1] = ns
            n[i2] = nc
            n[i3] = na

            # phase match condition
            f = 0.5 * np.arctan2(
                (k / nc**2) * (p / ns**2 - q / na**2),
                ((k / nc**2) ** 2 + (p / ns**2) * (q / na**2))
            )

            # normalization
            p2 = ne**2 / nc**2 + ne**2 / ns**2 - 1
            q2 = ne**2 / nc**2 + ne**2 / na**2 - 1
            C = -np.sqrt(nc**2 / n0 / ne / (t + 1 / (p * p2) + 1 / (q * q2)))

            # modal field
            Hm = np.zeros(len(y), dtype=complex)
            Hm[i1] = C * np.cos(k * t / 2 + f) * np.exp(p * (t / 2 + y[i1]))
            Hm[i2] = C * np.cos(k * y[i2] - f)
            Hm[i3] = C * np.cos(k * t / 2 - f) * np.exp(q * (t / 2 - y[i3]))

            # field components
            E[:, m, 0] = 0
            E[:, m, 1] = -ne * n0 / (n**2) * Hm
            E[:, m, 2] = -1j * n0 / (k0 * nc**2) * np.concatenate(([0], np.diff(Hm)))
            H[:, m, 0] = Hm
            H[:, m, 1] = 0
            H[:, m, 2] = 0

    return E, H, y, neff


def test():
    import matplotlib.pyplot as plt
    wavelengh = np.arange(1.5, 1.6, 0.001)
    Neff = np.zeros(len(wavelengh))

    for i in range(len(wavelengh)):
        (E, H, y, neff) = slabmode(lambda0=wavelengh[i], t=0.8, na=1, nc=2.12, ns=1.44, modes=4, polarisation="TE")
        Neff[i] = neff[0]
    
    plt.plot(wavelengh, Neff)
    plt.xlim([1.5, 1.6])
    plt.show()



if __name__ == '__main__':
    test()
