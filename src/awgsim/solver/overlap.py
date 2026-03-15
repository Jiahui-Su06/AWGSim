import numpy as np
from numpy.typing import ArrayLike


def overlap(
    x: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    hu: ArrayLike | None = None,
    hv: ArrayLike | None = None
) -> float:
    """
    Compute 1D overlap integral, with or without magnetic field.

    Parameters
    ----------
    x : array_like
        Coordinate vector.
    u : array_like
        Incident electric field.
    v : array_like
        Outgoing electric field.
    hu : array_like, optional
        Incident magnetic field.
    hv : array_like, optional
        Outgoing magnetic field.

    Returns
    -------
    t : float
        Power coupling efficiency.
    """
    x = np.asarray(x).reshape(-1)
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)

    if hu is not None and hv is not None:
        hu = np.asarray(hu).reshape(-1)
        hv = np.asarray(hv).reshape(-1)

        if not (len(x) == len(u) == len(v) == len(hu) == len(hv)):
            raise ValueError("x, u, v, hu, hv must have the same length.")

        # calculate 1D overlap with E and H
        uu = np.trapezoid(u * np.conj(hu), x)
        vv = np.trapezoid(v * np.conj(hv), x)
        uv = np.trapezoid(u * np.conj(hv), x)
        vu = np.trapezoid(v * np.conj(hu), x)

        t = abs(np.real(uv * vu / vv) / np.real(uu))
    else:
        if not (len(x) == len(u) == len(v)):
            raise ValueError("x, u, v must have the same length.")

        # calculate 1D overlap of u and v
        uu = np.trapezoid(np.conj(u) * u, x)
        vv = np.trapezoid(np.conj(v) * v, x)
        uv = np.trapezoid(np.conj(u) * v, x)

        t = abs(uv) / (np.sqrt(uu) * np.sqrt(vv))

    return t
