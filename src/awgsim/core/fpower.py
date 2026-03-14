import numpy as np
from numpy.typing import ArrayLike


def fpower(
    x: ArrayLike,
    Ex: ArrayLike,
    Hy: ArrayLike,
    Ey: ArrayLike | None = None,
    Hx: ArrayLike | None = None,
) -> float:
    """Calculate the optical power from field components.

    Args:
        x: Sampling points along the integration axis.
        Ex: x-component of the electric field.
        Hy: y-component of the magnetic field.
        Ey: y-component of the electric field.
        Hx: x-component of the magnetic field.

    Returns:
        The integrated optical power.

    Raises:
        ValueError: If ``Ey`` and ``Hx`` are not provided together.
    """
    if (Ey is None) != (Hx is None):
        raise ValueError("Ey and Hx must be provided together")

    x = np.asarray(x)
    Ex = np.asarray(Ex)
    Hy = np.asarray(Hy)

    if Ey is not None:
        Ey = np.asarray(Ey)
    if Hx is not None:
        Hx = np.asarray(Hx)

    if Ey is not None and Hx is not None:
        Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
    else:
        Sz = 0.5 * np.real(Ex * np.conj(Hy))

    return float(np.trapezoid(Sz, x))
