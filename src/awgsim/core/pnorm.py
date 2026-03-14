import numpy as np
from numpy.typing import ArrayLike

from .fpower import fpower


def pnorm(
    x: ArrayLike,
    Ex: ArrayLike,
    Hy: ArrayLike,
    Ey: ArrayLike | None = None,
    Hx: ArrayLike | None = None,
):
    """Normalize fields to unit power or unit field norm.

    If ``Ex``, ``Hy``, ``Ey``, and ``Hx`` are provided, the fields are
    normalized by the total power computed from the full Poynting-vector
    expression.

    If only ``Ex`` and ``Hy`` are provided, the fields are normalized by
    the corresponding reduced power expression.

    If only ``Ex`` is provided, the field is normalized by the quantity
    returned by ``fpower(x, Ex)``, which may represent a field norm rather
    than physical power depending on how ``fpower`` is defined.

    Args:
        x: Sampling points along the transverse coordinate.
        Ex: x-component of the electric field.
        Hy: y-component of the magnetic field.
        Ey: y-component of the electric field.
        Hx: x-component of the magnetic field.

    Returns:
        Normalized field arrays. Returns one, two, or four arrays depending
        on which field components are provided.

    Raises:
        ValueError: If the field component combination is incomplete.
        ZeroDivisionError: If the computed power is zero.
    """
    Ex = np.asarray(Ex)
    Hy = np.asarray(Hy)

    if Ey is not None:
        Ey = np.asarray(Ey)
    if Hx is not None:
        Hx = np.asarray(Hx)

    if (Ey is None) != (Hx is None):
        raise ValueError("Ey and Hx must be provided together")

    if Ey is not None and Hx is not None:
        P = fpower(x, Ex, Hy, Ey, Hx)
        if P == 0:
            raise ZeroDivisionError("Computed power is zero")
        scale = np.sqrt(P)
        return Ex / scale, Hy / scale, Ey / scale, Hx / scale

    P = fpower(x, Ex, Hy)
    if P == 0:
        raise ZeroDivisionError("Computed power is zero")
    scale = np.sqrt(P)
    return Ex / scale, Hy / scale
