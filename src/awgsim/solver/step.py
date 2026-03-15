import numpy as np
from numpy.typing import ArrayLike


def step(
    x: ArrayLike
) -> np.ndarray:
    """Return the unit step function of ``x``.

    Values smaller than 0 map to 0, and values greater than or equal to 0 map to 1.

    Args:
        x: A scalar, list, tuple, or NumPy array.

    Returns:
        A NumPy array containing 0.0 and 1.0.
    """
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError("x must be a numeric scalar or array-like object") from e
    
    return np.where(x < 0, 0.0, 1.0)
