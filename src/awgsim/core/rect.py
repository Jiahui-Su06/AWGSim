from numpy.typing import ArrayLike
import numpy as np

from .step import step


def rect(x: ArrayLike) -> np.ndarray:
    """Return the rectangular function evaluated at ``x``.

    The function is defined as::

        rect(x) = step(1/2 - x) * step(x + 1/2)

    Args:
        x: A numeric scalar or array-like input.

    Returns:
        The rectangular function evaluated element-wise on ``x``.
    """
    return step(0.5 - x) * step(x + 0.5)
