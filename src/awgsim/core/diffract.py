import numpy as np
from numpy.typing import ArrayLike


def diffract(
    lmbda: float,
    ui: ArrayLike,
    xi: ArrayLike,
    xf: ArrayLike,
    zf: float | ArrayLike,
    method: str = "rs",
):
    """
    One-dimensional Rayleigh-Sommerfeld diffraction integral calculation.

    This function numerically solves the Rayleigh-Sommerfeld integral from an
    input field vector at z=0 to output coordinates (xf, zf) in the x-z plane.

    Args:
        lmbda: Propagation wavelength.
        ui: Input-plane complex amplitude.
        xi: Input-plane coordinate vector.
        xf: Output-plane coordinate (single value or vector of coordinates).
        zf: Propagation distance between input and output planes. Can be a
            scalar or a vector with the same length as ``xf``.
        method: Diffraction method, either ``"rs"`` or ``"fr"``.

    Returns:
        A tuple ``(uf, xf)`` where:
            - ``uf`` is the output-plane field amplitude
            - ``xf`` is the output-plane coordinate vector

    NOTE:
        Uses the retarded phase convention: ``exp(-1j * k * z)``.
    """
    ui = np.asarray(ui, dtype=complex).reshape(-1)
    xi = np.asarray(xi, dtype=float).reshape(-1)
    xf = np.asarray(xf, dtype=float).reshape(-1)

    if np.isscalar(zf):
        zf = float(zf) * np.ones(len(xf))
    else:
        zf = np.asarray(zf, dtype=float).reshape(-1)
        if len(zf) != len(xf):
            raise ValueError("Coordinate vectors x and z must be the same length.")

    k = 2 * np.pi / lmbda
    uf = np.zeros(len(xf), dtype=complex)

    for i in range(len(xf)):
        r = np.sqrt((xf[i] - xi) ** 2 + zf[i] ** 2)

        if method == "rs":
            uf[i] = np.sqrt(zf[i] / (2 * np.pi)) * np.trapezoid(
                ui * (1j * k + 1 / r) * np.exp(-1j * k * r) / r**2,
                xi)
        elif method == "fr":
            uf[i] = np.sqrt(1j / lmbda / zf[i]) * np.exp(-1j * k * zf[i]) * np.trapezoid(
                ui * np.exp(-1j * k / 2 / zf[i] * (xi - xf[i]) ** 2), xi)
        else:
            raise ValueError("method must be 'rs' or 'fr'")

    return uf, xf
