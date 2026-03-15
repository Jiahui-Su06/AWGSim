import numpy as np
from numpy.typing import ArrayLike

from .Field import Field
from ..solver import rectf, overlap


def pnorm(x: ArrayLike, u: ArrayLike) -> np.ndarray:
    """
    Normalize a 1D field so that integral |u|^2 dx = 1.
    """
    x = np.asarray(x).reshape(-1)
    u = np.asarray(u).reshape(-1)

    p = np.trapezoid(np.abs(u) ** 2, x)
    if p == 0:
        return u
    return u / np.sqrt(p)


def aw(
    model,
    lmbda: float,
    F0,
    mode_type: str = "gaussian",
    phase_error_var: float = 0.0,
    insertion_loss: float = 0.0,
    propagation_loss: float = 0.0,
):
    """
    Couple an input field to the AWG array apertures and propagate the fields
    along the waveguide array to the other end.

    Parameters
    ----------
    model : AWG
        AWG model object.
    lmbda : float
        Wavelength.
    F0 : Field
        Input field.
    mode_type : str, optional
        Aperture mode type: {'rect', 'gaussian', 'solve'}.
    phase_error_var : float, optional
        Phase error variance.
    insertion_loss : float, optional
        Insertion loss in dB.
    propagation_loss : float, optional
        Propagation loss in dB/cm.
    Returns
    -------
    Field
        Output field at the array apertures.
    """
    mode_type = mode_type.lower()
    if mode_type not in {"rect", "gaussian", "solve"}:
        raise ValueError(f"Wrong mode type '{mode_type}'.")

    x0 = np.asarray(F0.x).reshape(-1)
    u0 = np.asarray(F0.Ex).reshape(-1)   # same TODO as MATLAB: choose correct component
    P0 = F0.power()

    k0 = 2 * np.pi / lmbda
    nc = model.get_array_waveguide().index(lmbda, 1)
    nc = float(np.asarray(nc).reshape(-1)[0])

    # inputs
    pnoise = np.sqrt(phase_error_var) * np.random.randn(model.N)
    iloss = 10 ** (-abs(insertion_loss) / 10)

    aperture = model.get_array_aperture()

    Ex = np.zeros_like(u0, dtype=complex)

    for i in range(model.N):
        xc = (i - (model.N - 1) / 2) * model.d

        # get mode
        Fk = aperture.mode(lmbda, x=x0 - xc, mode_type=mode_type).normalize()

        # truncate applicable coupling range
        Ek = np.asarray(Fk.Ex).reshape(-1) * rectf((x0 - xc) / model.d)

        # normalize mode field
        Ek = pnorm(Fk.x, Ek)

        # coupling efficiency / amplitude
        t = overlap(x0, u0, Ek)

        # total phase delay
        L = i * model.dl + model.L0
        phase = k0 * nc * L + pnoise[i]

        # total losses
        ploss = 10 ** (-abs(propagation_loss * L * 1e-4) / 10)
        t = t * ploss * iloss**2

        # assemble waveguide field
        efield = P0 * t * Ek * np.exp(-1j * phase)

        # combine to total field
        Ex = Ex + efield

    return Field(x0, Ex)
