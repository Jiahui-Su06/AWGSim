import numpy as np

from ..solver import rectf, overlap


def ow(
    model,
    lmbda: float,
    F0,
    mode_type: str = "gaussian",
) -> np.ndarray:
    """
    Compute output waveguide coupling.

    Parameters
    ----------
    model : AWG
        AWG model object.
    lmbda : float
        Wavelength.
    F0 : Field
        Input field at the output slab / aperture plane.
    mode_type : str, optional
        Aperture mode type: {'rect', 'gaussian', 'solve'}.

    Returns
    -------
    np.ndarray
        Output coupling / transmission for each output waveguide.
    """
    mode_type = mode_type.lower()
    if mode_type not in {"rect", "gaussian", "solve"}:
        raise ValueError(f"Wrong mode type '{mode_type}'.")

    x0 = np.asarray(F0.x).reshape(-1)
    u0 = np.asarray(F0.Ex).reshape(-1)   # same TODO as MATLAB
    P0 = F0.power()

    aperture = model.get_output_aperture()

    T = np.zeros(model.No, dtype=float)
    pitch = max(model.do, model.wo)

    for i in range(model.No):
        xc = model.lo + (i - (model.No - 1) / 2) * pitch

        # get offset mode field
        Fk = aperture.mode(lmbda, x=x0 - xc, mode_type=mode_type)
        Ek = np.asarray(Fk.Ex).reshape(-1)

        # truncate applicable coupling range
        Ek = Ek * rectf((x0 - xc) / pitch)

        # compute transmission with respect to input power
        T[i] = np.real(P0 * overlap(x0, u0, Ek) ** 2)

    return T