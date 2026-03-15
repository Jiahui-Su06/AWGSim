import numpy as np
from typing import Any
from tqdm import tqdm

from .SimulationOptions import SimulationOptions
from .simulate import simulate


def spectrum(
    model,
    lmbda: float,
    bandwidth: float,
    options: "SimulationOptions | dict[str, Any] | None" = None,
    points: int = 1000,
    samples: int = 100,
) -> dict[str, np.ndarray]:
    """
    Simulate the entire AWG over a wavelength range and extract transmission.

    Parameters
    ----------
    model : AWG
        AWG model object.
    lmbda : float
        Center wavelength.
    bandwidth : float
        Total wavelength span.
    options : SimulationOptions | dict | None, optional
        Simulation options. If None, defaults are used.
    points : int, optional
        Number of sampling points used inside each simulation.
    samples : int, optional
        Number of wavelength samples across the bandwidth.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
        - "wavelength": sampled wavelength array
        - "transmission": transmission matrix of shape (samples, model.No)
    """
    if options is None:
        options = SimulationOptions()
    elif isinstance(options, dict):
        options = SimulationOptions(**options)

    # generate simulation wavelengths
    wvl = lmbda + np.linspace(-0.5, 0.5, samples) * bandwidth

    # calculate transmission data
    T = np.zeros((samples, model.No), dtype=float)

    iterator = wvl
    if tqdm is not None:
        iterator = tqdm(wvl, desc="Computing AWG spectrum", unit="wl")

    for i, wl in enumerate(iterator):
        R = simulate(
            model,
            wl,
            options=options,
            points=points,
        )
        T[i, :] = np.asarray(R["transmission"]).reshape(-1)

    return {
        "wavelength": wvl,
        "transmission": T,
    }