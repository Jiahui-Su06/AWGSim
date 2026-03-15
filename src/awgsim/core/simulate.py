from typing import Any

from .SimulationOptions import SimulationOptions
from .aw import aw
from .iw import iw
from .ow import ow
from .fpr1 import fpr1
from .fpr2 import fpr2


def simulate(
    model,
    lmbda: float,
    input: int | float | dict[str, Any] = 0,
    options: "SimulationOptions | None" = None,
    points: int = 2000,
) -> dict[str, Any]:
    """
    Simulate the entire AWG from input to output at a given wavelength.

    Parameters
    ----------
    model : AWG
        AWG model object.
    lmbda : float
        Wavelength.
    input : int | float | dict, optional
        Input channel index / offset / configuration.
        If a dict-like options object is accidentally passed here,
        it is treated as `options` and input is reset to 0,
        matching the MATLAB behavior.
    options : SimulationOptions | None, optional
        Simulation options. If None, defaults are used.
    points : int, optional
        Number of sampling points used in intermediate propagation steps.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "transmission"
        - "input_field"
        - "array_field"
        - "output_field"
    """
    # MATLAB behavior:
    # if nargin < 3
    #     input = 0;
    # elseif isstruct(input)
    #     varargin = {input, varargin{:}};
    #     input = 0;
    # end
    if isinstance(input, dict):
        if options is None:
            # convert dict to SimulationOptions if needed
            options = SimulationOptions(**input)
        input = 0

    if options is None:
        options = SimulationOptions()

    # Input waveguide field
    if options.custom_input_field is not None:
        F_iw = iw(model, lmbda, input, options.custom_input_field)
    else:
        F_iw = iw(
            model,
            lmbda,
            input,
            mode_type=options.mode_type,
            points=points,
        )

    # First free propagation region
    F_fpr1 = fpr1(model, lmbda, F_iw, points=points)

    # Array waveguides
    F_aw = aw(
        model,
        lmbda,
        F_fpr1,
        mode_type=options.mode_type,
        phase_error_var=options.phase_error_variance,
        insertion_loss=options.insertion_loss,
        propagation_loss=options.propagation_loss,
    )

    # Second free propagation region
    F_fpr2 = fpr2(model, lmbda, F_aw, points=points)

    # Output waveguide coupling
    T = ow(
        model,
        lmbda,
        F_fpr2,
        mode_type=options.mode_type,
    )

    return {
        "transmission": T,
        "input_field": F_iw,
        "array_field": F_aw,
        "output_field": F_fpr2,
    }