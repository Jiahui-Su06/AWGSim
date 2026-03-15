from dataclasses import dataclass
from typing import Any


@dataclass
class SimulationOptions:
    """
    Option set for AWG simulations.

    Parameters
    ----------
    mode_type : str
        Aperture mode approximation, one of {'rect', 'gaussian', 'solve'}.
    use_magnetic_field : bool
        Whether to use magnetic field in overlap integrals.
    insertion_loss : float
        Overall insertion loss in dB.
    propagation_loss : float
        Propagation loss in dB.
    phase_error_variance : float
        Random phase error variance applied to each waveguide.
    custom_input_field : Any | None
        Arbitrary input field object instead of auto-generated field.
    """

    mode_type: str = "gaussian"
    use_magnetic_field: bool = False
    insertion_loss: float = 0.0
    propagation_loss: float = 0.0
    phase_error_variance: float = 0.0
    custom_input_field: Any | None = None

    def __post_init__(self) -> None:
        self.mode_type = self.mode_type.lower()
        if self.mode_type not in {"rect", "gaussian", "solve"}:
            raise ValueError(
                "mode_type must be one of {'rect', 'gaussian', 'solve'}."
            )

        if not isinstance(self.use_magnetic_field, bool):
            raise TypeError("use_magnetic_field must be bool.")

        if self.insertion_loss < 0:
            raise ValueError("insertion_loss must be >= 0.")

        if self.propagation_loss < 0:
            raise ValueError("propagation_loss must be >= 0.")

        if self.phase_error_variance < 0:
            raise ValueError("phase_error_variance must be >= 0.")
