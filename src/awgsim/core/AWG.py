from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .Waveguide import Waveguide
from .Aperture import Aperture
from ..material import Material

@dataclass
class AWG:
    """
    Arrayed Waveguide Grating model.

    Parameters
    ----------
    lambda_c : float
        Design center wavelength.
    clad : Material | float | callable | str
        Top cladding material.
    core : Material | float | callable | str
        Core material.
    subs : Material | float | callable | str
        Bottom cladding material.
    w : float
        Array waveguide core width.
    h : float
        Waveguide core height.
    t : float
        Waveguide slab thickness for rib waveguides.
    N : int
        Number of arrayed waveguides.
    m : int
        Diffraction order.
    R : float
        Grating radius of curvature (focal length).
    d : float
        Array aperture spacing.
    g : float
        Gap width between array apertures.
    Ni : int
        Number of input waveguides.
    No : int
        Number of output waveguides.
    wi : float
        Input waveguide aperture width.
    wo : float
        Output waveguide aperture width.
    di : float
        Input waveguide spacing.
    do : float
        Output waveguide spacing.
    li : float
        Input waveguide offset spacing.
    lo : float
        Output waveguide offset spacing.
    L0 : float
        Minimum waveguide length offset.
    defocus : float
        Added defocus to R.
    confocal : bool
        Use confocal arrangement rather than Rowland.
    """

    lambda_c: float = 1.550
    clad: "Material | float | callable | str" = field(default_factory=lambda: Material("SiO2"))
    core: "Material | float | callable | str" = field(default_factory=lambda: Material("Si"))
    subs: "Material | float | callable | str" = field(default_factory=lambda: Material("SiO2"))

    w: float = 0.450
    h: float = 0.220
    t: float = 0.0

    N: int = 40
    m: int = 30
    R: float = 100.0
    d: float = 1.300
    g: float = 0.200

    Ni: int = 1
    No: int = 8

    wi: float = 1.000
    wo: float = 1.200

    di: float = 0.000
    do: float = 3.600

    li: float = 0.0
    lo: float = 0.0
    L0: float = 0.0

    defocus: float = 0.0
    confocal: bool = False

    # calculated properties
    dl: float = field(init=False)
    wg: float = field(init=False)

    def __post_init__(self) -> None:
        self._validate_positive("lambda_c", self.lambda_c)
        self._validate_positive("w", self.w)
        self._validate_positive("h", self.h)
        self._validate_nonnegative("t", self.t)
        self._validate_positive("N", self.N)
        self._validate_positive("m", self.m)
        self._validate_positive("R", self.R)
        self._validate_positive("d", self.d)
        self._validate_nonnegative("g", self.g)
        self._validate_positive("Ni", self.Ni)
        self._validate_positive("No", self.No)
        self._validate_positive("wi", self.wi)
        self._validate_positive("wo", self.wo)
        self._validate_nonnegative("di", self.di)
        self._validate_nonnegative("do", self.do)
        self._validate_nonnegative("li", self.li)
        self._validate_nonnegative("lo", self.lo)
        self._validate_nonnegative("L0", self.L0)
        self._validate_nonnegative("defocus", self.defocus)

        self.clad = self._ensure_material(self.clad)
        self.core = self._ensure_material(self.core)
        self.subs = self._ensure_material(self.subs)

        self.wg = self.d - self.g
        if self.wg <= 0:
            raise ValueError("Computed array aperture width wg = d - g must be positive.")

        # Equivalent to MATLAB:
        # nc = obj.getArrayWaveguide().index(obj.lambda_c, 1);
        nc = self.get_array_waveguide().index(self.lambda_c, 1)
        nc0 = float(np.asarray(nc).reshape(-1)[0])

        # Equivalent to MATLAB:
        # obj.dl = obj.m * obj.lambda_c / nc;
        self.dl = self.m * self.lambda_c / nc0

    @staticmethod
    def _ensure_material(value) -> "Material":
        if isinstance(value, Material):
            return value
        return Material(value)

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    @staticmethod
    def _validate_nonnegative(name: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"{name} must be nonnegative.")

    def get_slab_waveguide(self) -> "Waveguide":
        return Waveguide(
            clad=self.clad,
            core=self.core,
            subs=self.subs,
            h=self.h,
            t=self.h,
        )

    def get_array_waveguide(self) -> "Waveguide":
        return Waveguide(
            clad=self.clad,
            core=self.core,
            subs=self.subs,
            w=self.w,
            h=self.h,
            t=self.t,
        )

    def get_input_aperture(self) -> "Aperture":
        return Aperture(
            clad=self.clad,
            core=self.core,
            subs=self.subs,
            w=self.wi,
            h=self.h,
        )

    def get_array_aperture(self) -> "Aperture":
        return Aperture(
            clad=self.clad,
            core=self.core,
            subs=self.subs,
            w=self.wg,
            h=self.h,
        )

    def get_output_aperture(self) -> "Aperture":
        return Aperture(
            clad=self.clad,
            core=self.core,
            subs=self.subs,
            w=self.wo,
            h=self.h,
        )
