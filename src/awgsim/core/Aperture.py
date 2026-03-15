from .Waveguide import Waveguide

class Aperture(Waveguide):
    """
    Aperture class.

    Represents a waveguide cross section to query normal modes and
    calculate overlap for butt coupling.
    """

    def index(self, *args, **kwargs):
        raise NotImplementedError("Aperture does not expose index().")

    def groupindex(self, *args, **kwargs):
        raise NotImplementedError("Aperture does not expose groupindex().")
