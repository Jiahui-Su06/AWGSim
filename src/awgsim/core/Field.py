import numpy as np


def _data_format(D, sz: tuple[int, int]) -> np.ndarray:
    D = np.asarray(D)

    if D.size == 0:
        return np.zeros(sz, dtype=complex)

    if sz[0] > 1 and sz[1] > 1:
        if D.shape != sz:
            raise ValueError(
                "Wrong data format. The field must contain the same number "
                "of rows as the y-coordinate points and the same number of "
                "columns as the x-coordinate points."
            )
        return D.astype(complex)

    expected_len = max(sz)
    if D.ndim != 1:
        D = D.reshape(-1)

    if len(D) != expected_len:
        raise ValueError(
            "Wrong data format. Expecting field data to be the same size "
            "as the coordinate elements."
        )

    if sz[0] > 1 and sz[1] == 1:
        return D.reshape(sz).astype(complex)

    if sz[1] > 1 and sz[0] == 1:
        return D.reshape(sz).astype(complex)

    return D.reshape(sz).astype(complex)


class Field:
    """
    Transverse electromagnetic field description.

    Parameters
    ----------
    X : ArrayLike | tuple | list
        Coordinate data:
        - 1D vector: x coordinates only
        - (x, y): bidimensional coordinates
        - (None, y) or ([], y): y only
    E : ArrayLike | list | tuple
        Electric field data:
        - vector/array -> mapped to Ex by default
        - (Ex, Ey, Ez)
    H : ArrayLike | list | tuple | None
        Magnetic field data:
        - vector/array -> mapped to Hx by default
        - (Hx, Hy, Hz)
    """

    def __init__(self, X, E, H=None):
        self._scalar = True
        self._dimens = 1
        self._xdata: list[np.ndarray] = [np.array([], dtype=float), np.array([], dtype=float)]

        x = np.array([], dtype=float)
        y = np.array([], dtype=float)

        if X is None:
            raise ValueError("At least one coordinate vector must be provided.")

        if isinstance(X, (list, tuple)):
            if len(X) < 1:
                raise ValueError("At least one coordinate vector must be provided.")

            x0 = X[0]
            if x0 is not None and np.size(x0) > 0:
                x = np.asarray(x0, dtype=float).reshape(-1)

            if len(X) > 1:
                y0 = X[1]
                if y0 is not None and np.size(y0) > 0:
                    y = np.asarray(y0, dtype=float).reshape(-1)

                self._dimens = 3
                if x.size == 0:
                    self._dimens = 2
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim > 2 or (arr.ndim == 2 and min(arr.shape) > 1):
                raise ValueError("Wrong coordinate format. Must be a 1-D vector.")
            x = arr.reshape(-1)
            self._dimens = 1

        if x.size == 0 and y.size == 0:
            raise ValueError("At least one coordinate vector must be provided.")

        if not np.isrealobj(x) or not np.isrealobj(y):
            raise ValueError("Coordinate vectors must be real numbers.")

        self._xdata = [x, y]

        sz = (max(1, len(y)), max(1, len(x)))

        if E is None:
            raise ValueError("Electric field data is empty.")

        self._edata = self._parse_field_data(E, sz, electric=True)
        self._hdata = self._parse_field_data(H, sz, electric=False)

    def _parse_field_data(self, U, sz: tuple[int, int], electric: bool) -> list[np.ndarray]:
        if U is None:
            return [
                np.zeros(sz, dtype=complex),
                np.zeros(sz, dtype=complex),
                np.zeros(sz, dtype=complex),
            ]

        Ux = np.array([], dtype=complex)
        Uy = np.array([], dtype=complex)
        Uz = np.array([], dtype=complex)

        if isinstance(U, (list, tuple)):
            self._scalar = False

            if len(U) > 0 and U[0] is not None and np.size(U[0]) > 0:
                Ux = np.asarray(U[0]).reshape(np.asarray(U[0]).shape if self._dimens > 2 else (-1,))
            if len(U) > 1 and U[1] is not None and np.size(U[1]) > 0:
                Uy = np.asarray(U[1]).reshape(np.asarray(U[1]).shape if self._dimens > 2 else (-1,))
            if len(U) > 2 and U[2] is not None and np.size(U[2]) > 0:
                Uz = np.asarray(U[2]).reshape(np.asarray(U[2]).shape if self._dimens > 2 else (-1,))
        else:
            if electric:
                self._scalar = True
            Ux = np.asarray(U).reshape(np.asarray(U).shape if self._dimens > 2 else (-1,))

        return [
            _data_format(Ux, sz),
            _data_format(Uy, sz),
            _data_format(Uz, sz),
        ]

    @property
    def x(self) -> np.ndarray:
        return self._xdata[0]

    @property
    def y(self) -> np.ndarray:
        return self._xdata[1]

    @property
    def Ex(self) -> np.ndarray:
        return self._edata[0]

    @property
    def Ey(self) -> np.ndarray:
        return self._edata[1]

    @property
    def Ez(self) -> np.ndarray:
        return self._edata[2]

    @property
    def Hx(self) -> np.ndarray:
        return self._hdata[0]

    @property
    def Hy(self) -> np.ndarray:
        return self._hdata[1]

    @property
    def Hz(self) -> np.ndarray:
        return self._hdata[2]

    @property
    def E(self) -> np.ndarray:
        if self.is_scalar():
            return self._edata[0]
        return np.stack(self._edata, axis=2)

    @property
    def H(self) -> np.ndarray:
        if self.is_scalar():
            return self._hdata[0]
        return np.stack(self._hdata, axis=2)

    def poynting(self) -> np.ndarray:
        """
        Return z component of the Poynting vector
        (or field intensity if no magnetic field is available).
        """
        if self.has_magnetic():
            return self.Ex * np.conj(self.Hy) - self.Ey * np.conj(self.Hx)
        return self.Ex * np.conj(self.Ex)

    def power(self) -> complex:
        """
        Return total power carried by the field.
        For 1D fields this is linear power density.
        """
        S = self.poynting()

        if self._dimens == 3:
            return np.trapezoid(np.trapezoid(S, self.y, axis=0), self.x, axis=0)

        if self._dimens == 1:
            return np.trapezoid(S.reshape(-1), self.x)

        return np.trapezoid(S.reshape(-1), self.y)

    def normalize(self, P: float | complex = 1) -> "Field":
        P0 = self.power()
        scale = np.sqrt(P) / np.sqrt(P0)

        self._edata = [scale * u for u in self._edata]
        self._hdata = [scale * u for u in self._hdata]
        return self

    def get_magnitude_e(self) -> np.ndarray:
        return np.sqrt(np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2)

    def get_magnitude_h(self) -> np.ndarray:
        return np.sqrt(np.abs(self.Hx) ** 2 + np.abs(self.Hy) ** 2 + np.abs(self.Hz) ** 2)

    def is_scalar(self) -> bool:
        return self._scalar

    def has_x(self) -> bool:
        return self._dimens in (1, 3)

    def has_y(self) -> bool:
        return self._dimens in (2, 3)

    def is_bidimensional(self) -> bool:
        return self._dimens > 2

    def has_electric(self) -> bool:
        return any(np.any(u) for u in self._edata)

    def has_magnetic(self) -> bool:
        return any(np.any(u) for u in self._hdata)

    def is_electromagnetic(self) -> bool:
        return self.has_electric() and self.has_magnetic()

    def get_size(self) -> tuple[int, int]:
        return (max(1, len(self.y)), max(1, len(self.x)))

    def offset_coordinates(self, dx: float = 0.0, dy: float = 0.0) -> None:
        if self._xdata[0].size > 0:
            self._xdata[0] = self._xdata[0] + dx
        if self._xdata[1].size > 0:
            self._xdata[1] = self._xdata[1] + dy