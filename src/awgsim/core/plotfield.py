import numpy as np
import matplotlib.pyplot as plt

from .Field import Field


def plotfield(
    X,
    Y=None,
    *,
    y=0,
    plot_phase: bool = False,
    plot_power: bool = False,
    unwrap_phase: bool = False,
    normalize_phase: bool = False,
    figure=None,
):
    """
    Plot field data.

    Parameters
    ----------
    X : ArrayLike | Field
        Coordinate vector or Field object.
    Y : ArrayLike | list | tuple | None, optional
        Field data if X is not a Field.
    y : ignored
        Kept only for loose compatibility with the MATLAB signature.
    plot_phase : bool, optional
        If True, plot |U|^2 and phase instead of real/imaginary parts.
    plot_power : bool, optional
        If True, add one more subplot row for power density / intensity.
    unwrap_phase : bool, optional
        If True, unwrap phase in 1D plots.
    normalize_phase : bool, optional
        If True, normalize phase by pi.
    figure : int | matplotlib.figure.Figure | None, optional
        Figure handle or figure number. If None, create a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    # If X is already a Field object, use it directly
    if hasattr(X, "poynting") and hasattr(X, "has_electric"):
        F = X
    else:
        F = Field(X, Y)

    rows = 1
    if F.is_electromagnetic():
        rows = 2
    if plot_power:
        rows += 1

    if figure is None:
        fig = plt.figure()
    elif isinstance(figure, plt.Figure):
        fig = figure
        fig.clf()
    else:
        fig = plt.figure(figure)
        fig.clf()

    if F.is_bidimensional():
        _plot_field_2d_layout(
            fig,
            F,
            rows,
            plot_phase=plot_phase,
            plot_power=plot_power,
            unwrap_phase=unwrap_phase,
            normalize_phase=normalize_phase,
        )
    else:
        _plot_field_1d_layout(
            fig,
            F,
            rows,
            plot_phase=plot_phase,
            plot_power=plot_power,
            unwrap_phase=unwrap_phase,
            normalize_phase=normalize_phase,
        )

    fig.tight_layout()
    return fig


def _plot_field_2d_layout(
    fig,
    F,
    rows: int,
    *,
    plot_phase: bool,
    plot_power: bool,
    unwrap_phase: bool,
    normalize_phase: bool,
):
    row_offset = 0

    if F.is_scalar():
        if F.has_electric():
            ax = fig.add_subplot(rows, 1, 1)
            _plot_field_2d(
                ax,
                F.x,
                F.y,
                F.E,
                "x",
                "y",
                "E",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            row_offset = 1

        if F.has_magnetic():
            ax = fig.add_subplot(rows, 1, row_offset + 1)
            _plot_field_2d(
                ax,
                F.x,
                F.y,
                F.H,
                "x",
                "y",
                "H",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            row_offset += 1

        if plot_power:
            ax = fig.add_subplot(rows, 1, row_offset + 1)
            _plot_power_2d(ax, F)

    else:
        if F.has_electric():
            ax = fig.add_subplot(rows, 3, 1)
            _plot_field_2d(
                ax,
                F.x,
                F.y,
                F.Ex,
                "x",
                "y",
                "Ex",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            ax = fig.add_subplot(rows, 3, 2)
            _plot_field_2d(
                ax,
                F.x,
                F.y,
                F.Ey,
                "x",
                "y",
                "Ey",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            ax = fig.add_subplot(rows, 3, 3)
            _plot_field_2d(
                ax,
                F.x,
                F.y,
                F.Ez,
                "x",
                "y",
                "Ez",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            row_offset = 1

        if F.has_magnetic():
            ax = fig.add_subplot(rows, 3, row_offset * 3 + 1)
            _plot_field_2d(
                ax,
                F.x,
                F.y,
                F.Hx,
                "x",
                "y",
                "Hx",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            ax = fig.add_subplot(rows, 3, row_offset * 3 + 2)
            _plot_field_2d(
                ax,
                F.x,
                F.y,
                F.Hy,
                "x",
                "y",
                "Hy",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            ax = fig.add_subplot(rows, 3, row_offset * 3 + 3)
            _plot_field_2d(
                ax,
                F.x,
                F.y,
                F.Hz,
                "x",
                "y",
                "Hz",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            row_offset += 1

        if plot_power:
            ax = fig.add_subplot(rows, 1, row_offset + 1)
            _plot_power_2d(ax, F)


def _plot_field_1d_layout(
    fig,
    F,
    rows: int,
    *,
    plot_phase: bool,
    plot_power: bool,
    unwrap_phase: bool,
    normalize_phase: bool,
):
    a = F.x
    t = "x"
    if F.has_y():
        a = F.y
        t = "y"

    row_offset = 0

    if F.is_scalar():
        if F.has_electric():
            ax = fig.add_subplot(rows, 1, 1)
            _plot_field_1d(
                ax,
                a,
                np.asarray(F.E).reshape(-1),
                t,
                "E",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            row_offset = 1

        if F.has_magnetic():
            ax = fig.add_subplot(rows, 1, row_offset + 1)
            _plot_field_1d(
                ax,
                a,
                np.asarray(F.H).reshape(-1),
                t,
                "H",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            row_offset += 1

        if plot_power:
            ax = fig.add_subplot(rows, 1, row_offset + 1)
            _plot_power_1d(ax, a, np.asarray(F.poynting()).reshape(-1), t)

    else:
        if F.has_electric():
            ax = fig.add_subplot(rows, 3, 1)
            _plot_field_1d(
                ax, a, np.asarray(F.Ex).reshape(-1), t, "Ex",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            ax = fig.add_subplot(rows, 3, 2)
            _plot_field_1d(
                ax, a, np.asarray(F.Ey).reshape(-1), t, "Ey",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            ax = fig.add_subplot(rows, 3, 3)
            _plot_field_1d(
                ax, a, np.asarray(F.Ez).reshape(-1), t, "Ez",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            row_offset = 1

        if F.has_magnetic():
            ax = fig.add_subplot(rows, 3, row_offset * 3 + 1)
            _plot_field_1d(
                ax, a, np.asarray(F.Hx).reshape(-1), t, "Hx",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            ax = fig.add_subplot(rows, 3, row_offset * 3 + 2)
            _plot_field_1d(
                ax, a, np.asarray(F.Hy).reshape(-1), t, "Hy",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            ax = fig.add_subplot(rows, 3, row_offset * 3 + 3)
            _plot_field_1d(
                ax, a, np.asarray(F.Hz).reshape(-1), t, "Hz",
                plot_phase=plot_phase,
                unwrap_phase=unwrap_phase,
                normalize_phase=normalize_phase,
            )
            row_offset += 1

        if plot_power:
            ax = fig.add_subplot(rows, 1, row_offset + 1)
            _plot_power_1d(ax, a, np.asarray(F.poynting()).reshape(-1), t)


def _plot_field_2d(
    ax,
    x,
    y,
    u,
    xname: str,
    yname: str,
    uname: str,
    *,
    plot_phase: bool,
    unwrap_phase: bool,
    normalize_phase: bool,
):
    u = np.asarray(u)

    if plot_phase:
        u1 = np.abs(u) ** 2
        u2 = np.angle(u)
        if unwrap_phase:
            u2 = np.unwrap(u2, axis=-1)
        if normalize_phase:
            u2 = u2 / np.pi
        utitle = f"|{uname}|^2"
    else:
        u1 = np.real(u)
        u2 = np.imag(u)
        utitle = f"Re{{{uname}}}"

    pcm = ax.pcolormesh(x, y, u1, shading="auto")
    fig = ax.figure
    fig.colorbar(pcm, ax=ax)

    cs = ax.contour(x, y, u2, colors="white", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8)

    ax.set_xlabel(f"{xname} (um)")
    ax.set_ylabel(f"{yname} (um)")
    ax.set_title(utitle)
    ax.tick_params(labelsize=12)


def _plot_field_1d(
    ax,
    x,
    u,
    xname: str,
    uname: str,
    *,
    plot_phase: bool,
    unwrap_phase: bool,
    normalize_phase: bool,
):
    if plot_phase:
        u1 = np.abs(u) ** 2
        u2 = np.angle(u)
        if unwrap_phase:
            u2 = np.unwrap(u2)
        if normalize_phase:
            u2 = u2 / np.pi
        u1label = f"|{uname}|^2"
        u2label = f"phi({uname})"
    else:
        u1 = np.real(u)
        u2 = np.imag(u)
        u1label = f"Re{{{uname}}}"
        u2label = f"Im{{{uname}}}"

    ax2 = ax.twinx()

    ax.plot(x, u1, linewidth=2)
    ax.set_ylabel(u1label)
    ax.set_xlabel(f"{xname} (um)")
    ax.tick_params(labelsize=12)

    ax2.plot(x, u2, linewidth=2)
    ax2.set_ylabel(u2label)
    ax2.tick_params(labelsize=12)

    ax.set_xlim(np.min(x), np.max(x))

    if plot_phase and normalize_phase:
        meany = np.mean(u2)
        miny = meany + min(-0.5, np.min(u2) - meany)
        maxy = meany + max(0.5, np.max(u2) - meany)
        ax2.set_ylim(miny, maxy)

        ticks = ax2.get_yticks()
        ax2.set_yticklabels([f"{tick:g}π" for tick in ticks])


def _plot_power_1d(ax, x, s, xname: str):
    ax.plot(x, np.real(s), linewidth=2)
    ax.set_xlabel(f"{xname} (um)")
    ax.set_ylabel("Power density")
    ax.set_title("Poynting / Intensity")
    ax.tick_params(labelsize=12)
    ax.set_xlim(np.min(x), np.max(x))


def _plot_power_2d(ax, F):
    s = np.asarray(F.poynting())
    pcm = ax.pcolormesh(F.x, F.y, np.real(s), shading="auto")
    ax.figure.colorbar(pcm, ax=ax)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Poynting / Intensity")
    ax.tick_params(labelsize=12)
