import numpy as np
import pandas as pd


def analyse(results: dict) -> pd.DataFrame:
    """
    Perform analysis on AWG output spectrum.

    Parameters
    ----------
    results : dict
        Must contain:
        - "wavelength": 1D array
        - "transmission": 2D array with shape (n_wavelengths, n_channels)

    Returns
    -------
    pandas.DataFrame
        Analysis table with one column: "Value".
    """
    wavelength = np.asarray(results["wavelength"]).reshape(-1)
    transmission = np.asarray(results["transmission"])

    if transmission.ndim != 2:
        raise ValueError("transmission must be a 2D array.")
    if transmission.shape[0] != len(wavelength):
        raise ValueError("wavelength length must match transmission rows.")

    # avoid log10 issues at zero
    TdB = 10 * np.log10(np.maximum(transmission, np.finfo(float).tiny))

    num_channels = transmission.shape[1]
    center_channel = num_channels // 2  # MATLAB floor(num_channels/2)+1 -> Python 0-based

    # insertion loss
    IL = abs(np.max(TdB[:, center_channel]))

    # helper: find first/last index meeting condition
    def _find_last(mask: np.ndarray) -> int | None:
        idx = np.flatnonzero(mask)
        return int(idx[-1]) if idx.size else None

    def _find_first(mask: np.ndarray) -> int | None:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if idx.size else None

    # 10 dB bandwidth
    t0 = TdB[:, center_channel] - IL
    ic = int(np.argmax(t0))

    ia = _find_last(t0[: ic + 1] < -10)
    ib_rel = _find_first(t0[ic:] < -10)
    ib = ic + ib_rel if ib_rel is not None else None

    BW = np.nan
    if ia is not None and ib is not None:
        BW = (wavelength[ib] - wavelength[ia]) * 1e3

    # 3 dB bandwidth
    ia3 = _find_last(t0[: ic + 1] < -3)
    ib3_rel = _find_first(t0[ic:] < -3)
    ib3 = ic + ib3_rel if ib3_rel is not None else None

    BW3 = np.nan
    if ia3 is not None and ib3 is not None:
        BW3 = (wavelength[ib3] - wavelength[ia3]) * 1e3

    # use 3 dB region for crosstalk window, matching original code behavior
    ia_ct = ia3
    ib_ct = ib3

    NU = 0.0
    CS = 0.0
    XT = 0.0
    XTn = 0.0

    if num_channels > 1:
        # non-uniformity
        NU = abs(np.max(TdB[:, 0])) - IL

        # Adjacent Crosstalk
        if ia_ct is not None and ib_ct is not None:
            window = slice(ia_ct, ib_ct + 1)

            if num_channels < 3:
                if center_channel - 1 >= 0:
                    XT = np.max(TdB[window, center_channel - 1])
                else:
                    XT = np.max(TdB[window, center_channel + 1])
            else:
                xt1 = np.max(TdB[window, center_channel - 1])
                xt2 = np.max(TdB[window, center_channel + 1])
                XT = max(xt1, xt2)

            XT = XT - IL

            # Non-adjacent Crosstalk
            XTn = -100.0
            for i in range(num_channels):
                if i != center_channel:
                    xt = np.max(TdB[window, i])
                    XTn = max(XTn, xt)

            XTn = XTn - IL
        else:
            XT = np.nan
            XTn = np.nan

        # Channel spacing
        if num_channels < 3:
            if center_channel - 1 >= 0:
                ia_cs = int(np.argmax(TdB[:, center_channel - 1]))
                CS = 1e3 * abs(wavelength[ia_cs] - wavelength[ic])
            else:
                ia_cs = int(np.argmax(TdB[:, center_channel + 1]))
                CS = 1e3 * abs(wavelength[ia_cs] - wavelength[ic])
        else:
            ia_cs = int(np.argmax(TdB[:, center_channel - 1]))
            ib_cs = int(np.argmax(TdB[:, center_channel + 1]))

            sp1 = abs(wavelength[ia_cs] - wavelength[ic])
            sp2 = abs(wavelength[ib_cs] - wavelength[ic])
            CS = max(sp1, sp2) * 1e3

    table = pd.DataFrame(
        {
            "Value": [
                IL,
                NU,
                CS,
                BW3,
                BW,
                XT,
                XTn,
            ]
        },
        index=[
            "Insertion loss (dB)",
            "Loss non-uniformity (dB)",
            "Channel spacing (nm)",
            "-3dB bandwidth (nm)",
            "-10dB bandwidth (nm)",
            "Adjacent channel Crosstalk (dB)",
            "Non-adjacent channel crosstalk (dB)",
        ],
    )

    return table
