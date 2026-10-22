"""Physics-informed derived features, computed on the raw series.

`docs/1_analise-dados-reservatorio.md` §4-5 identifies the mass-balance
integral `cumsum(vazao_entrada - vazao_distribuicao)` as the variable
`nivel_res` actually tracks — but the model never sees it directly: it
only sees the raw instantaneous inputs and has to infer the integral
itself from a 24-step window, which may not be enough signal. Adding
the integral as an explicit extra input node gives the model the state
variable directly, instead of asking it to reconstruct it.

Computed over the **full, chronologically concatenated** series before
splitting — the integral must not reset at a split boundary — then
sliced back per split, so it never uses future rows relative to its
own position (`cumsum` at row `i` only sums rows `<= i`).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mimo_sys.preprocessors.timeseries import ewt_causal_components
from mimo_sys.training.data import FEATURE_COLS

__all__ = [
    "MASS_BALANCE_COL",
    "add_ewma_feature",
    "add_mass_balance_feature",
    "bad_sample_mask",
    "atrous_components",
    "ewt_online_components",
    "seasonal_components",
    "seasonal_median",
]

MASS_BALANCE_COL = "balanco_massa_cumsum"
# Physical plausibility limits (docs/1_analise-dados-reservatorio.md §2).
MIN_LEVEL_M = 1.0
MIN_PRESSURE_MCA = 5.0
MAX_LEVEL_JUMP_M_PER_H = 0.3


def add_mass_balance_feature(
    raw: npt.NDArray,
    inflow_idx: int,
    outflow_idx: int,
    window: int | None = None,
) -> npt.NDArray:
    """Append a mass-balance integral of the imbalance as a column.

    Args:
        raw: `(n_rows, n_nodes)`, chronologically ordered, no gaps.
        inflow_idx: Column of `vazao_entrada` (`u4`).
        outflow_idx: Column of `vazao_distribuicao` (`y1`).
        window: `None` sums the whole history up to each row (a true
            unbounded integral — but it then drifts to a scale the
            preprocessor's train-fitted scaler never saw in val/test,
            since the sum keeps growing). A positive int instead sums
            only the trailing `window` rows (rolling sum), which stays
            in a stationary range and is what generalises — use this
            for anything longer than the training split.

    Returns:
        `(n_rows, n_nodes + 1)`, the new column appended last.
    """
    imbalance = raw[:, inflow_idx] - raw[:, outflow_idx]
    if window is None:
        integral = np.cumsum(imbalance)
    else:
        padded = np.concatenate([np.zeros(window), imbalance])
        cumsum = np.cumsum(padded)
        integral = cumsum[window:] - cumsum[:-window]
    return np.concatenate([raw, integral[:, None]], axis=1)


def bad_sample_mask(
    raw: npt.NDArray, guard: int = 3, saturation: float = 199.0
) -> npt.NDArray:
    """Rows with a sensor failure, plus `guard` rows on each side.

    Rules from `docs/1_analise-dados-reservatorio.md` §2 (`nivel_res`
    below 1 m, a level jump above 0.3 m/h, `pressao` below 5 mca,
    negative `vazao_distribuicao`) plus flows at the sensor's full
    scale (~199.6 l/s, seen right after outages). The guard band
    covers the restart transients around each failure. Generic
    detectors (MAD, IsolationForest, LOF) were tried and flag tens to
    hundreds of normal rows; these physical rules do not.

    Args:
        raw: `(n, >= N_NODES)` rows in `FEATURE_COLS` order.
        guard: Rows marked before and after each failing row.
        saturation: Flow value treated as a saturated reading.

    Returns:
        `(n,)` boolean mask.
    """
    level = raw[:, FEATURE_COLS.index("nivel_res")]
    outflow = raw[:, FEATURE_COLS.index("vazao_distribuicao")]
    inflow = raw[:, FEATURE_COLS.index("vazao_entrada")]
    pressure = raw[:, FEATURE_COLS.index("pressao")]
    bad = (level < MIN_LEVEL_M) | (pressure < MIN_PRESSURE_MCA) | (outflow < 0)
    bad |= (outflow >= saturation) | (inflow >= saturation + 1)
    bad[1:] |= np.abs(np.diff(level)) > MAX_LEVEL_JUMP_M_PER_H
    if guard:
        idx = np.flatnonzero(bad)
        for k in range(-guard, guard + 1):
            bad[np.clip(idx + k, 0, len(bad) - 1)] = True
    return bad


def seasonal_median(
    x: npt.NDArray, period: int = 24, days: int = 7
) -> npt.NDArray:
    """Median of the same hour over the previous `days` periods.

    An estimate of a time-of-day setpoint (e.g. the pressure plateaus
    at ~28/26/20 mca): row `t` only uses rows `t - period, ...,
    t - days * period`, so for any horizon `<= period` its future
    values are already known at the forecast origin. The first rows
    use whatever previous periods exist (the row itself before one
    full period).
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for t in range(len(x)):
        past = x[t - period : max(t - (days + 1) * period, -1) : -period]
        out[t] = np.median(past) if len(past) else x[t]
    return out


def atrous_components(x: npt.NDArray, levels: int) -> npt.NDArray:
    """Causal Haar à trous decomposition that adds up to `x` exactly.

    `c_0 = x`, `c_j(t) = (c_{j-1}(t) + c_{j-1}(t - 2^(j-1))) / 2`,
    `w_j = c_{j-1} - c_j`; the sum `w_1 + ... + w_J + c_J` telescopes
    back to `x` (Renaud, Starck & Murtagh 2005). Row `t` only uses rows
    `t - 2^J + 1 .. t`; the first rows repeat `x[0]` as history. Detail
    `w_j` holds periods of roughly `2^j`..`2^(j+1)` rows, so `levels=4`
    on hourly data matches `evaluation.BANDS` (2-4h ... 16-32h, >32h).

    Returns:
        `(n, levels + 1)`: `[w_1, ..., w_J, c_J]`.
    """
    c = np.asarray(x, dtype=float)
    comps = []
    for j in range(1, levels + 1):
        lag = 2 ** (j - 1)
        shifted = np.concatenate([np.full(lag, c[0]), c[:-lag]])
        smooth = 0.5 * (c + shifted)
        comps.append(c - smooth)
        c = smooth
    comps.append(c)
    return np.stack(comps, axis=1)


def ewt_online_components(
    x: npt.NDArray,
    boundaries: npt.NDArray,
    window: int = 168,
    extend: int = 24,
) -> npt.NDArray:
    """Real-time EWT bands: row `t` is the last row of the EWT of `x[..t]`.

    Each row is what `preprocessors.ewt_causal_components` returns for
    the last observed row of a past-only window ending at `t`, i.e.
    exactly what production computes when row `t` arrives. Causal by
    construction, sums to `x` on every row; the price is the EWT's edge
    error, which the model sees identically in training and serving.

    Returns:
        `(n, n_bands)`.
    """
    x = np.asarray(x, dtype=float)
    rows = [
        ewt_causal_components(
            x[max(0, t - window + 1) : t + 1], boundaries, extend
        )[-1]
        for t in range(len(x))
    ]
    return np.stack(rows)


def seasonal_components(x: npt.NDArray, period: int = 24) -> npt.NDArray:
    """`[x(t) - x(t - period), x(t - period)]`, causal, sums to `x`.

    The second column (the seasonal-naive base) is already observed at
    a forecast origin for any horizon `<= period`, so it is a known
    future input; only the seasonal difference has to be predicted.
    The first `period` rows use `x[0]` as the base.
    """
    x = np.asarray(x, dtype=float)
    base = np.concatenate([np.full(period, x[0]), x[:-period]])
    return np.stack([x - base, base], axis=1)


def add_ewma_feature(
    raw: npt.NDArray, col_idx: int, span: float
) -> npt.NDArray:
    """Append an exponentially-weighted moving average of one column.

    A smoothed, bounded companion to `add_mass_balance_feature`'s
    unbounded cumulative sum — same causal guarantee (value at row `i`
    only depends on rows `<= i`), but stationary, since it is a
    weighted average rather than a running total.

    Args:
        raw: `(n_rows, n_nodes)`, chronologically ordered, no gaps.
        col_idx: Column to smooth (e.g. the mass-balance column just
            appended by `add_mass_balance_feature`).
        span: EWMA span in rows (`alpha = 2 / (span + 1)`).

    Returns:
        `(n_rows, n_nodes + 1)`, the new column appended last.
    """
    alpha = 2.0 / (span + 1.0)
    series = raw[:, col_idx]
    smoothed = np.empty_like(series)
    smoothed[0] = series[0]
    for i in range(1, len(series)):
        smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i - 1]
    return np.concatenate([raw, smoothed[:, None]], axis=1)
