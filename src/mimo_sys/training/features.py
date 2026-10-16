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

__all__ = [
    "MASS_BALANCE_COL",
    "add_ewma_feature",
    "add_mass_balance_feature",
]

MASS_BALANCE_COL = "balanco_massa_cumsum"


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
