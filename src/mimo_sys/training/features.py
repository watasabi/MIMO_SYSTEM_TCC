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

__all__ = ["MASS_BALANCE_COL", "add_mass_balance_feature"]

MASS_BALANCE_COL = "balanco_massa_cumsum"


def add_mass_balance_feature(
    raw: npt.NDArray, inflow_idx: int, outflow_idx: int
) -> npt.NDArray:
    """Append `cumsum(raw[:, inflow_idx] - raw[:, outflow_idx])` as a column.

    Args:
        raw: `(n_rows, n_nodes)`, chronologically ordered, no gaps.
        inflow_idx: Column of `vazao_entrada` (`u4`).
        outflow_idx: Column of `vazao_distribuicao` (`y1`).

    Returns:
        `(n_rows, n_nodes + 1)`, the new column appended last.
    """
    imbalance = raw[:, inflow_idx] - raw[:, outflow_idx]
    integral = np.cumsum(imbalance)
    return np.concatenate([raw, integral[:, None]], axis=1)
