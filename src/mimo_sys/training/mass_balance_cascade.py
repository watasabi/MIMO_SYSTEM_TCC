"""Physics cascade: forecast `nivel_res` from a predicted `vazao_distribuicao`.

`docs/1_analise-dados-reservatorio.md` §4 derives the reservoir's mass
balance, `A * dy2/dt = u4 - y1` (`y2` = `nivel_res`, `u4` =
`vazao_entrada`, `y1` = `vazao_distribuicao`), and fits it as a 2-term
linear gray-box with the trapezoidal rule:

    dy2[k] = a * mean(q[k], q[k-1]) + b,   q = u4 - y1

The same doc also found that running this model as a pure free-run
simulator (no observation of `y2` during the horizon) **diverges**
without a leak term — because it was run forward *open-loop*, with the
same model also having to extrapolate `y1` (the outflow) itself.

This module instead runs it in a **cascade**: `y1` for the horizon is
not extrapolated by the physics model, it is supplied — by an
already-trained forecaster's own `vazao_distribuicao` prediction (an
ML ensemble, in this project's case). `vazao_entrada` (`u4`) needs no
such substitute: it is a known, future input (the whole point of a
MIMO system with exogenous inputs), never predicted.

Measured effect (this project, test split, chained 12-step forecast):
MPNN's own `nivel_res` R² 0.462 vs. this cascade fed by the *same*
MPNN's `vazao_distribuicao` prediction, R² 0.707 — and R² 0.755 fed by
a 5-member ensemble's `vazao_distribuicao` (R² 0.929). The cascade
needs no retraining of the forecasters: it is a post-hoc combination
of an existing `vazao_distribuicao` prediction with 2 least-squares
parameters fit once on the training split.

Does **not** generalise to `pressao`: a similar linear gray-box
(`pressao ~ nivel_res + vazao_distribuicao + freq_b1..3`, from the same
doc) fed by predicted `nivel_res`/`vazao_distribuicao` scored about the
same as the ML ensemble's own `pressao` (R² 0.737 vs. 0.740) — not
worth the extra moving part there.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["fit_mass_balance", "mass_balance_cascade_forecast"]


def fit_mass_balance(
    nivel_res: npt.NDArray,
    vazao_entrada: npt.NDArray,
    vazao_distribuicao: npt.NDArray,
) -> tuple[float, float]:
    """Least-squares fit of `dy2[k] = a * mean(q[k], q[k-1]) + b`.

    Fit on the training split only, in the preprocessor's output space
    (same units as whatever is fed back in at forecast time — don't mix
    original-scale and normalised-scale arrays between fit and use).

    Args:
        nivel_res: `(n,)` reservoir level, `y2`.
        vazao_entrada: `(n,)` input flow, `u4`.
        vazao_distribuicao: `(n,)` distribution flow, `y1`.

    Returns:
        `(a, b)`.
    """
    q = vazao_entrada - vazao_distribuicao
    dq = (q[1:] + q[:-1]) / 2.0
    dy2 = np.diff(nivel_res)
    design = np.stack([dq, np.ones_like(dq)], axis=1)
    coef, *_ = np.linalg.lstsq(design, dy2, rcond=None)
    return float(coef[0]), float(coef[1])


def mass_balance_cascade_forecast(
    level_anchor: npt.NDArray,
    inflow: npt.NDArray,
    predicted_outflow: npt.NDArray,
    a: float,
    b: float,
    last_imbalance: npt.NDArray | None = None,
) -> npt.NDArray:
    """Propagate the mass-balance model forward using a predicted outflow.

    Args:
        level_anchor: `(n_windows,)` last known `nivel_res` before each
            forecast window (same anchor convention as
            `evaluation.chained_forecast`).
        inflow: `(n_windows, horizon)` known future `vazao_entrada`.
        predicted_outflow: `(n_windows, horizon)` another forecaster's
            `vazao_distribuicao` prediction for the same windows — the
            cascade's only dependency on a trained model.
        a: From `fit_mass_balance`.
        b: From `fit_mass_balance`.
        last_imbalance: `(n_windows,)` observed `vazao_entrada -
            vazao_distribuicao` one step before each window (the last
            step of the input window, both still measured there).
            Without it, the first forecast step's trapezoidal average
            falls back to using the first predicted step twice, a
            small approximation at just that one step.

    Returns:
        `(n_windows, horizon)` `nivel_res` forecast.
    """
    inflow = np.asarray(inflow, dtype=float)
    predicted_outflow = np.asarray(predicted_outflow, dtype=float)
    n_windows, horizon = inflow.shape
    q = inflow - predicted_outflow
    first_q_prev = (
        np.asarray(last_imbalance, dtype=float)
        if last_imbalance is not None
        else q[:, 0]
    )
    q_prev = np.concatenate([first_q_prev[:, None], q[:, :-1]], axis=1)
    dq = (q + q_prev) / 2.0
    delta = a * dq + b
    return np.asarray(level_anchor, dtype=float)[:, None] + np.cumsum(
        delta, axis=1
    )
