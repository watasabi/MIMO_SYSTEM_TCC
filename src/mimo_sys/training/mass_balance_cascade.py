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
import torch

__all__ = [
    "apply_level_blend",
    "fit_level_blend",
    "fit_mass_balance",
    "fit_mass_balance_simulation",
    "mass_balance_cascade_forecast",
    "mass_balance_cascade_forecast_torch",
]


def fit_mass_balance(
    nivel_res: npt.NDArray,
    vazao_entrada: npt.NDArray,
    vazao_distribuicao: npt.NDArray,
    exclude: npt.NDArray | None = None,
) -> tuple[float, float]:
    """Least-squares fit of `dy2[k] = a * mean(q[k], q[k-1]) + b`.

    Fit on the training split only, in the preprocessor's output space
    (same units as whatever is fed back in at forecast time — don't mix
    original-scale and normalised-scale arrays between fit and use).

    Args:
        nivel_res: `(n,)` reservoir level, `y2`.
        vazao_entrada: `(n,)` input flow, `u4`.
        vazao_distribuicao: `(n,)` distribution flow, `y1`.
        exclude: `(n,)` rows left out of the fit (sensor failures, see
            `features.bad_sample_mask`); a step is dropped when either
            of its two rows is.

    Returns:
        `(a, b)`.
    """
    q = vazao_entrada - vazao_distribuicao
    dq = (q[1:] + q[:-1]) / 2.0
    dy2 = np.diff(nivel_res)
    if exclude is not None:
        keep = ~(exclude[1:] | exclude[:-1])
        dq, dy2 = dq[keep], dy2[keep]
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
    leak: float = 0.0,
    level_ref: float = 0.0,
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
        leak: `lambda` of the leaky integrator of
            `docs/1_analise-dados-reservatorio.md` §5: each step also
            subtracts `lambda * (level - level_ref)` (an uninstrumented,
            level-dependent outflow). 0 is the plain cascade.
        level_ref: Level around which the leak acts.

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
    anchor = np.asarray(level_anchor, dtype=float)
    if leak == 0.0:
        return anchor[:, None] + np.cumsum(delta, axis=1)
    out = np.empty_like(delta)
    level = anchor
    for k in range(horizon):
        level = level + delta[:, k] - leak * (level - level_ref)
        out[:, k] = level
    return out


def fit_mass_balance_simulation(  # noqa: PLR0913, PLR0914
    nivel_res: npt.NDArray,
    vazao_entrada: npt.NDArray,
    vazao_distribuicao: npt.NDArray,
    horizon: int = 12,
    exclude: npt.NDArray | None = None,
    stride: int = 1,
) -> tuple[float, float, float, float]:
    """`(a, b, leak, level_ref)` fitted on the 12-step simulation error.

    `fit_mass_balance` minimises the one-step error; the cascade is
    then run for `horizon` steps, where a small bias in `b` or a missing
    leak integrates. This fits the three parameters on exactly that use:
    windows of the training split, the measured inflow and outflow fed
    in, the simulated level compared with the measured one at every
    step. Starts from the one-step solution with no leak.

    Args:
        nivel_res, vazao_entrada, vazao_distribuicao: `(n,)` training rows.
        horizon: Steps simulated per window.
        exclude: Rows to keep out (sensor failures); windows touching
            one are skipped.
        stride: Distance between window origins.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415

    n = len(nivel_res)
    bad = np.zeros(n, bool) if exclude is None else exclude
    origins = np.array(
        [
            o
            for o in range(1, n - horizon + 1, stride)
            if not bad[o - 1 : o + horizon].any()
        ]
    )
    steps = origins[:, None] + np.arange(horizon)
    inflow, outflow = vazao_entrada[steps], vazao_distribuicao[steps]
    truth = nivel_res[steps]
    anchor = nivel_res[origins - 1]
    last = vazao_entrada[origins - 1] - vazao_distribuicao[origins - 1]
    level_ref = float(np.mean(nivel_res[~bad]))
    a0, b0 = fit_mass_balance(
        nivel_res, vazao_entrada, vazao_distribuicao, exclude
    )

    def residual(p: npt.NDArray) -> npt.NDArray:
        sim = mass_balance_cascade_forecast(
            anchor, inflow, outflow, p[0], p[1], last, p[2], level_ref
        )
        return (sim - truth).ravel()

    fit = least_squares(
        residual, x0=[a0, b0, 0.0], bounds=([0, -1, 0], [1, 1, 1])
    )
    a, b, leak = (float(v) for v in fit.x)
    return a, b, leak, level_ref


def mass_balance_cascade_forecast_torch(
    level_anchor: torch.Tensor,
    inflow: torch.Tensor,
    predicted_outflow: torch.Tensor,
    a: float,
    b: float,
    last_imbalance: torch.Tensor,
) -> torch.Tensor:
    """Autograd version of `mass_balance_cascade_forecast`.

    Same formula, differentiable w.r.t. `predicted_outflow` — used to
    train a forecaster's `vazao_distribuicao` head *through* the
    cascade, instead of combining the two post-hoc: the gradient of
    the `nivel_res` error reaches the outflow prediction, pushing the
    outflow forecast's own integrated (cumulative) error toward zero,
    which is what the cascade's accuracy actually depends on (see
    `docs/7_tecnica-cascata-fisica.md`). `a`/`b` stay fixed constants
    fit once by `fit_mass_balance` (not learned here); only the
    forecaster's weights receive gradient from this loss.

    Args:
        level_anchor: `(batch,)`.
        inflow: `(batch, horizon)`, known future `vazao_entrada`.
        predicted_outflow: `(batch, horizon)`, the forecaster's own
            `vazao_distribuicao` prediction — the only tensor this
            needs gradient through.
        a: From `fit_mass_balance`.
        b: From `fit_mass_balance`.
        last_imbalance: `(batch,)`.

    Returns:
        `(batch, horizon)` `nivel_res` forecast.
    """
    q = inflow - predicted_outflow
    q_prev = torch.cat([last_imbalance[:, None], q[:, :-1]], dim=1)
    delta = a * (q + q_prev) / 2.0 + b
    return level_anchor[:, None] + torch.cumsum(delta, dim=1)


def fit_level_blend(
    y_true: npt.NDArray, direct: npt.NDArray, cascade: npt.NDArray
) -> npt.NDArray:
    """Per-step weight of the cascade against the direct `nivel_res` head.

    `nivel[h] = w[h] * cascade[h] + (1 - w[h]) * direct[h]`, `w` fitted
    by least squares and clipped to `[0, 1]`. The cascade integrates
    the predicted outflow, so its error grows with the step while the
    direct head's does not; the weights follow that (measured: ~1 at
    the first step, falling to ~0.6 with known inflow and ~0.1
    without). Fit on the validation forecasts only.

    Args:
        y_true, direct, cascade: `(n_origins, horizon)`.

    Returns:
        `(horizon,)` weights.
    """
    diff = cascade - direct
    target = y_true - direct
    num = (diff * target).sum(axis=0)
    den = (diff**2).sum(axis=0)
    return np.clip(num / np.maximum(den, 1e-12), 0.0, 1.0)


def apply_level_blend(
    weights: npt.NDArray, direct: npt.NDArray, cascade: npt.NDArray
) -> npt.NDArray:
    """`weights * cascade + (1 - weights) * direct`, per horizon step."""
    return weights * cascade + (1 - weights) * direct
