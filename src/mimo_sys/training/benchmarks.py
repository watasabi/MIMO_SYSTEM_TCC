"""Pieces of the temporal cross-validation benchmark.

Expanding-window folds, the forecast metrics of the article, the
re-estimated ARMAX and the adapter that lets tree ensembles use the
production forecast contract (`evaluation.forecast_origins`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from mimo_sys.training.data import INPUT_COLS, N_TARGETS, TARGET_IDX

ARMAX_RIDGE = 1e-6
ARMAX_ELS_ITERATIONS = 5
ARMAX_RESIDUAL_SWEEPS = 3
ARMAX_CLIP = 20.0  # standard deviations; keeps unstable candidates finite
SEASON = 24
INFLOW = INPUT_COLS.index("vazao_entrada")
# Lags given to the tree ensembles. Flattening a whole 168 h window gives
# ~1400 features, and the histograms of a multi-output tree grow with
# features x outputs x nodes: tens of GB on this problem.
TREE_LAGS = (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 168)


@dataclass(frozen=True)
class TemporalFold:
    """Rows `[0, train_end)`, `[train_end, val_end)`, `[val_end, test_end)`."""

    train_end: int
    val_end: int
    test_end: int


def temporal_folds(
    n_rows: int, first_train: int = 888, block: int = 220, n_folds: int = 5
) -> list[TemporalFold]:
    """Expanding-window folds, each validated and tested on later blocks.

    Args:
        n_rows: Rows in the series.
        first_train: Training rows of the first fold.
        block: Rows of each validation and test block.
        n_folds: Number of folds.

    Returns:
        One `TemporalFold` per fold; the test blocks do not overlap.
    """
    folds = [
        TemporalFold(
            first_train + k * block,
            first_train + (k + 1) * block,
            first_train + (k + 2) * block,
        )
        for k in range(n_folds)
    ]
    if folds[-1].test_end > n_rows:
        raise ValueError("the folds do not fit in the series")
    return folds


def valid_origins(
    bad: npt.NDArray, start: int, n_origins: int, horizon: int
) -> npt.NDArray:
    """Origins whose `horizon` rows touch no flagged sensor failure.

    Args:
        bad: `(n_rows,)` failure mask of the whole series.
        start: Row of the first origin.
        n_origins: Consecutive origins.
        horizon: Rows forecast from each origin.

    Returns:
        `(n_origins,)` boolean mask.
    """
    flagged = np.concatenate([[0], np.cumsum(bad)])
    rows = start + np.arange(n_origins)
    return flagged[rows + horizon] - flagged[rows] == 0


def seasonal_scale(y_train: npt.NDArray, season: int = SEASON) -> npt.NDArray:
    """MASE denominator: training MAE of the seasonal naive forecast."""
    return np.abs(y_train[season:] - y_train[:-season]).mean(axis=0)


def forecast_metrics(
    y: npt.NDArray, pred: npt.NDArray, scale: npt.NDArray
) -> dict[str, npt.NDArray]:
    """R², RMSE, MAE, sMAPE (%), MASE and KGE of each output.

    Args:
        y: `(n, n_outputs)` measured values.
        pred: `(n, n_outputs)` forecasts.
        scale: `(n_outputs,)` denominator of the MASE.

    Returns:
        Metric name -> `(n_outputs,)` values.
    """
    err = y - pred
    mae = np.abs(err).mean(axis=0)
    y_centered = y - y.mean(axis=0)
    p_centered = pred - pred.mean(axis=0)
    y_std, p_std = y.std(axis=0), pred.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (y_centered * p_centered).mean(axis=0) / (y_std * p_std)
        kge = 1 - np.sqrt(
            (corr - 1) ** 2
            + (p_std / y_std - 1) ** 2
            + (pred.mean(axis=0) / y.mean(axis=0) - 1) ** 2
        )
    return {
        "r2": 1 - (err**2).sum(axis=0) / (y_centered**2).sum(axis=0),
        "rmse": np.sqrt((err**2).mean(axis=0)),
        "mae": mae,
        "smape": 100
        * np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(pred)), axis=0),
        "mase": mae / scale,
        "kge": kge,
    }


def tree_features(x: npt.NDArray, past: int, known: list[int]) -> npt.NDArray:
    """Selected past lags (`TREE_LAGS`) and the known future inputs.

    Args:
        x: `(n, past + horizon, n_nodes)` windows.
        past: Past rows in each window.
        known: Columns known over the horizon.

    Returns:
        `(n, n_features)`.
    """
    n = len(x)
    rows = [past - lag for lag in TREE_LAGS if lag <= past]
    return np.hstack(
        [x[:, rows].reshape(n, -1), x[:, past:][:, :, known].reshape(n, -1)]
    )


class TreeForecaster(nn.Module):
    """A fitted multi-output regressor behind the forecaster contract.

    The regressor predicts the change from each output's last observed
    value, flattened over the horizon, as the neural baselines do.

    Args:
        predict: Maps `(n, n_features)` to `(n, horizon * n_targets)`.
        past: Past rows in each window.
        horizon: Forecast horizon.
        known: Columns known over the horizon.
    """

    def __init__(
        self,
        predict: Callable[[npt.NDArray], npt.NDArray],
        past: int,
        horizon: int,
        known: list[int],
    ) -> None:
        super().__init__()
        self.predict = predict
        self.past = past
        self.horizon = horizon
        self.known = known

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast `(batch, horizon, N_TARGETS)`."""
        window = x.detach().cpu().numpy()
        features = tree_features(window, self.past, self.known)
        delta = np.asarray(self.predict(features)).reshape(
            -1, self.horizon, N_TARGETS
        )
        last = window[:, self.past - 1, TARGET_IDX]
        return torch.as_tensor(last[:, None] + delta, dtype=x.dtype)


@dataclass(frozen=True)
class Armax:
    """MIMO ARMAX estimated on standardized series."""

    order: tuple[int, int, int]
    theta: tuple[npt.NDArray, ...]
    y_mean: npt.NDArray
    y_std: npt.NDArray
    u_mean: npt.NDArray
    u_std: npt.NDArray


def _regressors(  # noqa: PLR0913 - the terms of one ARMAX equation
    y: npt.NDArray,
    u: npt.NDArray,
    e: npt.NDArray,
    t: int | npt.NDArray,
    order: tuple[int, int, int],
    out: int,
) -> npt.NDArray:
    """Lagged outputs, inputs and residuals of equation `out` at `t`."""
    na, nb, nc = order
    columns = [y[t - lag] for lag in range(1, na + 1)]
    columns += [u[t - lag] for lag in range(1, nb + 1)]
    columns += [e[t - lag][..., out, None] for lag in range(1, nc + 1)]
    return np.hstack(columns)


def _armax_residuals(
    model: Armax, y: npt.NDArray, u: npt.NDArray
) -> npt.NDArray:
    """One-step residuals; row `t` only depends on rows up to `t`."""
    t = np.arange(max(model.order), len(y))
    e = np.zeros_like(y)
    for _ in range(ARMAX_RESIDUAL_SWEEPS):
        new = np.zeros_like(y)
        for out, theta in enumerate(model.theta):
            x = _regressors(y, u, e, t, model.order, out)
            new[t, out] = y[t, out] - x @ theta
        e = new
    return e


def fit_armax(  # noqa: PLR0914 - linear on purpose
    y: npt.NDArray,
    u: npt.NDArray,
    order: tuple[int, int, int],
    bad: npt.NDArray | None = None,
) -> Armax:
    """Extended least squares, one equation per output.

    Args:
        y: `(n, 3)` training outputs.
        u: `(n, 4)` training inputs.
        order: `(n_a, n_b, n_c)`.
        bad: `(n,)` flagged rows; equations touching one are skipped.

    Returns:
        The fitted model.
    """
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    u_mean, u_std = u.mean(axis=0), u.std(axis=0)
    ys, us = (y - y_mean) / y_std, (u - u_mean) / u_std
    lag = max(order)
    t = np.arange(lag, len(ys))
    keep = np.ones(len(t), dtype=bool)
    if bad is not None:
        flagged = np.concatenate([[0], np.cumsum(bad)])
        keep = flagged[t + 1] - flagged[t - lag] == 0
    e = np.zeros_like(ys)
    theta: list[npt.NDArray] = []
    for _ in range(ARMAX_ELS_ITERATIONS):
        theta = []
        new = np.zeros_like(ys)
        for out in range(ys.shape[1]):
            x = _regressors(ys, us, e, t, order, out)
            x_fit = x[keep]
            gram = x_fit.T @ x_fit + ARMAX_RIDGE * np.eye(x.shape[1])
            coef = np.linalg.solve(gram, x_fit.T @ ys[t[keep], out])
            theta.append(coef)
            new[t, out] = ys[t, out] - x @ coef
        e = new
    return Armax(order, tuple(theta), y_mean, y_std, u_mean, u_std)


def armax_forecast(  # noqa: PLR0913 - the forecast contract
    model: Armax,
    y: npt.NDArray,
    u: npt.NDArray,
    origins: npt.NDArray,
    horizon: int,
    inflow_known: bool = True,
) -> npt.NDArray:
    """Free-run from each origin, without any output after it.

    Args:
        model: Fitted ARMAX.
        y: `(n, 3)` outputs; only rows before each origin are used.
        u: `(n, 4)` inputs, covering the horizons.
        origins: First forecast rows.
        horizon: Forecast steps.
        inflow_known: Otherwise the future inflow is taken from the same
            hours one day earlier, as in the `pumps` scenario.

    Returns:
        `(len(origins), horizon, 3)` forecasts in the original scale.
    """
    ys = (y - model.y_mean) / model.y_std
    us = (u - model.u_mean) / model.u_std
    residuals = _armax_residuals(model, ys, us)
    lag = max(model.order)
    n = len(origins)
    sim_y = np.zeros((lag + horizon, n, ys.shape[1]))
    sim_e = np.zeros_like(sim_y)
    sim_u = np.zeros((lag + horizon, n, us.shape[1]))
    for i, origin in enumerate(origins):
        sim_y[:lag, i] = ys[origin - lag : origin]
        sim_e[:lag, i] = residuals[origin - lag : origin]
        sim_u[:, i] = us[origin - lag : origin + horizon]
        if not inflow_known:
            shifted = us[origin - SEASON : origin - SEASON + horizon, INFLOW]
            sim_u[lag:, i, INFLOW] = shifted
    for step in range(lag, lag + horizon):
        for out, theta in enumerate(model.theta):
            x = _regressors(sim_y, sim_u, sim_e, step, model.order, out)
            sim_y[step, :, out] = np.clip(x @ theta, -ARMAX_CLIP, ARMAX_CLIP)
    forecast = sim_y[lag:].transpose(1, 0, 2)
    return forecast * model.y_std + model.y_mean
