"""Forecast evaluation in the original scale: metrics and spectral bands.

The evaluation contract is the one of the collection experiments:
chained forecast of `OUTPUT_WINDOW` steps over the test range,
reconstructed with the preprocessor's `inverse_transform`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn

from mimo_sys.preprocessors import TimeSeriesPreprocessor
from mimo_sys.training.data import (
    INPUT_WINDOW,
    OUTPUT_COLS,
    OUTPUT_WINDOW,
    TARGET_IDX,
)

# Bands anchored on the EDA (`docs/5_eda-series-temporais.md` §7):
# daily peak at ~24h, second harmonic at ~12h. Names avoid `>`/`+`,
# which MLflow rejects in metric names.
BANDS = [
    ("tendencia_acima_32h", 32.0, np.inf),
    ("diario_16-32h", 16.0, 32.0),
    ("harmonico_8-16h", 8.0, 16.0),
    ("curto_4-8h", 4.0, 8.0),
    ("ruido_2-4h", 2.0, 4.0),
]


def metrics_table(y_true: npt.NDArray, y_pred: npt.NDArray) -> pd.DataFrame:
    """MSE, MAE, R² and SMAPE per output, `(n, n_targets)` inputs."""
    err = y_true - y_pred
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    smape = 100 * np.mean(
        2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-12), axis=0
    )
    return pd.DataFrame(
        {
            "mse": (err**2).mean(axis=0),
            "mae": np.abs(err).mean(axis=0),
            "r2": 1 - (err**2).sum(axis=0) / ss_tot,
            "smape_%": smape,
        },
        index=OUTPUT_COLS,
    )


def amplitude_spectrum(x: npt.NDArray) -> npt.NDArray:
    """One-sided amplitude spectrum, `(n_freqs, n_outputs)`."""
    return np.abs(np.fft.rfft(x, axis=0)) * 2.0 / len(x)


def band_table(y_true: npt.NDArray, y_pred: npt.NDArray) -> pd.DataFrame:
    """Spectral MSE/SMAPE and energy error per band and output."""
    n = len(y_true)
    freqs = np.fft.rfftfreq(n, d=1.0)
    period = np.divide(
        1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0
    )
    amp_true, amp_pred = amplitude_spectrum(y_true), amplitude_spectrum(y_pred)
    rows = []
    for name, lo, hi in BANDS:
        mask = (period >= lo) & (period < hi)
        if not mask.any():
            continue
        at, ap = amp_true[mask], amp_pred[mask]
        for j, out in enumerate(OUTPUT_COLS):
            e_true = (at[:, j] ** 2).sum()
            e_pred = (ap[:, j] ** 2).sum()
            rows.append(
                {
                    "banda": name,
                    "saida": out,
                    "mse_espectral": ((ap[:, j] - at[:, j]) ** 2).mean(),
                    "smape_espectral_%": 100
                    * np.mean(
                        2
                        * np.abs(ap[:, j] - at[:, j])
                        / (np.abs(at[:, j]) + np.abs(ap[:, j]) + 1e-12)
                    ),
                    "energia_rel_%": 100 * (e_pred - e_true) / e_true,
                }
            )
    return pd.DataFrame(rows).set_index(["banda", "saida"])


def chained_forecast(
    model: nn.Module,
    raw: npt.NDArray,
    preprocessor: TimeSeriesPreprocessor,
    start: int,
    stop: int,
    device: torch.device | str = "cpu",
    seq_first: bool = False,
    input_window: int = INPUT_WINDOW,
    output_window: int = OUTPUT_WINDOW,
) -> npt.NDArray:
    """Chained `output_window`-step forecast in the original scale.

    Args:
        model: Forecaster with the shared contract (or sequence-first
            when `seq_first`), already in eval mode.
        raw: Raw `(n_rows, n_nodes)` series covering `start -
            input_window` to `stop`.
        preprocessor: Fitted on the training rows only.
        start: Row of the first forecast step.
        stop: One past the last row that can be forecast.
        device: Where to run the model.
        seq_first: Model takes `(window, batch, nodes)`.
        input_window: Lag window the model was trained with.
        output_window: Forecast horizon the model was trained with.

    Returns:
        `(n_steps, n_targets)` predictions for rows
        `start .. start + n_steps`, reconstructed with
        `inverse_transform(..., reference=raw)`.
    """
    series = preprocessor.transform_series(raw)
    anchors = np.arange(start, stop - output_window + 1, output_window)
    outs = []
    # One window per call: StemGNN aggregates its adjacency over the
    # batch, so batching windows would change its forecasts.
    for t0 in anchors:
        window = torch.as_tensor(
            series[t0 - input_window : t0], dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            if seq_first:
                out = model(window.unsqueeze(1)).squeeze(1)
            else:
                out = model(window.unsqueeze(0)).squeeze(0)
        outs.append(out.cpu().numpy())
    preds = preprocessor.inverse_transform(
        np.stack(outs), anchors, feats=TARGET_IDX, reference=raw
    )
    return preds.reshape(-1, len(TARGET_IDX))
