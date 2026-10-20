"""Forecast evaluation in the original scale: metrics, bands, plots.

The evaluation contract is the one of the collection experiments:
chained forecast of `OUTPUT_WINDOW` steps over the test range,
reconstructed with the preprocessor's `inverse_transform`. Plots are
Plotly, logged as PNG (what the MLflow UI previews inline) plus an
interactive HTML copy — see `log_figure`.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import nn

from mimo_sys.preprocessors import TimeSeriesPreprocessor
from mimo_sys.training.data import (
    INPUT_WINDOW,
    OUTPUT_COLS,
    OUTPUT_WINDOW,
    TARGET_IDX,
    with_future_exog,
)

LABELS = ["Vazão de distribuição (l/s)", "Nível (m)", "Pressão (mca)"]
PLOT_W, PLOT_H = 1100, 760

# kaleido runs headless Chrome for PNG export and raises a sporadic
# TimeoutError in long batches; a larger timeout trades speed for
# stability.
try:
    import plotly.io as pio

    pio.defaults.timeout = 120
except Exception:  # noqa: BLE001 - best-effort robustness tweak
    pass

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
    future_exog: bool = False,
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
        future_exog: Model was trained with the known future inputs
            appended (`data.with_future_exog`); `raw` must then also
            cover `stop`'s last horizon.

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
        window = series[t0 - input_window : t0]
        if future_exog:
            window = with_future_exog(
                window[None], series[None, t0 : t0 + output_window]
            )[0]
        window = torch.as_tensor(window, dtype=torch.float32).to(device)
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


def plot_series(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    name: str,
    idx: npt.NDArray,
    table: pd.DataFrame,
) -> go.Figure:
    """Measured vs. predicted on the 3 outputs, R²/SMAPE in subtitles."""
    titles = [
        f"{LABELS[j]} — R² {table.loc[o, 'r2']:.3f}, "
        f"SMAPE {table.loc[o, 'smape_%']:.1f}%"
        for j, o in enumerate(OUTPUT_COLS)
    ]
    fig = make_subplots(rows=3, cols=1, subplot_titles=titles)
    for i in range(3):
        for label, arr, color in (
            ("Medido", y_true, "black"),
            (name, y_pred, "firebrick"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=arr[:, i],
                    name=label,
                    line=dict(color=color, width=1.2),
                    legendgroup=label,
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )
    fig.update_xaxes(title_text="amostra", row=3, col=1)
    fig.update_layout(
        title_text=f"{name} — medido vs. previsto",
        height=PLOT_H,
        width=PLOT_W,
        template="plotly_white",
    )
    return fig


def plot_spectrum(
    y_true: npt.NDArray, y_pred: npt.NDArray, name: str, bands: pd.DataFrame
) -> go.Figure:
    """Measured vs. predicted spectrum, worst band in the subtitle."""
    n = len(y_true)
    freqs = np.fft.rfftfreq(n, d=1.0)
    period = np.divide(
        1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0
    )
    amp_true, amp_pred = amplitude_spectrum(y_true), amplitude_spectrum(y_pred)
    titles = []
    for j, out in enumerate(OUTPUT_COLS):
        sub = bands.xs(out, level="saida")
        worst = sub["energia_rel_%"].abs().idxmax()
        titles.append(
            f"{LABELS[j]} — pior banda {worst}: "
            f"{sub.loc[worst, 'energia_rel_%']:+.0f}% energia, "
            f"SMAPE espectral {sub.loc[worst, 'smape_espectral_%']:.0f}%"
        )
    fig = make_subplots(rows=3, cols=1, subplot_titles=titles)
    for i in range(3):
        for label, amp, color in (
            ("Medido", amp_true, "black"),
            (name, amp_pred, "firebrick"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=period[1:],
                    y=amp[1:, i],
                    name=label,
                    line=dict(color=color, width=1.2),
                    legendgroup=label,
                    showlegend=(i == 0),
                ),
                row=i + 1,
                col=1,
            )
        for mark in (12.0, 24.0):
            fig.add_vline(
                x=mark,
                line=dict(color="gray", dash="dot", width=1),
                row=i + 1,
                col=1,
            )
    fig.update_xaxes(
        type="log", autorange="reversed", title_text="período (h)"
    )
    fig.update_yaxes(type="log", title_text="amplitude")
    fig.update_layout(
        title_text=f"{name} — espectro (pontilhado: 12h e 24h)",
        height=PLOT_H + 140,
        width=PLOT_W,
        template="plotly_white",
    )
    return fig


def log_figure(
    fig: go.Figure, filename: str, figures_dir: Path | None = None
) -> None:
    """Log `fig` as PNG (what the MLflow UI previews) plus HTML.

    The interactive HTML goes under `interativo/` in the active run's
    artifacts (and, if `figures_dir` is given, a local copy too), for
    zoom/hover when the PNG isn't enough. PNG export goes through
    kaleido, which is unstable in long batches — a failure there only
    warns, it never aborts the run.
    """
    try:
        mlflow.log_figure(fig, f"{filename}.png")
    except Exception as exc:  # noqa: BLE001 - kaleido is flaky in batches
        print(f"    [aviso] PNG de {filename} falhou ({type(exc).__name__})")
        mlflow.set_tag(f"png_falhou__{filename}", type(exc).__name__)
    if figures_dir is not None:
        html = figures_dir / f"{filename}.html"
        html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html, include_plotlyjs="cdn")
        mlflow.log_artifact(str(html), artifact_path="interativo")
    else:
        mlflow.log_figure(fig, f"interativo/{filename}.html")
