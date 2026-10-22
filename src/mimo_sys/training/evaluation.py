"""Forecast evaluation in the original scale: metrics, bands, plots.

The evaluation contract is the one of the collection experiments:
chained forecast of `OUTPUT_WINDOW` steps over the test range,
reconstructed with the preprocessor's `inverse_transform`. Plots are
Plotly, logged as PNG (what the MLflow UI previews inline) plus an
interactive HTML copy — see `log_figure`.
"""

from __future__ import annotations

import math
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
    INPUT_IDX,
    INPUT_WINDOW,
    N_TARGETS,
    OUTPUT_COLS,
    OUTPUT_WINDOW,
    TargetLayout,
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


def forecast_at_origin(  # noqa: PLR0913 - the production contract
    model: nn.Module,
    preprocessor: TimeSeriesPreprocessor,
    history: npt.NDArray,
    future_known: npt.NDArray | None,
    *,
    known_idx: list[int],
    input_window: int = INPUT_WINDOW,
    output_window: int = OUTPUT_WINDOW,
    layout: TargetLayout = TargetLayout(),
    device: torch.device | str = "cpu",
    seq_first: bool = False,
    return_aux: bool = False,
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
    """One forecast in production conditions, original scale.

    The signature *is* the leakage guard: the function only ever sees
    the rows observed before the origin and the horizon values of the
    columns known ahead of time (`known_idx`), never the future
    outputs. Unknown horizon cells are NaN going into the
    preprocessor, so a transform that mixes rows across time (EWT,
    a centred Savgol) puts NaN into the past window and raises here
    instead of leaking.

    Args:
        model: Forecaster in eval mode.
        preprocessor: Fitted on the training rows only.
        history: `(t0, n_cols)` raw rows before the origin, every
            column the model was trained on (derived ones included).
        future_known: `(output_window, len(known_idx))` raw horizon
            values of the known columns; `None` for models trained
            without `future_exog`.
        known_idx: Columns of `future_known`.
        input_window: Lag window the model was trained with.
        output_window: Forecast horizon the model was trained with.
        layout: How the model's outputs add up to the 3 outputs.
        device: Where to run the model.
        seq_first: Model takes `(window, batch, nodes)`.
        return_aux: Also return the auxiliary columns' forecast
            (`layout.aux_cols`, original scale).

    Returns:
        `(output_window, N_TARGETS)` forecast of rows `t0 .. t0 + H`
        (and `(output_window, len(aux_cols))` with `return_aux`).
    """
    t0, n_cols = history.shape
    block = history
    if future_known is not None:
        future = np.full((output_window, n_cols), np.nan)
        future[:, known_idx] = future_known
        block = np.concatenate([history, future])
    leak = (
        "the preprocessor mixes horizon rows into the past window "
        "(non-causal transform): not usable in production"
    )
    try:
        series = preprocessor.transform_series(block)
    except ValueError as err:  # e.g. the EWT's FFT refuses the NaN rows
        raise ValueError(leak) from err
    window = series[t0 - input_window : t0]
    if not np.isfinite(window).all():
        raise ValueError(leak)
    if future_known is not None:
        horizon = series[t0 : t0 + output_window]
        if not np.isfinite(horizon[:, known_idx]).all():
            raise ValueError("known horizon columns are not finite")
        window = with_future_exog(
            window[None], np.nan_to_num(horizon)[None], known_idx
        )[0]
    x = torch.as_tensor(window, dtype=torch.float32).to(device)
    with torch.no_grad():
        if seq_first:
            out = model(x.unsqueeze(1)).squeeze(1)
        else:
            out = model(x.unsqueeze(0)).squeeze(0)
    parts = preprocessor.inverse_transform(
        out.float().cpu().numpy()[None],
        np.array([t0]),
        feats=list(layout.model_cols),
        reference=history,
    )[0]
    parts, aux = (
        parts[:, : len(layout.pred_cols)],
        parts[:, len(layout.pred_cols) :],
    )
    pred = np.zeros((output_window, N_TARGETS))
    for j, i in enumerate(layout.pred_output):
        pred[:, i] += parts[:, j]
    for col, i in zip(layout.known_cols, layout.known_output, strict=True):
        pred[:, i] += future_known[:, list(known_idx).index(col)]
    return (pred, aux) if return_aux else pred


def forecast_origins(  # noqa: PLR0913 - mirrors forecast_at_origin
    model: nn.Module,
    raw: npt.NDArray,
    preprocessor: TimeSeriesPreprocessor,
    origins: npt.NDArray,
    *,
    output_window: int = OUTPUT_WINDOW,
    future_exog: bool = False,
    future_known_idx: list[int] | None = None,
    **kwargs: object,
) -> npt.NDArray:
    """`forecast_at_origin` at every origin, slicing `raw` honestly.

    Returns:
        `(len(origins), output_window, N_TARGETS)`, original scale.
    """
    known_idx = INPUT_IDX if future_known_idx is None else future_known_idx
    # One window per call: StemGNN aggregates its adjacency over the
    # batch, so batching windows would change its forecasts.
    outs = [
        forecast_at_origin(
            model,
            preprocessor,
            raw[:t0],
            raw[t0 : t0 + output_window][:, known_idx]
            if future_exog
            else None,
            known_idx=list(known_idx),
            output_window=output_window,
            **kwargs,
        )
        for t0 in origins
    ]
    if kwargs.get("return_aux"):
        return tuple(np.stack(part) for part in zip(*outs, strict=True))
    return np.stack(outs)


def chained_forecast(  # noqa: PLR0913
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
    future_known_idx: list[int] | None = None,
    layout: TargetLayout = TargetLayout(),
) -> npt.NDArray:
    """Chained `output_window`-step forecast in the original scale.

    One `forecast_at_origin` every `output_window` rows, so each
    forecast only uses the rows before its own origin.

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
        future_known_idx: Columns known in the future (default: all
            inputs, see `data.with_future_exog`).
        layout: How the model's outputs add up to the 3 outputs.

    Returns:
        `(n_steps, n_targets)` predictions for rows
        `start .. start + n_steps`.
    """
    origins = np.arange(start, stop - output_window + 1, output_window)
    preds = forecast_origins(
        model,
        raw,
        preprocessor,
        origins,
        output_window=output_window,
        future_exog=future_exog,
        future_known_idx=future_known_idx,
        input_window=input_window,
        layout=layout,
        device=device,
        seq_first=seq_first,
    )
    return preds.reshape(-1, N_TARGETS)


def r2_per_output(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    """R² of each output, pooling every leading axis (`(..., 3)`)."""
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    sse = ((yt - yp) ** 2).sum(axis=0)
    sst = ((yt - yt.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - sse / sst


def paired_block_bootstrap(  # noqa: PLR0913
    y_true: npt.NDArray,
    pred_a: npt.NDArray,
    pred_b: npt.NDArray,
    block: int = 24,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, npt.NDArray | float]:
    """CI of `R²(b) - R²(a)` on the same origins, moving-block bootstrap.

    Consecutive origins share most of their horizon, so origins are
    resampled in blocks of `block` to keep that dependence. With
    several seeds (a leading axis on the predictions, seed `s` of `a`
    paired with seed `s` of `b`) every replicate resamples the same
    origins for all seeds and averages the seed deltas.

    Args:
        y_true: `(n_origins, horizon, n_outputs)`.
        pred_a, pred_b: Same shape, or `(n_seeds, ...)` of it.

    Returns:
        `delta` (per output), `delta_mean` (of the output mean) and
        their 2.5/97.5% percentiles (`ci_low`/`ci_high`,
        `ci_low_mean`/`ci_high_mean`).
    """
    if pred_a.ndim == y_true.ndim:
        pred_a, pred_b = pred_a[None], pred_b[None]

    def delta_on(idx: npt.NDArray) -> npt.NDArray:
        return np.mean(
            [
                r2_per_output(y_true[idx], b[idx])
                - r2_per_output(y_true[idx], a[idx])
                for a, b in zip(pred_a, pred_b, strict=True)
            ],
            axis=0,
        )

    rng = np.random.default_rng(seed)
    n = len(y_true)
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    deltas = np.empty((n_boot, y_true.shape[-1]))
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        deltas[b] = delta_on(
            (starts[:, None] + np.arange(block)).reshape(-1)[:n]
        )
    delta = delta_on(np.arange(n))
    mean = deltas.mean(axis=1)
    return {
        "delta": delta,
        "ci_low": np.percentile(deltas, 2.5, axis=0),
        "ci_high": np.percentile(deltas, 97.5, axis=0),
        "delta_mean": float(delta.mean()),
        "ci_low_mean": float(np.percentile(mean, 2.5)),
        "ci_high_mean": float(np.percentile(mean, 97.5)),
    }


def diebold_mariano(
    y_true: npt.NDArray,
    pred_a: npt.NDArray,
    pred_b: npt.NDArray,
    lag: int = OUTPUT_WINDOW - 1,
) -> tuple[float, float]:
    """Diebold-Mariano test, `a` vs `b`, Newey-West (Bartlett) variance.

    The per-origin loss is the squared error averaged over horizon and
    outputs, each output divided by its variance (the R² scale), so a
    positive statistic means `b` has the smaller loss.

    Args:
        y_true: `(n_origins, horizon, n_outputs)`.
        pred_a, pred_b: Same shape, or `(n_seeds, ...)` of it (the loss
            is then averaged over seeds before the test).
        lag: HAC truncation; `horizon - 1` for overlapping origins.

    Returns:
        `(statistic, two-sided p-value)`.
    """
    if pred_a.ndim == y_true.ndim:
        pred_a, pred_b = pred_a[None], pred_b[None]
    var = y_true.reshape(-1, y_true.shape[-1]).var(axis=0)
    loss_a = (((y_true - pred_a) ** 2) / var).mean(axis=(0, 2, 3))
    loss_b = (((y_true - pred_b) ** 2) / var).mean(axis=(0, 2, 3))
    d = loss_a - loss_b
    n = len(d)
    dc = d - d.mean()
    hac = dc @ dc / n
    for k in range(1, lag + 1):
        hac += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    if hac <= 0:  # identical forecasts: no difference to test
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(hac / n)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


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
