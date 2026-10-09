"""Avaliação e plots comuns aos experimentos `collection_models`.

Usado por `05_collection_models_v1.py` (split 70/15/15, pesos já
treinados) e `06_collection_models_kfold.py` (TimeSeriesSplit, treina
um modelo por fold). Mantém num só lugar o contrato de avaliação —
forecast encadeado de `OUTPUT_WINDOW` passos na escala original — e as
figuras registradas no MLflow.

As figuras são logadas como **PNG** via `mlflow.log_figure`, porque a UI
do MLflow mostra PNG inline mas serve HTML como texto cru/download — o
que torna o artefato inútil de olhar pela interface.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import nn

INPUT_COLS = ["freq_b1", "freq_b2", "freq_b3", "vazao_entrada"]
OUTPUT_COLS = ["vazao_distribuicao", "nivel_res", "pressao"]
FEATURE_COLS = INPUT_COLS + OUTPUT_COLS
N_NODES = len(FEATURE_COLS)
TARGET_IDX = [FEATURE_COLS.index(c) for c in OUTPUT_COLS]
N_TARGETS = len(TARGET_IDX)
LABELS = ["Vazão de distribuição (l/s)", "Nível (m)", "Pressão (mca)"]

INPUT_WINDOW = 24
OUTPUT_WINDOW = 12
SEED = 42

# Bandas ancoradas na EDA (`docs/5_eda-series-temporais.md` §7): pico
# diário em ~24h, segundo harmônico em ~12h.
# Nomes restritos a alfanumérico/underscore/traço: o MLflow rejeita
# `>` e `+` em nome de métrica.
BANDS = [
    ("tendencia_acima_32h", 32.0, np.inf),
    ("diario_16-32h", 16.0, 32.0),
    ("harmonico_8-16h", 8.0, 16.0),
    ("curto_4-8h", 4.0, 8.0),
    ("ruido_2-4h", 2.0, 4.0),
]

PLOT_W, PLOT_H = 1100, 760

# O kaleido roda Chrome headless para gerar PNG e, em lote longo
# (k-fold são dezenas de figuras), estoura o timeout default. Uma folga
# maior troca velocidade por estabilidade.
try:
    import plotly.io as pio

    pio.defaults.timeout = 120
except Exception:  # noqa: BLE001 - só um ajuste de robustez
    pass


# ------------------------------------------------------------ métricas


def metrics_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """MSE, MAE, R² e SMAPE por saída."""
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


def amplitude_spectrum(x: np.ndarray) -> np.ndarray:
    """Espectro de amplitude unilateral, `(n_freqs, n_saidas)`."""
    return np.abs(np.fft.rfft(x, axis=0)) * 2.0 / len(x)


def band_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """MSE/SMAPE espectral e erro de energia por banda e por saída."""
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


# --------------------------------------------------------------- plots


def plot_series(y_true, y_pred, name, idx, table):
    """Medido vs. previsto nas 3 saídas, com R²/SMAPE no subtítulo."""
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


def plot_spectrum(y_true, y_pred, name, bands):
    """Espectro medido vs. previsto, pior banda destacada no subtítulo."""
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


def plot_lag_attention(weights, name):
    """Heatmap passo de saída × lag de entrada (t-1 = mais recente)."""
    out_window, in_window = weights.shape
    entropy = float(-(weights * np.log(weights + 1e-12)).sum(1).mean())
    rel = entropy / float(np.log(in_window))
    fig = go.Figure(
        data=go.Heatmap(
            z=weights,
            x=[f"t-{in_window - k}" for k in range(in_window)],
            y=[f"t+{t + 1}" for t in range(out_window)],
            colorscale="Viridis",
            colorbar=dict(title="Peso"),
        )
    )
    fig.update_layout(
        title_text=(
            f"{name} — attention sobre os lags "
            f"(entropia relativa {rel:.3f}; 1,0 = uniforme)"
        ),
        xaxis_title="Lag de entrada",
        yaxis_title="Passo de saída previsto",
        template="plotly_white",
        height=520,
        width=PLOT_W,
    )
    return fig


def plot_node_attention(matrix, name):
    """Heatmap nó → nó (atenção do StemGNN ou edge_strength do MPNN)."""
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=FEATURE_COLS,
            y=FEATURE_COLS,
            colorscale="Magma",
            colorbar=dict(title="Peso"),
        )
    )
    fig.update_layout(
        title_text=f"{name} — peso entre nós (origem → destino)",
        xaxis_title="Nó de destino",
        yaxis_title="Nó de origem",
        template="plotly_white",
        height=560,
        width=760,
    )
    return fig


def plot_r2_heatmap(r2: pd.DataFrame, title: str):
    """Resumo R² modelo × saída, para o run pai."""
    fig = go.Figure(
        data=go.Heatmap(
            z=r2.to_numpy(),
            x=list(r2.columns),
            y=list(r2.index),
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=r2.round(3).to_numpy(),
            texttemplate="%{text}",
            colorbar=dict(title="R²"),
        )
    )
    fig.update_layout(
        title_text=title,
        template="plotly_white",
        height=120 + 42 * len(r2),
        width=PLOT_W,
    )
    return fig


def plot_band_heatmap(freq: pd.DataFrame, output: str, title: str):
    """Erro de energia por banda × modelo, para uma saída."""
    sub = freq.xs(output, level="saida")["energia_rel_%"].unstack("banda")
    order = [b[0] for b in BANDS if b[0] in sub.columns]
    sub = sub[order]
    limit = float(np.abs(sub.to_numpy()).max())
    fig = go.Figure(
        data=go.Heatmap(
            z=sub.to_numpy(),
            x=list(sub.columns),
            y=list(sub.index),
            colorscale="RdBu",
            zmid=0,
            zmin=-limit,
            zmax=limit,
            text=sub.round(0).to_numpy(),
            texttemplate="%{text}%",
            colorbar=dict(title="energia %"),
        )
    )
    fig.update_layout(
        title_text=title,
        xaxis_title="banda",
        template="plotly_white",
        height=120 + 42 * len(sub),
        width=PLOT_W,
    )
    return fig


def log_figure(fig, filename: str, figures_dir: Path) -> None:
    """Loga o figure como PNG (visível na UI) e salva uma cópia local.

    `mlflow.log_figure` com `.png` é o que a UI consegue pré-visualizar;
    o HTML interativo vai junto, em `interativo/`, para abrir no
    navegador quando precisar de zoom/hover.

    O render de PNG passa pelo kaleido, que levanta `TimeoutError`
    esporádico em lote longo. Como o HTML não depende dele, uma falha de
    PNG não pode derrubar a corrida inteira — só registra o aviso.
    """
    try:
        mlflow.log_figure(fig, f"{filename}.png")
    except Exception as exc:  # noqa: BLE001 - kaleido é instável em lote
        print(f"    [aviso] PNG de {filename} falhou ({type(exc).__name__})")
        mlflow.set_tag(f"png_falhou__{filename}", type(exc).__name__)
    html = figures_dir / f"{filename}.html"
    html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html, include_plotlyjs="cdn")
    mlflow.log_artifact(str(html), artifact_path="interativo")


def log_metrics_table(table: pd.DataFrame, bands: pd.DataFrame) -> None:
    """Registra métricas por saída e por banda no run ativo."""
    for out in OUTPUT_COLS:
        for metric, value in table.loc[out].items():
            key = metric.replace("_%", "_pct")
            mlflow.log_metric(f"{out}__{key}", float(value))
    mlflow.log_metric("r2_medio", float(table["r2"].mean()))
    mlflow.log_metric("smape_medio_pct", float(table["smape_%"].mean()))
    mlflow.log_metric("mse_medio", float(table["mse"].mean()))
    for (banda, out), row in bands.iterrows():
        mlflow.log_metric(
            f"freq__{out}__{banda}__energia_rel_pct",
            float(row["energia_rel_%"]),
        )
        mlflow.log_metric(
            f"freq__{out}__{banda}__smape_espectral_pct",
            float(row["smape_espectral_%"]),
        )


# ---------------------------------------------------------- attention


def lag_attention(model, window: torch.Tensor) -> np.ndarray | None:
    """Mapa `(output_window, input_window)`, ou `None` se não houver.

    Só os decoders que atendem sobre os passos do encoder têm esse eixo.
    `StemGNN`/`MPNN`/`SymbolicGraphNetwork` expõem peso entre *nós* —
    ver `node_attention`.
    """
    if hasattr(model, "last_attention_map"):
        with torch.no_grad():
            model(window)
        return model.last_attention_map[0].cpu().numpy()
    if type(model).__name__ == "Seq2SeqLatentGNN":
        encoded, decoded = [], []
        h1 = model.encoder.register_forward_hook(
            lambda m, i, o: encoded.append(o[0].detach())
        )
        h2 = model.decoder.register_forward_hook(
            lambda m, i, o: decoded.append(o[0].detach())
        )
        try:
            with torch.no_grad():
                model(window)
        finally:
            h1.remove()
            h2.remove()
        enc = encoded[0]
        weights = torch.cat(
            [
                torch.softmax(torch.bmm(d, enc.transpose(1, 2)), dim=-1)
                for d in decoded
            ],
            dim=1,
        )
        return weights[0].cpu().numpy()
    return None


def node_attention(model, window: torch.Tensor) -> np.ndarray | None:
    """Peso entre nós, `(n_nodes, n_nodes)`, ou `None`."""
    with torch.no_grad():
        if type(model).__name__ == "StemGNN":
            model(window)
            return model.last_adjacency.cpu().numpy()
        if hasattr(model, "edge_strength"):
            return model.edge_strength(window).cpu().numpy()
    return None


def log_attention(model, window: torch.Tensor, name, figures_dir) -> None:
    """Loga os mapas de attention disponíveis para o modelo."""
    weights = lag_attention(model, window)
    if weights is None:
        mlflow.set_tag("attention_lags", "nao")
    else:
        log_figure(
            plot_lag_attention(weights, name), "attention_lags", figures_dir
        )
        entropy = float(-(weights * np.log(weights + 1e-12)).sum(1).mean())
        mlflow.log_metric("attention_entropia", entropy)
        mlflow.log_metric(
            "attention_entropia_rel",
            entropy / float(np.log(weights.shape[1])),
        )
        mlflow.log_metric(
            "attention_lag_recente_pct", float(weights[:, -1].mean() * 100)
        )
        mlflow.set_tag("attention_lags", "sim")

    nodes = node_attention(model, window)
    if nodes is not None:
        log_figure(
            plot_node_attention(nodes, name), "attention_nos", figures_dir
        )
        mlflow.set_tag("attention_nos", "sim")


# ------------------------------------------------------------ forecast


def chained_forecast(
    model: nn.Module,
    series_scaled: np.ndarray,
    scaler,
    start: int,
    stop: int,
    device,
    seq_first: bool = False,
) -> np.ndarray:
    """Forecast encadeado de `OUTPUT_WINDOW` passos, escala original."""
    preds = []
    for t0 in range(start, stop - OUTPUT_WINDOW + 1, OUTPUT_WINDOW):
        window = torch.FloatTensor(series_scaled[t0 - INPUT_WINDOW : t0])
        window = (
            window.unsqueeze(1) if seq_first else window.unsqueeze(0)
        ).to(device)
        with torch.no_grad():
            out = model(window)
        out = out.squeeze(1) if seq_first else out.squeeze(0)
        preds.append(out.cpu().numpy())
    preds = np.concatenate(preds)
    return (
        preds * scaler.data_range_[TARGET_IDX] + scaler.data_min_[TARGET_IDX]
    )
