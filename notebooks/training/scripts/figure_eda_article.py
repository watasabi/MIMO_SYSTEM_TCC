"""Figuras da análise exploratória no formato do artigo.

Reproduz os cálculos de `notebooks/eda/03_eda_series_temporais.ipynb` e
`07_decomposicao_saidas.ipynb`, divididos em figuras de página inteira.

Uso (da raiz do repo):
    uv run --extra ml python notebooks/training/scripts/figure_eda_article.py
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_style import (
    COLORS,
    LINES,
    MARKERS,
    OUTPUT_LABELS,
    OUTPUT_NAMES,
    WIDTH,
    apply_style,
    panel_letter,
    save,
)
from scipy import signal
from statsmodels.tsa.seasonal import STL

from mimo_sys.training.data import (
    DATA_PROCESSED,
    FEATURE_COLS,
    INPUT_COLS,
    OUTPUT_COLS,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.features import atrous_components

BANDS = ["2–4", "4–8", "8–16", "16–32", ">32"]
WARMUP = 32  # linhas em que o histórico da wavelet ainda é preenchido
FINAL_TRAIN = "mimo_cv5_f5"
ARX_ORDER = 2


def coherence_figure(data: pd.DataFrame, inputs: list[str], name: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.0), sharex=True)
    for k, (ax, out) in enumerate(zip(axes, OUTPUT_COLS, strict=True)):
        for i, col in enumerate(inputs):
            f, cxy = signal.coherence(
                data[col].to_numpy(), data[out].to_numpy(), fs=1.0, nperseg=256
            )
            ax.plot(f, cxy, color=COLORS[i], ls=LINES[i])
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"Coerência\n{OUTPUT_NAMES[k].lower()}")
        panel_letter(ax, k)
    axes[-1].set_xlabel("Frequência (ciclos por hora)")
    fig.tight_layout()
    save(fig, name)


def arx_poles(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Polos de um ARX de ordem 2 ajustado por mínimos quadrados."""
    k = np.arange(ARX_ORDER, len(y))
    x = np.column_stack([y[k - 1], y[k - 2], u[k], u[k - 1]])
    coef = np.linalg.lstsq(x, y[k], rcond=None)[0]
    return np.roots([1.0, -coef[0], -coef[1]])


def poles_figure(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    offsets = np.linspace(-0.24, 0.24, len(INPUT_COLS))
    for i, col in enumerate(INPUT_COLS):
        for k, out in enumerate(OUTPUT_COLS):
            poles = arx_poles(data[col].to_numpy(), data[out].to_numpy())
            ax.scatter(
                k + offsets[i],
                np.abs(poles).max(),
                marker=MARKERS[i],
                color=COLORS[i],
                s=26,
                zorder=3,
            )
    ax.axhline(1, color="0.45", lw=0.8, ls="--")
    ax.set_xticks(range(3), OUTPUT_NAMES)
    ax.set_ylabel("Módulo do polo dominante")
    fig.tight_layout()
    save(fig, "eda_polos_arx.pdf")


def stl_figure(data: pd.DataFrame, k: int, name: str) -> None:
    res = STL(data[OUTPUT_COLS[k]], period=24, robust=True).fit()
    parts = [
        ("Medido", res.observed),
        ("Tendência", res.trend),
        ("Sazonal", res.seasonal),
        ("Resíduo", res.resid),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(WIDTH, 4.8), sharex=True)
    for ax, (label, part) in zip(axes, parts, strict=True):
        ax.plot(part.index, part.to_numpy(), color=COLORS[0], lw=0.7)
        ax.set_ylabel(label)
    axes[0].set_ylabel(f"Medido\n{OUTPUT_LABELS[k].split(' ')[-1]}")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    fig.align_ylabels(axes)
    fig.tight_layout()
    save(fig, name)


def variance_figure() -> None:
    train = load_splits(FINAL_TRAIN)["train"]
    fig, axes = plt.subplots(
        3, 1, figsize=(WIDTH, 4.6), sharex=True, sharey=True
    )
    for k, (ax, col) in enumerate(zip(axes, TARGET_IDX, strict=True)):
        comps = atrous_components(train[:, col], 4)[WARMUP:]
        share = 100 * comps.var(axis=0) / train[WARMUP:, col].var()
        print(
            OUTPUT_COLS[k], dict(zip(BANDS, np.round(share, 1), strict=True))
        )
        ax.bar(BANDS, share, color=COLORS[1], width=0.62)
        ax.set_ylabel(f"Variância (%)\n{OUTPUT_NAMES[k].lower()}")
        panel_letter(ax, k)
    axes[-1].set_xlabel("Banda de período (horas)")
    fig.tight_layout()
    save(fig, "eda_variancia_bandas.pdf")


def main() -> None:
    apply_style()
    raw = pd.read_parquet(DATA_PROCESSED / "mimo_data.parquet")[FEATURE_COLS]
    data = raw.interpolate().ffill().bfill()
    coherence_figure(data, INPUT_COLS[:3], "eda_coerencia_bombas.pdf")
    coherence_figure(data, INPUT_COLS[3:], "eda_coerencia_entrada.pdf")
    poles_figure(data)
    for k, name in enumerate(("vazao", "nivel", "pressao")):
        stl_figure(data, k, f"eda_stl_{name}.pdf")
    variance_figure()


if __name__ == "__main__":
    main()
