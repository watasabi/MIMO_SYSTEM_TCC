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
    INPUT_NAMES,
    OUTPUT_LABELS,
    OUTPUT_NAMES,
    WIDTH,
    apply_style,
    panel_letter,
    save,
)
from matplotlib.ticker import FuncFormatter, MaxNLocator
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
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def coherence_figure(data: pd.DataFrame) -> None:
    """Coerência de cada entrada (linhas) com cada saída (colunas)."""
    fig, axes = plt.subplots(
        len(INPUT_COLS), 3, figsize=(WIDTH, 6.2), sharex=True, sharey=True
    )
    for i, col in enumerate(INPUT_COLS):
        for k, out in enumerate(OUTPUT_COLS):
            f, cxy = signal.coherence(
                data[col].to_numpy(), data[out].to_numpy(), fs=1.0, nperseg=256
            )
            axes[i, k].plot(f, cxy, color=COLORS[1], lw=0.8)
            axes[i, k].set_ylim(0, 1)
            axes[i, k].set_yticks([0, 0.5, 1])
            axes[i, k].set_xticks([0, 0.25, 0.5])
            axes[i, k].xaxis.set_major_formatter(COMMA)
            axes[i, k].yaxis.set_major_formatter(COMMA)
        axes[i, 0].set_ylabel(
            INPUT_NAMES[i].replace("Vazão de ", "Vazão de\n")
        )
    for k, name in enumerate(OUTPUT_NAMES):
        axes[0, k].set_title(name)
        axes[-1, k].set_xlabel("Frequência (ciclos/h)")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    save(fig, "eda_coerencia.pdf")


def arx_poles(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Polos de um ARX de ordem 2 ajustado por mínimos quadrados."""
    k = np.arange(ARX_ORDER, len(y))
    x = np.column_stack([y[k - 1], y[k - 2], u[k], u[k - 1]])
    coef = np.linalg.lstsq(x, y[k], rcond=None)[0]
    return np.roots([1.0, -coef[0], -coef[1]])


def poles_figure(data: pd.DataFrame) -> None:
    """Polos no plano z, por par entrada (linha) e saída (coluna)."""
    theta = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(
        len(INPUT_COLS), 3, figsize=(WIDTH, 7.6), sharex=True, sharey=True
    )
    for i, col in enumerate(INPUT_COLS):
        for k, out in enumerate(OUTPUT_COLS):
            ax = axes[i, k]
            poles = arx_poles(data[col].to_numpy(), data[out].to_numpy())
            ax.plot(
                np.cos(theta), np.sin(theta), color="0.45", lw=0.7, ls="--"
            )
            ax.axhline(0, color="0.8", lw=0.5)
            ax.axvline(0, color="0.8", lw=0.5)
            ax.scatter(poles.real, poles.imag, color=COLORS[2], s=20, zorder=3)
            for p in poles:
                ax.annotate(
                    f"{p.real:.2f}".replace(".", ","),
                    (p.real, p.imag),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                )
            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(-1.25, 1.25)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.set_aspect("equal")
        axes[i, 0].set_ylabel(
            INPUT_NAMES[i].replace("Vazão de ", "Vazão de\n")
        )
    for k, name in enumerate(OUTPUT_NAMES):
        axes[0, k].set_title(name)
        axes[-1, k].set_xlabel("Parte real")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    save(fig, "eda_polos_arx.pdf")


def stl_figure(data: pd.DataFrame) -> None:
    """Um grupo por saída, com os componentes na largura toda da página."""
    rows = ["Medido", "Tendência", "Sazonal", "Resíduo"]
    fig = plt.figure(figsize=(WIDTH, 8.3))
    outer = fig.add_gridspec(3, 1, hspace=0.2, left=0.17, right=0.99)
    shared = None
    for k, out in enumerate(OUTPUT_COLS):
        res = STL(data[out], period=24, robust=True).fit()
        parts = (res.observed, res.trend, res.seasonal, res.resid)
        inner = outer[k].subgridspec(4, 1, hspace=0.16)
        for r, (label, part) in enumerate(zip(rows, parts, strict=True)):
            ax = fig.add_subplot(inner[r], sharex=shared)
            shared = shared or ax
            ax.plot(part.index, part.to_numpy(), color=COLORS[0], lw=0.45)
            ax.set_ylabel(label, rotation=0, ha="right", va="center")
            ax.yaxis.set_label_coords(-0.075, 0.5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
            )
            ax.tick_params(labelbottom=k == 2 and r == 3)  # noqa: PLR2004
            if r == 0:
                ax.set_title(
                    f"({chr(97 + k)}) {OUTPUT_LABELS[k]}", loc="left", pad=3
                )
    shared.xaxis.set_major_locator(mdates.MonthLocator())
    shared.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    save(fig, "eda_stl_saidas.pdf")


def variance_figure() -> None:
    train = load_splits(FINAL_TRAIN)["train"]
    fig, axes = plt.subplots(
        3, 1, figsize=(WIDTH, 3.6), sharex=True, sharey=True
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
    coherence_figure(data)
    poles_figure(data)
    stl_figure(data)
    variance_figure()


if __name__ == "__main__":
    main()
