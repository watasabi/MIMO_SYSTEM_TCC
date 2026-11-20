"""Figura dos folds da validação cruzada sobre as séries, para o artigo.

No topo, as faixas de treino, validação e teste de cada fold; embaixo,
as séries medidas com o bloco de teste de cada fold colorido e as
falhas de sensor sombreadas, no mesmo eixo de horas.

Uso (da raiz do repo):
    .venv/bin/python notebooks/training/scripts/figure_cv_folds.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from run_cv_benchmark import BAD, FOLDS, RAW

FIGURES = Path(__file__).resolve().parents[3] / "outputs" / "figures"
WIDTH_IN = 6.3
TRAIN, VAL = "#bdbdbd", "#f0a35e"
TEST = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#937860"]
SERIES = (
    (3, "Entrada (l/s)"),
    (4, "Distribuição (l/s)"),
    (5, "Nível (m)"),
    (6, "Pressão (mca)"),
)

mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
    }
)


def failure_spans() -> list[tuple[int, int]]:
    rows = np.flatnonzero(BAD)
    starts = rows[np.r_[True, np.diff(rows) > 1]]
    ends = rows[np.r_[np.diff(rows) > 1, True]]
    return [(int(a), int(b) + 1) for a, b in zip(starts, ends, strict=True)]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    spans = failure_spans()
    fig, axes = plt.subplots(
        len(SERIES) + 1,
        1,
        figsize=(WIDTH_IN, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1, 1, 1, 1]},
    )
    bars = axes[0]
    for k, f in enumerate(FOLDS):
        y = len(FOLDS) - 1 - k
        bars.barh(y, f.train_end, color=TRAIN, height=0.7)
        bars.barh(
            y, f.val_end - f.train_end, left=f.train_end, color=VAL, height=0.7
        )
        bars.barh(
            y,
            f.test_end - f.val_end,
            left=f.val_end,
            color=TEST[k],
            height=0.7,
        )
    bars.set_yticks(range(len(FOLDS)))
    bars.set_yticklabels([f"fold {k}" for k in range(len(FOLDS), 0, -1)])
    bars.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        bars.spines[side].set_visible(False)
    for ax, (col, label) in zip(axes[1:], SERIES, strict=True):
        for k, f in enumerate(FOLDS):
            ax.axvspan(f.val_end, f.test_end, color=TEST[k], alpha=0.16, lw=0)
        for a, b in spans:
            ax.axvspan(a, b, color="black", alpha=0.35, lw=0)
        ax.plot(np.arange(len(RAW)), RAW[:, col], color="0.15", lw=0.55)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    for a, b in spans:
        bars.axvspan(a, b, color="black", alpha=0.35, lw=0)
    axes[-1].set_xlabel("Hora da série")
    axes[-1].set_xlim(0, len(RAW))
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(FIGURES / "cv_folds_series.pdf")
    plt.close(fig)
    print("figura em", FIGURES / "cv_folds_series.pdf")


if __name__ == "__main__":
    main()
