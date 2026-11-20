"""Figura das sete séries do reservatório, com as falhas de sensor.

Gera `dados_series.pdf` para a subseção Materiais do artigo: as quatro
entradas e as três saídas no tempo, na série de 2.208 horas usada na
validação cruzada, com as linhas de falha de sensor sombreadas (as 22
marcadas pelas regras físicas, com margem, e o trecho 1.445--1.488). Não
marca nenhuma divisão de dados: os folds estão na figura própria
(`figure_cv_folds.py`).

Uso (da raiz do repo):
    .venv/bin/python notebooks/training/scripts/figure_dataset_series.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from run_cv_benchmark import BAD, RAW

OUT = Path.home() / "code/TCC_MIMO_SYS_overleaf/imagens/dados_series.pdf"
PANELS = (
    (0, "Bomba 1\n$u_1$ (Hz)"),
    (1, "Bomba 2\n$u_2$ (Hz)"),
    (2, "Bomba 3\n$u_3$ (Hz)"),
    (3, "Entrada\n$u_4$ (l/s)"),
    (4, "Distribuição\n$y_1$ (l/s)"),
    (5, "Nível\n$y_2$ (m)"),
    (6, "Pressão\n$y_3$ (mca)"),
)

mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 9,
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
    spans = failure_spans()
    hours = np.arange(len(RAW))
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.3, 7.6), sharex=True)
    for ax, (col, label) in zip(axes, PANELS, strict=True):
        for a, b in spans:
            ax.axvspan(a, b, color="#c44e52", alpha=0.25, lw=0)
        ax.plot(hours, RAW[:, col], lw=0.5, color="0.15")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Hora da série (27/07/2014 a 27/10/2014)")
    axes[-1].set_xlim(0, len(RAW))
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.3)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
