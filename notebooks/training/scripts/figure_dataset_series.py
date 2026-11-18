"""Figura das sete séries do reservatório.

Gera `dados_series.pdf` para a subseção Materiais do artigo. As linhas
verticais marcam o início dos blocos de teste da validação cruzada.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from figure_style import COLORS, WIDTH, apply_style, save

from mimo_sys.training.benchmarks import temporal_folds
from mimo_sys.training.data import FEATURE_COLS, load_splits

LABELS = {
    "freq_b1": "$u_1$ (Hz)",
    "freq_b2": "$u_2$ (Hz)",
    "freq_b3": "$u_3$ (Hz)",
    "vazao_entrada": "$u_4$ (l/s)",
    "vazao_distribuicao": "$y_1$ (l/s)",
    "nivel_res": "$y_2$ (m)",
    "pressao": "$y_3$ (mca)",
}

apply_style()
splits = load_splits()
data = np.concatenate([splits[name] for name in ("train", "val", "test")])
fig, axes = plt.subplots(len(LABELS), 1, figsize=(WIDTH, 6.6), sharex=True)
for ax, (col, label) in zip(axes, LABELS.items(), strict=True):
    ax.plot(data[:, FEATURE_COLS.index(col)], color=COLORS[0], lw=0.55)
    for fold in temporal_folds(len(data)):
        ax.axvline(fold.val_end, color=COLORS[2], lw=0.8, ls="--")
    ax.set_ylabel(label)
axes[-1].set_xlabel("Hora")
axes[-1].set_xlim(0, len(data))
fig.align_ylabels(axes)
fig.tight_layout()
save(fig, "dados_series.pdf")
