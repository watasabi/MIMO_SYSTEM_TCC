"""Figuras de conceito: GNN, GNN em séries temporais e o grafo do sistema.

1. `gnn_conceito`: (a) passagem de mensagens num grafo; (b) GNN
   espaço-temporal, cada nó carrega uma janela de série; (c) a aresta do
   GOTAS, em que a mensagem é um filtro `H_ij(f)`.
2. `gnn_reservatorio`: as 7 séries do treino como nós (trecho real de
   7 dias) e as arestas para as saídas, com espessura pela causalidade
   de Granger (`report_graph_dependency.py`).

Uso (da raiz do repo):
    uv run python notebooks/training/scripts/figure_gnn_concept.py
"""

from __future__ import annotations

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mimo_sys.training.data import (
    INPUT_COLS,
    OUTPUT_COLS,
    PROJECT_ROOT,
    load_splits,
)

FIG = PROJECT_ROOT / "docs" / "figures"
OVERLEAF = PROJECT_ROOT / "overleaf" / "imagens"
DEP = PROJECT_ROOT / "outputs" / "graph_dependency.json"
LABEL = {
    "freq_b1": "bomba 1",
    "freq_b2": "bomba 2",
    "freq_b3": "bomba 3",
    "vazao_entrada": "vazão de\nentrada",
    "vazao_distribuicao": "vazão de\ndistribuição",
    "nivel_res": "nível",
    "pressao": "pressão",
}
IN_C, OUT_C = "#f8cecc", "#e1d5e7"
STRONG_P = 1e-3


def arrow(ax, p, q, **kw) -> None:  # noqa: ANN001
    style = dict(
        arrowstyle="-|>",
        mutation_scale=12,
        lw=1.2,
        color="#333",
        shrinkA=16,
        shrinkB=16,
    )
    style.update(kw)
    ax.add_patch(FancyArrowPatch(p, q, **style))


def sparkline(fig, ax, xy, series, w=0.09, h=0.05, color="k"):  # noqa: ANN001
    """Tiny time-series plot centred at data coordinate `xy` of `ax`."""
    disp = ax.transData.transform(xy)
    fx, fy = fig.transFigure.inverted().transform(disp)
    sub = fig.add_axes([fx - w / 2, fy - h / 2, w, h])
    sub.plot(series, color=color, lw=0.8)
    sub.set_xticks([])
    sub.set_yticks([])
    for s in sub.spines.values():
        s.set_color("#999")
        s.set_linewidth(0.6)
    sub.patch.set_alpha(0.85)
    return sub


def concept() -> None:  # noqa: PLR0914, PLR0915 - one figure, linear
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # (a) passagem de mensagens
    ax = axes[0]
    pos = {
        "i": (5, 5.5),
        "a": (2, 8),
        "b": (8, 8.2),
        "c": (1.8, 3),
        "d": (8.2, 2.8),
    }
    for k, (x, y) in pos.items():
        fc = "#ffe6cc" if k == "i" else "#dae8fc"
        ax.add_patch(plt.Circle((x, y), 0.7, fc=fc, ec="#333", lw=1.4))
        ax.text(
            x,
            y,
            "$h_i$" if k == "i" else f"$h_{k}$",
            ha="center",
            va="center",
            fontsize=12,
        )
    for k in "abcd":
        arrow(ax, pos[k], pos["i"], color="#6c8ebf")
        mx = (pos[k][0] + pos["i"][0]) / 2
        my = (pos[k][1] + pos["i"][1]) / 2
        ax.text(mx, my + 0.35, f"$m_{{{k}i}}$", fontsize=10, color="#6c8ebf")
    ax.plot([2, 1.8], [8, 3], color="#bbb", lw=1, zorder=0)
    ax.plot([8, 8.2], [8.2, 2.8], color="#bbb", lw=1, zorder=0)
    ax.text(
        5,
        1.2,
        r"$m_{ji} = \psi(h_j, h_i)$"
        "\n"
        r"$h_i' = \phi(h_i,\ \sum_{j \in \mathcal{N}(i)} m_{ji})$",
        ha="center",
        fontsize=12,
    )
    ax.set_title(
        "(a) GNN: cada nó agrega mensagens\ndos vizinhos", fontsize=11
    )

    # (b) GNN espaço-temporal: cada nó é uma série
    ax = axes[1]
    pos_b = {0: (2, 7.5), 1: (5.2, 8.3), 2: (8, 6.5), 3: (3, 3.2), 4: (7, 2.8)}
    edges_b = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (1, 4)]
    for a, b in edges_b:
        arrow(ax, pos_b[a], pos_b[b], color="#999")
    t = np.arange(48)
    for k, (x, y) in pos_b.items():
        ax.add_patch(plt.Circle((x, y), 0.55, fc="white", ec="#333", lw=1.2))
        for d in range(3):  # cartões empilhados: x_t, x_{t+1}, x_{t+2}
            ax.add_patch(
                FancyBboxPatch(
                    (x - 1.25 + 0.18 * d, y + 1.25 - 0.22 * d),
                    1.5,
                    0.55,
                    boxstyle="round,pad=0.02",
                    fc=["#dae8fc", "#d5e8d4", "#ffe6cc"][d],
                    ec="#777",
                    lw=0.6,
                    zorder=3 + d,
                )
            )
        series = np.sin(2 * np.pi * (t + 5 * k) / 24) + 0.3 * rng.normal(
            size=48
        )
        sparkline(fig, ax, (x - 0.15, y + 1.1), series, w=0.06, h=0.028)
        ax.text(x, y, f"$x_{k}$", ha="center", va="center", fontsize=11)
    ax.text(
        5,
        0.6,
        "nó = série temporal (janela do passado)\n"
        "aresta = dependência entre séries\n"
        "saída = futuro de cada nó",
        ha="center",
        fontsize=10,
    )
    ax.set_title(
        "(b) GNN para séries temporais\n(STGNN, DCRNN, StemGNN)", fontsize=11
    )

    # (c) a aresta do GOTAS
    ax = axes[2]
    pj, pi = (1.6, 6.5), (8.4, 6.5)
    for (x, y), lab, fc in ((pj, "$s_j$", IN_C), (pi, r"$\hat y_i$", OUT_C)):
        ax.add_patch(plt.Circle((x, y), 0.75, fc=fc, ec="#333", lw=1.4))
        ax.text(x, y, lab, ha="center", va="center", fontsize=13)
    ax.add_patch(
        FancyBboxPatch(
            (3.6, 5.6),
            2.8,
            1.8,
            boxstyle="round,pad=0.05",
            fc="#ffe6cc",
            ec="#d79b00",
            lw=1.4,
        )
    )
    ax.text(5, 7.75, r"filtro $H_{ij}(f)$", ha="center", fontsize=11)
    arrow(ax, pj, (3.6, 6.5), color="#333", shrinkB=0)
    arrow(ax, (6.4, 6.5), pi, color="#333", shrinkA=0)
    f = np.linspace(0, 1, 100)
    gain = 1 / (1 + (f / 0.15) ** 2) + 0.4 * np.exp(-(((f - 0.5) / 0.05) ** 2))
    sub = sparkline(fig, ax, (5, 6.45), gain, w=0.06, h=0.07, color="#d79b00")
    sub.set_xlabel("f", fontsize=7, labelpad=0)
    sparkline(
        fig,
        ax,
        (1.6, 4.9),
        np.sin(t / 3.8) + 0.4 * np.sin(t / 1.2),
        w=0.05,
        h=0.03,
    )
    ax.text(1.6, 4.1, "janela de $s_j$", ha="center", fontsize=9)
    ax.text(
        5,
        3.3,
        "mensagem = sinal $s_j$ filtrado pela aresta\n"
        "agregação = soma sobre os emissores $j$",
        ha="center",
        fontsize=10,
    )
    ax.text(
        5,
        1.6,
        r"$\hat Y_i(f) = \sum_j H_{ij}(f)\,S_j(f)$"
        "\n"
        r"$\hat y_i(t) = \sum_j \sum_\tau h_{ij}(\tau)\,s_j(t-\tau) + b_i$",
        ha="center",
        fontsize=12,
    )
    ax.set_title(
        "(c) GOTAS: a mensagem é uma\nfunção de transferência",
        fontsize=11,
    )

    fig.savefig(FIG / "gnn_conceito.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "gnn_conceito.pdf", bbox_inches="tight")
    plt.close(fig)


def reservoir() -> None:  # noqa: PLR0914 - one figure, linear
    train = load_splits()["train"]
    cols = list(INPUT_COLS) + list(OUTPUT_COLS)
    window = train[-168:]
    granger = json.loads(DEP.read_text())["granger_p"]

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    pos = {
        "freq_b1": (1.6, 8.6),
        "freq_b2": (1.6, 6.4),
        "freq_b3": (1.6, 4.2),
        "vazao_entrada": (1.6, 2.0),
        "vazao_distribuicao": (8.6, 8.2),
        "nivel_res": (10.6, 5.0),
        "pressao": (8.6, 1.8),
    }
    for j in cols:
        for i in OUTPUT_COLS:
            if i == j:
                continue
            p = granger.get(f"{j} -> {i}", 1.0)
            strong = p < STRONG_P
            lw = 0.6 + min(-np.log10(max(p, 1e-30)), 30) / 7
            arrow(
                ax,
                pos[j],
                pos[i],
                color="#b85450" if strong else "#bbbbbb",
                lw=lw if strong else 0.8,
                linestyle="-" if strong else (0, (3, 3)),
                connectionstyle="arc3,rad=0.08",
                shrinkA=58,
                shrinkB=58,
                mutation_scale=16,
            )
    for j in cols:
        x, y = pos[j]
        fc = OUT_C if j in OUTPUT_COLS else IN_C
        ax.add_patch(
            plt.Circle((x, y), 0.95, fc=fc, ec="#333", lw=1.5, zorder=5)
        )
        ax.text(
            x,
            y + 0.25,
            LABEL[j],
            ha="center",
            va="center",
            fontsize=9,
            zorder=6,
        )
        sparkline(
            fig, ax, (x, y - 0.45), window[:, cols.index(j)], w=0.06, h=0.035
        )
    ax.text(1.6, 9.75, "entradas $u$", ha="center", fontsize=12, weight="bold")
    ax.text(9.6, 9.75, "saídas $y$", ha="center", fontsize=12, weight="bold")
    ax.text(
        5.6,
        0.25,
        "nó = série (trecho real de 7 dias do treino)  ·  "
        "vermelho: Granger p < 10⁻³ (espessura ∝ −log p)  ·  "
        "tracejado: dependência fraca",
        ha="center",
        fontsize=9,
        color="#444",
    )
    ax.set_title(
        "O reservatório como grafo: cada aresta j → i vira uma "
        "função de transferência $H_{ij}(f)$ no GOTAS",
        fontsize=12,
    )
    fig.savefig(FIG / "gnn_reservatorio.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "gnn_reservatorio.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    concept()
    reservoir()
    for name in ("gnn_conceito.pdf", "gnn_reservatorio.pdf"):
        shutil.copy(FIG / name, OVERLEAF / name)
