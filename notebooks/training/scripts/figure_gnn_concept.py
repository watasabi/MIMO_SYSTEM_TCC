"""Figuras de conceito: GNN, GNN em séries temporais e o grafo do sistema.

1. `gnn_conceito`: (a) passagem de mensagens num grafo; (b) GNN
   espaço-temporal, cada nó carrega uma janela de série; (c) a aresta do
   GOTAS, em que a mensagem é um filtro `H_ij(f)`.
2. `gnn_reservatorio`: as 7 séries do treino como nós (trecho real de
   7 dias) e as arestas para as saídas, com espessura pela causalidade
   de Granger (`report_graph_dependency.py`).
3. `stgnn_reservatorio`: GNN espaço-temporal no contexto do reservatório,
   no lugar da figura da DCRNN: as séries viram um grafo por hora, uma
   operação no grafo mistura os nós, uma operação no tempo encadeia as
   horas e a saída são os grafos (e as séries) do futuro.

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
OVERLEAF_REPO = PROJECT_ROOT.parent / "TCC_MIMO_SYS_overleaf" / "imagens"
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


# Grafo do reservatório em miniatura, em layout orgânico (estilo da DCRNN);
# arestas com dependência forte no treino (Granger, `gnn_reservatorio`).
MINI_POS = {
    "freq_b1": (-1.05, 0.95),
    "freq_b2": (-1.25, 0.0),
    "freq_b3": (-0.85, -0.85),
    "vazao_entrada": (0.25, -1.15),
    "vazao_distribuicao": (-0.15, 0.3),
    "nivel_res": (1.0, -0.3),
    "pressao": (0.8, 1.0),
}
MINI_EDGES = [
    ("freq_b1", "vazao_distribuicao"),
    ("freq_b2", "vazao_distribuicao"),
    ("freq_b3", "vazao_distribuicao"),
    ("freq_b1", "pressao"),
    ("freq_b2", "pressao"),
    ("vazao_entrada", "nivel_res"),
    ("vazao_distribuicao", "nivel_res"),
    ("vazao_distribuicao", "pressao"),
    ("nivel_res", "pressao"),
]
NODE_C, EDGE_C, CARD_EC = "#5b8fd1", "#7fa3d6", "#cccccc"


def mini_graph(  # noqa: PLR0913
    ax,  # noqa: ANN001
    cx: float,
    cy: float,
    scale: float,
    values: dict[str, float] | None,
    zorder: int,
    only_outputs: bool = False,
) -> None:
    """Reservoir graph on a card; node shade = value at that hour."""
    half = 1.7 * scale
    for dx, fc, ec in ((0.07, "#00000022", "none"), (0, "white", CARD_EC)):
        ax.add_patch(
            FancyBboxPatch(
                (cx - half + dx, cy - half - dx),
                2 * half,
                2 * half,
                boxstyle="round,pad=0.06",
                fc=fc,
                ec=ec,
                lw=0.9,
                zorder=zorder,
            )
        )
    xy = {
        k: (cx + scale * x, cy + scale * y) for k, (x, y) in MINI_POS.items()
    }
    for a, b in MINI_EDGES:
        ax.plot(
            [xy[a][0], xy[b][0]],
            [xy[a][1], xy[b][1]],
            color=EDGE_C,
            lw=0.8,
            zorder=zorder + 1,
        )
    for k, (x, y) in xy.items():
        if values is None or (only_outputs and k not in OUTPUT_COLS):
            fc = "#e3e3e3"
        else:
            fc = plt.cm.Blues(0.35 + 0.6 * values[k])
        ax.add_patch(
            plt.Circle(
                (x, y),
                0.24 * scale,
                fc=fc,
                ec="none",
                zorder=zorder + 2,
            )
        )


def stack(  # noqa: PLR0913
    ax,  # noqa: ANN001
    cx: float,
    cy: float,
    rows: list[dict[str, float] | None],
    scale: float = 0.62,
    only_outputs: bool = False,
    fs: float = 10.5,
) -> None:
    """Cards of mini graphs, older hours behind (lower left), as in DCRNN."""
    step, half = 0.34, 1.7 * 0.62
    n = len(rows)
    for d, values in enumerate(rows):
        off = (n - 1 - d) * step
        mini_graph(
            ax, cx - off, cy - off, scale, values, 10 + 4 * d, only_outputs
        )
    # seta do tempo ao longo da diagonal, junto aos cantos superiores
    back = (n - 1) * step
    p0 = (cx - back - half - 0.45, cy - back + half + 0.05)
    p1 = (cx - half - 0.45, cy + half + 0.05)
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9),
        zorder=30,
    )
    ax.text(
        (p0[0] + p1[0]) / 2 - 0.3,
        (p0[1] + p1[1]) / 2 + 0.3,
        "tempo",
        rotation=45,
        ha="center",
        va="center",
        fontsize=fs,
        color="#444",
    )


def block(ax, x0, y0, w, h, fc, ec, title) -> None:  # noqa: ANN001, PLR0913
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.1",
            fc=fc,
            ec=ec,
            lw=1.2,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    )
    ax.text(x0 + w / 2, y0 + h + 0.35, title, ha="center", fontsize=10)


def stgnn() -> None:  # noqa: PLR0914, PLR0915 - one figure, linear
    train = load_splits()["train"]
    cols = list(INPUT_COLS) + list(OUTPUT_COLS)
    seg = train[651:711]  # 60 h sem falhas, com as 7 séries variando
    norm = (seg - seg.min(0)) / (np.ptp(seg, 0) + 1e-9)
    past, n_past = 47, 4  # hora t e quantos grafos do passado desenhar
    future = [past + 4, past + 8, past + 12]

    def at(row: int) -> dict[str, float]:
        return {c: norm[row, cols.index(c)] for c in cols}

    fs_title, fs = 12, 10.5
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 34)
    ax.set_ylim(-0.9, 10)
    ax.axis("off")

    # 1. As séries do reservatório, com as horas marcadas.
    ax.text(
        2.6, 9.55, "séries do reservatório", ha="center", fontsize=fs_title
    )
    ys = np.linspace(8.5, 1.3, len(cols))
    for k, (c, y) in enumerate(zip(cols, ys, strict=True)):
        color = "#b85450" if c in INPUT_COLS else "#9673a6"
        sub = sparkline(
            fig,
            ax,
            (3.3, y),
            seg[: past + 1, k],
            w=0.075,
            h=0.05,
            color=color,
        )
        sub.set_xlim(0, past)
        sub.axvspan(past - n_past + 0.5, past, color="#dae8fc", lw=0)
        ax.text(
            1.3,
            y,
            LABEL[c].replace("\n", " "),
            ha="right",
            va="center",
            fontsize=fs,
        )
    ax.annotate(
        "tempo",
        xy=(4.9, 0.35),
        xytext=(1.8, 0.35),
        fontsize=fs,
        va="center",
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.8),
    )
    arrow(ax, (5.3, 5), (6.9, 5), shrinkA=0, shrinkB=0)
    ax.text(6.1, 3.85, "1 grafo\npor hora", ha="center", fontsize=fs)

    # 2. Um grafo por hora: mesma estrutura, sinais diferentes nos nós.
    stack(
        ax,
        10.2,
        4.6,
        [at(r) for r in range(past - n_past + 1, past + 1)],
    )
    ax.text(
        9.6, 8.0, "sinais no grafo\n(passado)", ha="center", fontsize=fs_title
    )
    ax.text(9.6, 1.75, r"$t-W+1,\ \dots,\ t$", ha="center", fontsize=fs)

    # 3. Bloco espaço-temporal: operação no grafo, depois no tempo.
    block(ax, 12.0, 1.4, 10.6, 6.4, "#f7f9fd", "#6c8ebf", "")
    ax.text(
        17.3,
        8.3,
        "bloco espaço-temporal\n(repetido em camadas)",
        ha="center",
        fontsize=fs_title,
    )
    # 3a. operação no grafo: mensagens dos vizinhos para um nó
    ax.add_patch(
        FancyBboxPatch(
            (12.4, 1.9),
            4.5,
            5.3,
            boxstyle="round,pad=0.08",
            fc="#dae8fc",
            ec="#6c8ebf",
            lw=1,
            zorder=2,
        )
    )
    ax.text(14.65, 6.65, "no grafo", ha="center", fontsize=fs_title, zorder=3)
    centre = (14.65, 4.75)
    nbrs = [(13.4, 5.75), (15.9, 5.75), (13.4, 3.75), (15.9, 3.75)]
    for q in nbrs:
        arrow(
            ax,
            q,
            centre,
            color="#6c8ebf",
            shrinkA=8,
            shrinkB=10,
            mutation_scale=9,
            lw=1,
            zorder=3,
        )
        ax.add_patch(
            plt.Circle(q, 0.32, fc="white", ec="#333", lw=0.9, zorder=4)
        )
    ax.add_patch(
        plt.Circle(centre, 0.42, fc="#ffe6cc", ec="#333", lw=1, zorder=4)
    )
    ax.text(
        14.65,
        2.3,
        "mistura entre\nos nós",
        ha="center",
        fontsize=fs,
        zorder=3,
    )
    arrow(ax, (17.0, 4.6), (17.6, 4.6), shrinkA=0, shrinkB=0, zorder=3)
    # 3b. operação no tempo: recorrência em cada nó
    ax.add_patch(
        FancyBboxPatch(
            (17.7, 1.9),
            4.6,
            5.3,
            boxstyle="round,pad=0.08",
            fc="#d5e8d4",
            ec="#82b366",
            lw=1,
            zorder=2,
        )
    )
    ax.text(20.0, 6.65, "no tempo", ha="center", fontsize=fs_title, zorder=3)
    cells = [(18.6, 4.6), (20.0, 4.6), (21.4, 4.6)]
    for k, q in enumerate(cells):
        ax.add_patch(
            FancyBboxPatch(
                (q[0] - 0.4, q[1] - 0.4),
                0.8,
                0.8,
                boxstyle="round,pad=0.03",
                fc="white",
                ec="#333",
                lw=0.9,
                zorder=4,
            )
        )
        ax.text(
            q[0],
            q[1] - 0.85,
            [r"$t-2$", r"$t-1$", r"$t$"][k],
            ha="center",
            fontsize=fs,
            zorder=4,
        )
        if k:
            arrow(
                ax,
                cells[k - 1],
                q,
                shrinkA=9,
                shrinkB=9,
                mutation_scale=8,
                lw=0.9,
                zorder=4,
            )
    ax.text(
        20.0,
        2.3,
        "memória em\ncada nó",
        ha="center",
        fontsize=fs,
        zorder=3,
    )
    arrow(ax, (22.8, 4.6), (23.6, 4.6), shrinkA=0, shrinkB=0)

    # 4. Saída: os grafos do futuro e as séries previstas.
    stack(ax, 26.9, 4.6, [at(r) for r in future], only_outputs=True)
    ax.text(26.4, 8.0, "previsão\n(futuro)", ha="center", fontsize=fs_title)
    ax.text(26.4, 1.75, r"$t+1,\ \dots,\ t+H$", ha="center", fontsize=fs)
    arrow(ax, (28.5, 4.6), (29.4, 4.6), shrinkA=0, shrinkB=0)
    rng = np.random.default_rng(7)
    for c, y in zip(OUTPUT_COLS, (7.0, 4.6, 2.2), strict=True):
        k = cols.index(c)
        sub = sparkline(
            fig,
            ax,
            (31.9, y),
            seg[:, k],
            w=0.07,
            h=0.07,
            color="black",
        )
        hz = np.arange(past, len(seg))
        noise = 0.04 * np.ptp(seg[:, k]) * rng.normal(size=len(hz)).cumsum()
        sub.plot(hz, seg[hz, k] + noise, color="tab:red", lw=1)
        sub.axvline(past, color="#999", lw=0.6)
        sub.set_xlim(0, len(seg) - 1)
        ax.text(
            31.9,
            y + 1.05,
            LABEL[c].replace("\n", " "),
            ha="center",
            fontsize=fs,
        )

    ax.text(
        17.0,
        -0.75,
        "nó = série  ·  aresta = dependência entre séries  ·  "
        "a estrutura do grafo é a mesma em todas as horas; "
        "muda só o valor em cada nó (cor)",
        ha="center",
        fontsize=fs,
        color="#444",
    )
    fig.savefig(FIG / "stgnn_reservatorio.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "stgnn_reservatorio.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    concept()
    reservoir()
    stgnn()
    for name in ("gnn_conceito.pdf", "gnn_reservatorio.pdf"):
        shutil.copy(FIG / name, OVERLEAF / name)
    shutil.copy(FIG / "stgnn_reservatorio.pdf", OVERLEAF_REPO)
