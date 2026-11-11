"""Arquitetura do GOTAS lida pelos canais, da esquerda para a direita.

Mostra o tensor mudando de forma (8 x 180 -> 8 x 181 -> 3 x 12) com um
trecho real do teste dentro de cada canal. Só as partes que diferem das
GNNs citadas ganham cor: a aresta como função de transferência (com o
|H(f)| aprendido em 8 bandas), as entradas futuras conhecidas, a
simulação recursiva e a cascata de balanço de massa. O resto é cinza.

Uso (da raiz do repo):
    uv run python notebooks/training/scripts/figure_gotas_canais.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from mimo_sys.training.data import PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "figures" / "gotas_canais.pdf"
OVERLEAF = Path.home() / "code/TCC_MIMO_SYS_overleaf/imagens/gotas_canais.pdf"
W, H = 168, 12
ORIGIN = 2000  # hora do teste usada como exemplo

GRAY, GRAY_FILL = "#8c8c8c", "#f0f0f0"
EDGE, EDGE_FILL = "#d79b00", "#fff3e6"  # aresta = função de transferência
FUT, FUT_FILL = "#b85450", "#f8cecc"  # entradas futuras conhecidas
REC = "#4f9e9a"  # simulação recursiva
PHYS, PHYS_FILL = "#5b8f45", "#d5e8d4"  # cascata de balanço de massa

data = __import__("pandas").read_parquet(
    PROJECT_ROOT / "data/processed/mimo_data.parquet"
)
data["balanco"] = (
    (data["vazao_entrada"] - data["vazao_distribuicao"]).rolling(24).sum()
)
window = data.iloc[ORIGIN - W + 1 : ORIGIN + H + 1]

rows = [
    ("freq_b1", "bomba 1", "input"),
    ("freq_b2", "bomba 2", "input"),
    ("freq_b3", "bomba 3", "input"),
    ("vazao_entrada", "vazão entrada", "input"),
    ("vazao_distribuicao", "vazão distr.", "output"),
    ("nivel_res", "nível", "output"),
    ("pressao", "pressão", "output"),
    ("balanco", "Σ(u₄ − y₁)", "extra"),
]
outputs = [
    ("vazao_distribuicao", "vazão"),
    ("nivel_res", "nível"),
    ("pressao", "pressão"),
]

fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 133)
ax.set_ylim(-4, 50)
ax.axis("off")

ROW_H, TOP = 3.6, 40.0


def row_y(k: int) -> float:
    return TOP - k * ROW_H


def curve(
    x0: float,
    w: float,
    y0: float,
    h: float,
    v: np.ndarray,
    color: str,
    lw: float = 0.8,
) -> None:
    v = np.asarray(v, dtype=float)
    span = np.ptp(v) or 1.0
    vn = (v - v.min()) / span
    ax.plot(
        np.linspace(x0, x0 + w, len(v)),
        y0 + 0.15 * h + 0.7 * h * vn,
        color=color,
        lw=lw,
    )


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    edge: str,
    fill: str,
    size: float = 8.5,
    bold: bool = False,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.25,rounding_size=0.8",
            ec=edge,
            fc=fill,
            lw=1.4,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        weight="bold" if bold else "normal",
    )


def arrow(
    p0: tuple,
    p1: tuple,
    color: str = "#333333",
    lw: float = 1.3,
    rad: float = 0.0,
    style: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=11,
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


# ---------------------------------------------- 1. tensor de entrada
X0, WT = 6.0, 22.0
xf = X0 + WT * W / (W + H)
for k, (col, lab, kind) in enumerate(rows):
    y = row_y(k)
    ax.add_patch(
        Rectangle((X0, y), WT, ROW_H * 0.9, ec=GRAY, fc="white", lw=0.6)
    )
    ax.text(
        X0 - 0.6, y + ROW_H * 0.45, lab, ha="right", va="center", fontsize=7.5
    )
    v = window[col].to_numpy()
    curve(X0, xf - X0, y, ROW_H * 0.9, v[:W], "#555555", 0.6)
    if kind == "input":
        ax.add_patch(
            Rectangle(
                (xf, y), X0 + WT - xf, ROW_H * 0.9, ec=FUT, fc=FUT_FILL, lw=0.8
            )
        )
        curve(xf, X0 + WT - xf, y, ROW_H * 0.9, v[W:], FUT, 1.0)
    else:
        ax.add_patch(
            Rectangle(
                (xf, y),
                X0 + WT - xf,
                ROW_H * 0.9,
                ec=GRAY,
                fc="white",
                hatch="////",
                lw=0.6,
            )
        )
ax.text(
    X0 + (xf - X0) / 2, TOP + 4.2, "passado (168 h)", ha="center", fontsize=8
)
ax.text(
    xf + (X0 + WT - xf) / 2,
    TOP + 4.2,
    "futuro\n(12 h)",
    ha="center",
    fontsize=8,
    color=FUT,
)
ax.text(
    X0 + WT / 2,
    row_y(7) - 2.2,
    "8 canais × 180 horas",
    ha="center",
    fontsize=8.5,
    weight="bold",
)
ax.text(
    xf - 1.5,
    row_y(7) - 4.4,
    "bombas: valor programado · saídas: 0",
    ha="right",
    fontsize=7.5,
    color=FUT,
)

# ---------------------------------------------- 2. centralizar + FFT
box(
    30.5,
    18,
    6,
    14,
    "centralizar\n(− média do\npassado)\n\nFFT\n360 pontos",
    GRAY,
    GRAY_FILL,
    size=7.5,
)
arrow((X0 + WT + 0.5, 25), (30.3, 25))

# ---------------------------------------------- 3. espectros dos canais
SX, SW = 40.0, 10.0
for k, (col, _, _) in enumerate(rows):
    y = row_y(k)
    v = window[col].to_numpy()
    v = np.where(np.arange(len(v)) < W, v - v[:W].mean(), 0.0)
    if rows[k][2] == "input":
        v[W:] = window[col].to_numpy()[W:] - window[col].to_numpy()[:W].mean()
    spec = np.log10(np.abs(np.fft.rfft(v, n=2 * (W + H))) + 1e-3)
    ax.add_patch(
        Rectangle((SX, y), SW, ROW_H * 0.9, ec=GRAY, fc="white", lw=0.6)
    )
    curve(SX, SW, y, ROW_H * 0.9, spec, GRAY, 0.6)
ax.text(
    SX + SW / 2,
    row_y(7) - 2.2,
    "8 × 181 frequências",
    ha="center",
    fontsize=8.5,
    weight="bold",
)
ax.text(SX + SW / 2, row_y(7) - 4.4, "(complexo)", ha="center", fontsize=7.5)
arrow((36.9, 25), (SX - 0.4, 25))

# ---------------------------------------------- 4. grafo de filtros
GX, GW = 53.0, 30.0
ax.add_patch(
    FancyBboxPatch(
        (GX, 6.5),
        GW,
        39.5,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        ec=EDGE,
        fc=EDGE_FILL,
        lw=2.0,
    )
)
ax.text(
    GX + GW / 2,
    44.3,
    "grafo de filtros",
    ha="center",
    fontsize=9.5,
    weight="bold",
    color=EDGE,
)
ax.text(
    GX + GW / 2,
    42.3,
    "cada aresta j → i é uma função de\n"
    "transferência H$_{ij}$(f), com 8 bandas",
    ha="center",
    va="top",
    fontsize=7.5,
)
src_x, dst_x = GX + 2.0, GX + GW - 2.5
dst_y = [row_y(1) + 1, row_y(4) + 1, row_y(6) + 1]
for k in range(len(rows)):
    for yd in dst_y:
        ax.plot(
            [src_x, dst_x],
            [row_y(k) + 1.6, yd],
            color="#e3c48d",
            lw=0.4,
            zorder=1,
        )
    ax.add_patch(
        plt.Circle((src_x, row_y(k) + 1.6), 0.7, fc="white", ec=GRAY, zorder=3)
    )
    arrow(
        (SX + SW + 0.3, row_y(k) + 1.6),
        (src_x - 0.8, row_y(k) + 1.6),
        color=GRAY,
        lw=0.6,
    )
for yd in dst_y:
    ax.add_patch(
        plt.Circle((dst_x, yd), 1.0, fc="white", ec=EDGE, lw=1.4, zorder=3)
    )
    ax.text(dst_x, yd, "Σ", ha="center", va="center", fontsize=8, zorder=4)

filters = {
    f["aresta"]: f["modulo"]
    for f in json.load(
        open(PROJECT_ROOT / "outputs/extracted_model.json", encoding="utf-8")
    )["filtros_full"]
}
insets = [
    ("freq_b2 -> vazao_distribuicao", "bomba 2 → vazão: ganho", 1, 0),
    ("vazao_entrada -> nivel_res", "vazão entrada → nível: integrador", 3, 1),
]
for key, title, k_src, k_dst in insets:
    ax.plot(
        [src_x, dst_x],
        [row_y(k_src) + 1.6, dst_y[k_dst]],
        color=EDGE,
        lw=1.8,
        zorder=2,
    )
for (key, title, _, _), (ix, iy) in zip(
    insets, [(GX + 6, 27.5), (GX + 6, 9.0)]
):
    iw, ih = 17.0, 7.5
    ax.add_patch(
        Rectangle((ix, iy), iw, ih, ec=EDGE, fc="white", lw=1.0, zorder=5)
    )
    mod = np.array(filters[key])
    xs = np.linspace(ix + 1, ix + iw - 1, len(mod))
    ys = iy + 1.0 + (ih - 3.0) * mod / mod.max()
    ax.plot(xs, ys, color=EDGE, lw=1.4, zorder=6)
    ax.plot(xs, ys, "o", color=EDGE, ms=2.5, zorder=6)
    ax.text(
        ix + iw / 2,
        iy + ih - 0.9,
        title,
        ha="center",
        va="center",
        fontsize=6.8,
        zorder=6,
    )
    ax.text(
        ix + 1,
        iy + 0.1,
        "lento",
        fontsize=5.5,
        va="bottom",
        zorder=6,
        color="#666666",
    )
    ax.text(
        ix + iw - 1,
        iy + 0.1,
        "rápido",
        fontsize=5.5,
        va="bottom",
        ha="right",
        zorder=6,
        color="#666666",
    )
ax.text(
    GX + GW / 2,
    7.3,
    "|H(f)| aprendido: 8 pontos interpolados",
    ha="center",
    fontsize=6.8,
    color=EDGE,
)

# ---------------------------------------------- 5. iRFFT e saídas diretas
box(86.0, 18, 5.5, 14, "iRFFT\n\n+ resíduo\nMLP", GRAY, GRAY_FILL, size=7.5)
for yd in dst_y:
    arrow((dst_x + 1.1, yd), (85.8, 25 + (yd - 25) * 0.25), color=GRAY, lw=0.7)
OX, OW = 94.0, 9.0
future = window.iloc[W:]
for k, (col, lab) in enumerate(outputs):
    y = dst_y[k] - 1.6
    ax.add_patch(
        Rectangle((OX, y), OW, ROW_H * 0.9, ec=GRAY, fc="white", lw=0.6)
    )
    curve(OX, OW, y, ROW_H * 0.9, future[col].to_numpy(), "#333333", 0.9)
    ax.text(OX + OW + 0.6, y + ROW_H * 0.45, lab, va="center", fontsize=7.5)
    arrow((91.7, 25), (OX - 0.3, y + ROW_H * 0.45), color=GRAY, lw=0.7)
ax.text(
    OX + OW / 2,
    dst_y[2] - 4.4,
    "3 saídas × passo k",
    ha="center",
    fontsize=8.5,
    weight="bold",
)

# ---------------------------------------------- 6. simulação recursiva
hx = xf + (X0 + WT - xf) / 2
ax.plot(
    [OX + OW / 2, OX + OW / 2, hx, hx],
    [dst_y[2] - 6.0, 0.3, 0.3, row_y(7) - 5.2],
    color=REC,
    lw=1.8,
)
arrow((hx, row_y(7) - 5.3), (hx, row_y(7) - 0.2), color=REC, lw=1.8)
ax.text(
    63,
    1.3,
    "× 12 passos: ŷ(t+k) volta para o futuro das saídas "
    "(simulação recursiva, como um ARX)",
    ha="center",
    fontsize=8,
    color=REC,
    weight="bold",
)

# ---------------------------------------------- 7. cascata de balanço
TX, TY, TW, TH = 117.0, 9.0, 9.0, 11.0
ax.add_patch(Rectangle((TX, TY), TW, TH, ec=PHYS, fc="white", lw=1.8))
ax.add_patch(Rectangle((TX, TY), TW, TH * 0.55, ec="none", fc=PHYS_FILL))
arrow((TX + 2.0, TY + TH + 4.5), (TX + 2.0, TY + TH - 1.0), color=PHYS)
ax.text(TX + 1.5, TY + TH + 4.6, "u₄", fontsize=8, color=PHYS, ha="right")
arrow((TX + TW - 0.5, TY + 1.0), (TX + TW + 4.0, TY - 1.5), color=PHYS)
ax.text(TX + TW + 4.3, TY - 2.3, "ŷ₁", fontsize=8, color=PHYS)
ax.text(
    TX + TW / 2,
    TY + TH + 8.5,
    "balanço de massa",
    ha="center",
    fontsize=8.5,
    weight="bold",
    color=PHYS,
)
ax.text(
    TX + TW / 2,
    TY - 3.2,
    "nível = cascata física\n+ saída direta (peso por passo)",
    ha="center",
    va="top",
    fontsize=7,
    color=PHYS,
)
arrow(
    (OX + OW + 5.5, dst_y[1] + 0.2),
    (TX + TW - 1.5, TY + TH + 0.3),
    color=PHYS,
    lw=1.2,
    rad=-0.3,
)

# ---------------------------------------------- legenda
ax.text(
    65,
    -2.6,
    "cinza: comum às GNNs para séries temporais citadas   ·   "
    "em cor: o que é próprio do GOTAS",
    ha="center",
    fontsize=8,
    style="italic",
    color="#444444",
)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
shutil.copy(OUT, OVERLEAF)
print(OUT)
