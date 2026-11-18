"""Arquitetura do GOTAS em duas figuras: fluxo da previsão e blocos internos.

Cada figura é descrita uma vez e gravada de dois modos: um `.drawio` editável
em `docs/figures/` e um PDF em `imagens/` do artigo. Depois de editar à mão no
draw.io, exporte para o mesmo nome de PDF e não rode este script de novo.

Uso (da raiz do repo):
    uv run --extra ml python \
        notebooks/training/scripts/figure_architecture_split.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from figure_style import WIDTH, apply_style, save
from matplotlib.patches import FancyBboxPatch

DRAWIO = Path(__file__).resolve().parents[3] / "docs" / "figures"
PAGE_CM = 16.0
PX_PER_CM = 40
FONT_PX = 14
COLORS = {
    "dados": ("#d6e6f2", "#1f6a9c"),
    "espectro": ("#fbe3cf", "#b8641c"),
    "residuo": ("#e8dcf1", "#6f4596"),
    "fisica": ("#d7ecdd", "#26784a"),
    "saida": ("#e7e7e7", "#444444"),
}
SIDES = {"r": (1, 0.5), "l": (0, 0.5), "t": (0.5, 0), "b": (0.5, 1)}


@dataclass(frozen=True)
class Box:
    """Bloco, com (x, y) no canto superior esquerdo, em cm."""

    key: str
    x: float
    y: float
    w: float
    h: float
    label: str
    kind: str


@dataclass(frozen=True)
class Edge:
    """Seta entre dois lados de blocos, com pontos intermediários."""

    start: str
    end: str
    out: str = "r"
    into: str = "l"
    via: tuple[tuple[float, float], ...] = ()
    faint: bool = False


def row(  # noqa: PLR0913
    y: float,
    items: list[tuple[str, str, str]],
    h: float = 1.25,
    x0: float = 0.0,
) -> list[Box]:
    """Blocos de mesma largura numa linha da página."""
    gap = 0.33
    w = (PAGE_CM - x0 - gap * (len(items) - 1)) / len(items)
    return [
        Box(key, x0 + k * (w + gap), y, w, h, label, kind)
        for k, (key, label, kind) in enumerate(items)
    ]


def chain(keys: list[str]) -> list[Edge]:
    return [Edge(a, b) for a, b in zip(keys, keys[1:], strict=False)]


def flow_figure() -> tuple[list[Box], list[Edge], dict[str, tuple], float]:
    """Caminho de uma previsão, da medição até o horizonte."""
    top = row(
        0.9,
        [
            ("dados", "Medições e\nentradas futuras", "dados"),
            ("seq", "Sequências\n$s_j$", "dados"),
            ("centro", "Centralização\n$\\tilde{s}_j$", "dados"),
            ("rfft", "RFFT\n$S_j(f)$", "espectro"),
            ("grafo", "Grafo de filtros\n$H_{ij}(f)$", "espectro"),
            ("irfft", "iRFFT\n$\\hat{y}_i(t+k)$", "espectro"),
        ],
    )
    w = top[0].w
    col = [b.x for b in top]
    boxes = [
        *top,
        Box("residuo", col[2], 3.1, w, 1.25, "Resíduo\n$r_i$", "residuo"),
        Box(
            "soma", col[5], 3.1, w, 1.25, "Soma com\n$r_i$ e $\\mu_i$", "saida"
        ),
        Box(
            "saidas",
            col[5],
            5.3,
            w,
            1.25,
            "Saídas diretas\n$\\hat{y}_1, \\hat{y}_2, \\hat{y}_3$",
            "saida",
        ),
        Box(
            "cascata",
            col[4],
            5.3,
            w,
            1.25,
            "Cascata\n$L^{\\mathrm{c}}$",
            "fisica",
        ),
        Box("mistura", col[3], 5.3, w, 1.25, "Combinação\n$w_k$", "fisica"),
        Box("fim", col[2], 5.3, w, 1.25, "Previsão de\n$K$ passos", "saida"),
    ]
    edges = [
        *chain(["dados", "seq", "centro", "rfft", "grafo", "irfft"]),
        Edge(
            "irfft",
            "seq",
            "t",
            "t",
            ((col[5] + w / 2, 0.35), (col[1] + w / 2, 0.35)),
        ),
        Edge("irfft", "soma", "b", "t"),
        Edge("centro", "residuo", "b", "t"),
        Edge("residuo", "soma"),
        Edge("soma", "saidas", "b", "t"),
        Edge("saidas", "cascata", "l", "r"),
        Edge("cascata", "mistura", "l", "r"),
        Edge(
            "saidas",
            "mistura",
            "b",
            "b",
            ((col[5] + w / 2, 7.0), (col[3] + w / 2, 7.0)),
        ),
        Edge("mistura", "fim", "l", "r"),
    ]
    return boxes, edges, {}, 7.2


def blocks_figure() -> tuple[list[Box], list[Edge], dict[str, tuple], float]:
    """Blocos internos: grafo, aresta, recursão, resíduo e nível."""
    senders = [
        ("u1", "$u_1$"),
        ("u2", "$u_2$"),
        ("u3", "$u_3$"),
        ("u4", "$u_4$"),
        ("y1", "$y_1$"),
        ("y2", "$y_2$"),
        ("y3", "$y_3$"),
        ("bal", "$\\Sigma(u_4-y_1)$"),
    ]
    boxes = row(
        0.45,
        [
            (
                k,
                label,
                "dados"
                if k[0] == "u"
                else ("fisica" if k == "bal" else "saida"),
            )
            for k, label in senders
        ],
        h=0.75,
        x0=0.8,
    )
    outs = [
        ("o1", 3.2, "$\\hat{y}_1$"),
        ("o2", 8.0, "$\\hat{y}_2$"),
        ("o3", 12.8, "$\\hat{y}_3$"),
    ]
    boxes += [Box(k, x, 2.55, 1.9, 0.75, t, "saida") for k, x, t in outs]
    edges = [
        Edge(s, o, "b", "t", faint=(s, o) not in {("u4", "o2"), ("y1", "o2")})
        for s, _ in senders
        for o, _, _ in outs
    ]
    b1 = row(
        4.1,
        [
            ("sj", "$\\tilde{s}_j$", "dados"),
            ("rf", "RFFT", "espectro"),
            ("mul", "Produto", "espectro"),
            ("sum", "Soma em $j$", "espectro"),
            ("ir", "iRFFT", "espectro"),
            ("yb", "$\\hat{y}_i$", "saida"),
        ],
        h=0.85,
        x0=0.8,
    )
    mul = next(b for b in b1 if b.key == "mul")
    boxes += b1
    boxes += [
        Box("c", 0.8, 5.45, 2.0, 0.85, "$c_{ij,m}$", "espectro"),
        Box("lam", 3.5, 5.45, 2.0, 0.85, "$\\Lambda_m(f)$", "espectro"),
        Box("h", mul.x, 5.45, mul.w, 0.85, "$H_{ij}(f)$", "espectro"),
    ]
    edges += chain(["sj", "rf", "mul", "sum", "ir", "yb"])
    edges += [Edge("c", "lam"), Edge("lam", "h"), Edge("h", "mul", "t", "b")]
    b2 = row(
        7.25,
        [
            ("sk", "Sequência\nno passo $k$", "dados"),
            ("f1", "Filtro na\nposição $W+k$", "espectro"),
            ("yk", "$\\hat{y}(t+k)$", "saida"),
        ],
        h=1.05,
        x0=0.8,
    )
    boxes += b2
    edges += chain(["sk", "f1", "yk"])
    edges.append(
        Edge(
            "yk",
            "sk",
            "b",
            "b",
            (
                (b2[2].x + b2[2].w / 2, 8.6),
                (b2[0].x + b2[0].w / 2, 8.6),
            ),
        )
    )
    b3 = row(
        9.15,
        [
            ("py", "Passado\nde $y_i$", "dados"),
            ("l1", "Camada\nlinear", "residuo"),
            ("tanh", "tanh", "residuo"),
            ("l2", "Camada\nlinear", "residuo"),
            ("ri", "$r_i$", "saida"),
        ],
        h=1.05,
        x0=0.8,
    )
    boxes += b3
    edges += chain(["py", "l1", "tanh", "l2", "ri"])
    b4 = row(
        10.95,
        [
            ("q", "$\\hat{q} = u_4 - \\hat{y}_1$", "dados"),
            ("bl", "Balanço\n$\\kappa,\\ \\delta$", "fisica"),
            ("lc", "$L^{\\mathrm{c}}(t+k)$", "fisica"),
            ("w", "Combinação\n$w_k$", "fisica"),
            ("nivel", "$\\hat{y}_2(t+k)$", "saida"),
        ],
        h=1.05,
        x0=0.8,
    )
    boxes += b4
    w_box = next(b for b in b4 if b.key == "w")
    boxes.append(
        Box(
            "d",
            w_box.x,
            12.35,
            w_box.w,
            0.8,
            "$\\hat{y}_2^{\\,\\mathrm{direto}}$",
            "saida",
        )
    )
    edges += chain(["q", "bl", "lc", "w", "nivel"])
    edges.append(Edge("d", "w", "t", "b"))
    letters = {
        "a": (0.0, 0.8),
        "b": (0.0, 4.45),
        "c": (0.0, 7.7),
        "d": (0.0, 9.6),
        "e": (0.0, 11.4),
    }
    return boxes, edges, letters, 13.3


def anchor(box: Box, side: str) -> tuple[float, float]:
    fx, fy = SIDES[side]
    return box.x + fx * box.w, box.y + fy * box.h


def draw_pdf(spec: tuple, name: str) -> None:
    boxes, edges, letters, height = spec
    by_key = {b.key: b for b in boxes}
    fig = plt.figure(figsize=(WIDTH, WIDTH * height / PAGE_CM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, PAGE_CM)
    ax.set_ylim(height, 0)
    ax.axis("off")
    for e in sorted(edges, key=lambda e: not e.faint):
        a = anchor(by_key[e.start], e.out)
        b = anchor(by_key[e.end], e.into)
        pts = [a, *e.via, b]
        color, lw = ("0.78", 0.6) if e.faint else ("0.2", 1.1)
        ax.plot(
            *zip(*pts[:-1], pts[-1], strict=True), color=color, lw=lw, zorder=1
        )
        ax.annotate(
            "",
            xy=pts[-1],
            xytext=pts[-2],
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": lw,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=2,
        )
    for b in boxes:
        fill, line = COLORS[b.kind]
        ax.add_patch(
            FancyBboxPatch(
                (b.x, b.y),
                b.w,
                b.h,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc=fill,
                ec=line,
                lw=0.9,
                zorder=3,
            )
        )
        ax.text(
            b.x + b.w / 2,
            b.y + b.h / 2,
            b.label,
            ha="center",
            va="center",
            zorder=4,
        )
    for letter, (x, y) in letters.items():
        ax.text(x, y, f"({letter})", va="center")
    save(fig, f"{name}.pdf")


def drawio_label(label: str) -> str:
    text = re.sub(r"\$(.+?)\$", r"\\(\1\\)", label).replace("\n", "<br>")
    return escape(text, quote=True)


def write_drawio(spec: tuple, name: str) -> None:  # noqa: PLR0914
    boxes, edges, letters, height = spec
    px = PX_PER_CM
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for b in boxes:
        fill, line = COLORS[b.kind]
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={line};fontSize={FONT_PX};"
            "fontFamily=Times New Roman;"
        )
        cells.append(
            f'<mxCell id="{b.key}" value="{drawio_label(b.label)}" '
            f'style="{style}" vertex="1" parent="1"><mxGeometry '
            f'x="{b.x * px:.1f}" y="{b.y * px:.1f}" width="{b.w * px:.1f}" '
            f'height="{b.h * px:.1f}" as="geometry"/></mxCell>'
        )
    for letter, (x, y) in letters.items():
        cells.append(
            f'<mxCell id="p{letter}" value="({letter})" style="text;html=1;'
            f'fontSize={FONT_PX};fontFamily=Times New Roman;" vertex="1" '
            f'parent="1"><mxGeometry x="{x * px:.1f}" '
            f'y="{(y - 0.3) * px:.1f}" '
            'width="30" height="24" as="geometry"/></mxCell>'
        )
    for k, e in enumerate(edges):
        ox, oy = SIDES[e.out]
        ix, iy = SIDES[e.into]
        color = "#c7c7c7" if e.faint else "#333333"
        style = (
            f"endArrow=block;html=1;rounded=1;strokeColor={color};"
            f"strokeWidth={1 if e.faint else 2};exitX={ox};exitY={oy};"
            f"entryX={ix};entryY={iy};"
        )
        points = "".join(
            f'<mxPoint x="{x * px:.1f}" y="{y * px:.1f}"/>' for x, y in e.via
        )
        array = f'<Array as="points">{points}</Array>' if points else ""
        cells.append(
            f'<mxCell id="e{k}" style="{style}" edge="1" parent="1" '
            f'source="{e.start}" target="{e.end}"><mxGeometry relative="1" '
            f'as="geometry">{array}</mxGeometry></mxCell>'
        )
    xml = (
        '<mxfile><diagram name="GOTAS"><mxGraphModel math="1" '
        f'pageWidth="{PAGE_CM * px:.0f}" pageHeight="{height * px:.0f}">'
        f"<root>{''.join(cells)}</root></mxGraphModel></diagram></mxfile>"
    )
    (DRAWIO / f"{name}.drawio").write_text(xml, encoding="utf-8")


def main() -> None:
    apply_style()
    for name, spec in (
        ("gotas_fluxo", flow_figure()),
        ("gotas_blocos", blocks_figure()),
    ):
        draw_pdf(spec, name)
        write_drawio(spec, name)


if __name__ == "__main__":
    main()
