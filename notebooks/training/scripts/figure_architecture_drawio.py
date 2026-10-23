"""Diagrama draw.io do modelo final (estilo Transformer/StemGNN).

Coluna principal com o fluxo de uma previsão de 12 h e painéis A-F com
a matemática de cada bloco: o grafo, a aresta espectral, a simulação
recursiva, o resíduo, a cascata com a combinação do nível e a perda.

Uso (da raiz do repo):
    uv run python notebooks/training/scripts/figure_architecture_drawio.py
"""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

from mimo_sys.training.data import PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "figures" / "stf_arquitetura.drawio"
COLORS = {
    "input": ("#f8cecc", "#b85450"),
    "spec": ("#ffe6cc", "#d79b00"),
    "norm": ("#fff2cc", "#d6b656"),
    "res": ("#dae8fc", "#6c8ebf"),
    "phys": ("#d5e8d4", "#82b366"),
    "out": ("#e1d5e7", "#9673a6"),
    "gray": ("#f5f5f5", "#666666"),
}
EDGE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;"
    "endFill=1;strokeWidth=1.5;strokeColor=#333333;fontSize=11;"
)
cells: list[str] = []


def vertex(
    cid: str, x: float, y: float, w: float, h: float, label: str, style: str
) -> None:
    cells.append(
        f'<mxCell id="{cid}" value={quoteattr(label)} style="{style}" '
        f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
        f'height="{h}" as="geometry"/></mxCell>'
    )


def box(
    cid: str,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    kind: str = "gray",
) -> None:
    fill, stroke = COLORS[kind]
    vertex(
        cid,
        x,
        y,
        w,
        h,
        label,
        f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};"
        f"strokeColor={stroke};strokeWidth=1.5;fontSize=12;",
    )


def panel(
    cid: str, x: float, y: float, w: float, h: float, label: str
) -> None:
    vertex(
        cid,
        x,
        y,
        w,
        h,
        f"<b>{label}</b>",
        "rounded=1;whiteSpace=wrap;html=1;arcSize=3;fillColor=#f9f9f9;"
        "strokeColor=#333333;strokeWidth=2;verticalAlign=top;"
        "align=left;spacingLeft=10;spacingTop=4;fontSize=13;",
    )


def text(
    cid: str,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    size: int = 12,
    align: str = "center",
) -> None:
    vertex(
        cid,
        x,
        y,
        w,
        h,
        label,
        f"text;html=1;whiteSpace=wrap;align={align};"
        f"verticalAlign=middle;fontSize={size};",
    )


def circle(cid: str, x: float, y: float, label: str, size: int = 34) -> None:
    vertex(
        cid,
        x,
        y,
        size,
        size,
        label,
        "ellipse;whiteSpace=wrap;html=1;fillColor=#ffffff;"
        "strokeColor=#333333;strokeWidth=1.5;fontSize=18;",
    )


def edge(
    cid: str,
    src: str,
    tgt: str,
    label: str = "",
    points: tuple = (),
    extra: str = "",
) -> None:
    pts = ""
    if points:
        pts = (
            '<Array as="points">'
            + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
            + "</Array>"
        )
    cells.append(
        f'<mxCell id="{cid}" value={quoteattr(label)} '
        f'style="{EDGE}{extra}" edge="1" parent="1" source="{src}" '
        f'target="{tgt}"><mxGeometry relative="1" as="geometry">{pts}'
        "</mxGeometry></mxCell>"
    )


def port(exit_: tuple, entry: tuple) -> str:
    return (
        f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
        f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    )


DASHED = "dashed=1;strokeColor=#999999;endArrow=open;endFill=0;"

# --------------------------------------------------------------- title
text(
    "title",
    40,
    10,
    1640,
    40,
    "<b>GOTAS</b>: grafo de funções de transferência simulado "
    "como ARX, com resíduo não linear e nível pela física",
    size=18,
)

# --------------------------------------------------------- main column
X, W = 140, 420
text(
    "in_lbl",
    120,
    1330,
    220,
    44,
    "Passado medido (168 h)<br><i>u₁..u₄, y₁..y₃</i>",
)
text(
    "fut_lbl",
    380,
    1330,
    220,
    44,
    "Entradas conhecidas (12 h)<br><i>bombas (+ u₄ no full)</i>",
)
box(
    "seq",
    X,
    1230,
    W,
    70,
    "<b>Sequência por nó</b> s<sub>j</sub> ∈ ℝ<sup>W+H</sup>, W = 168, "
    "H = 12<br>saídas futuras = 0 · nó extra: Σ(u₄ − y₁) das últimas horas",
    "input",
)
box(
    "norm",
    X,
    1150,
    W,
    50,
    "<b>Centralização</b> s̃<sub>j</sub> = s<sub>j</sub> − μ<sub>j</sub> "
    "(μ do passado; sem dividir por σ)",
    "norm",
)
vertex(
    "loop",
    X - 30,
    620,
    W + 60,
    500,
    "<b>× 12 passos: simulação recursiva (painel C)</b>",
    "rounded=1;whiteSpace=wrap;html=1;arcSize=3;fillColor=#f5f5f5;"
    "strokeColor=#333333;strokeWidth=2;verticalAlign=top;align=left;"
    "spacingLeft=10;fontSize=12;",
)
box(
    "rfft",
    X + 20,
    1040,
    W - 40,
    46,
    "<b>RFFT</b> com zero-padding 2(W+H) → S<sub>j</sub>(f)",
    "spec",
)
box(
    "filt",
    X + 20,
    940,
    W - 40,
    70,
    "<b>Grafo de filtros</b> (painéis A e B)<br>Y<sub>i</sub>(f) = "
    "Σ<sub>j</sub> H<sub>ij</sub>(f) · S<sub>j</sub>(f)",
    "spec",
)
box(
    "irfft",
    X + 20,
    860,
    W - 40,
    46,
    "<b>iRFFT</b>, posição W+k → ŷ<sub>i</sub>(t+k) + b<sub>i</sub>",
    "spec",
)
box(
    "write",
    X + 20,
    770,
    W - 40,
    62,
    "<b>Escreve ŷ(t+k) na sequência</b><br>a saída futura k passa a ser a "
    "própria previsão",
    "norm",
)
text(
    "loop_note",
    X + 20,
    670,
    W - 40,
    80,
    "<i>equação de 1 passo, iterada:</i><br>ŷ<sub>i</sub>(t) = "
    "Σ<sub>j</sub> Σ<sub>τ</sub> h<sub>ij</sub>(τ) s̃<sub>j</sub>(t−τ) + "
    "b<sub>i</sub>",
)
box(
    "resid",
    630,
    930,
    210,
    120,
    "<b>Resíduo não linear</b> (painel D)<br>passado de y<sub>i</sub> → "
    "Linear(168→16) → tanh → Linear(16→12)<br>iniciado em 0",
    "res",
)
circle("plus", X + W / 2 - 17, 560, "+")
text("plus_lbl", X + W / 2 + 22, 545, 70, 30, "+ μ<sub>i</sub>")
box(
    "direct",
    X,
    470,
    W,
    56,
    "<b>Saídas diretas</b> ŷ₁ vazão · ŷ₃ pressão · ŷ₂<sup>direto</sup> nível",
    "out",
)
box(
    "casc",
    630,
    370,
    250,
    96,
    "<b>Cascata física</b> (painel E)<br>balanço de massa com<br>q̂ = u₄ − ŷ₁ "
    "→ nível<sup>cascata</sup>",
    "phys",
)
box(
    "blend",
    X,
    350,
    W,
    70,
    "<b>Nível combinado</b> (painel E)<br>ŷ₂(t+h) = w<sub>h</sub>·cascata + "
    "(1 − w<sub>h</sub>)·direto",
    "phys",
)
text(
    "out_lbl",
    X,
    240,
    W,
    60,
    "<b>Previsão de 12 h</b><br>vazão, nível e pressão",
    size=14,
)

edge("e1", "seq", "norm")
edge("e2", "norm", "rfft")
edge("e3", "rfft", "filt")
edge("e4", "filt", "irfft")
edge("e5", "irfft", "write")
edge(
    "eloop",
    "write",
    "rfft",
    "próximo passo k+1",
    ((X - 10, 801), (X - 10, 1063)),
    port((0, 0.5), (0, 0.5)),
)
edge(
    "eres_in",
    "norm",
    "resid",
    "passado de y<sub>i</sub>",
    ((735, 1175),),
    port((1, 0.5), (0.5, 1)),
)
edge("e6", "loop", "plus", extra=port((0.5, 0), (0.5, 1)))
edge(
    "eres_out",
    "resid",
    "plus",
    points=((735, 577),),
    extra=port((0.5, 0), (1, 0.5)),
)
edge("e7", "plus", "direct")
edge("e8", "direct", "casc", "ŷ₁", ((755, 498),), port((1, 0.5), (0.5, 1)))
edge("e9", "direct", "blend", "ŷ₂ direto")
edge("e10", "casc", "blend", "cascata", extra=port((0, 0.5), (1, 0.5)))
edge("e11", "blend", "out_lbl")
edge("ein", "in_lbl", "seq", extra=port((0.5, 0), (0.25, 1)))
edge("efut", "fut_lbl", "seq", extra=port((0.5, 0), (0.75, 1)))
edge(
    "eu4",
    "fut_lbl",
    "casc",
    "u₄ futura (full) ou<br>do dia anterior (pumps)",
    ((920, 1352),),
    port((1, 0.5), (1, 0.5)) + "dashed=1;",
)

# ------------------------------------------------------ panels (right)
PX, PW = 1000, 680

# A: the graph (GNN view)
panel(
    "pA",
    PX,
    60,
    PW,
    300,
    "A · O grafo: 8 nós emissores → 3 saídas, cada aresta é H<sub>ij</sub>(f)",
)
senders = [
    ("u1", "u₁ bomba 1", "input"),
    ("u2", "u₂ bomba 2", "input"),
    ("u3", "u₃ bomba 3", "input"),
    ("u4", "u₄ vazão entrada", "input"),
    ("y1", "y₁ vazão dist.", "out"),
    ("y2", "y₂ nível", "out"),
    ("y3", "y₃ pressão", "out"),
    ("bm", "Σ(u₄ − y₁)", "norm"),
]
for k, (cid, lab, kind) in enumerate(senders):
    fill, stroke = COLORS[kind]
    vertex(
        f"n_{cid}",
        PX + 20,
        95 + 31 * k,
        150,
        25,
        lab,
        f"rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor={fill};"
        f"strokeColor={stroke};fontSize=11;",
    )
targets = [("t1", "ŷ₁(t+k)"), ("t2", "ŷ₂(t+k)"), ("t3", "ŷ₃(t+k)")]
for k, (cid, lab) in enumerate(targets):
    fill, stroke = COLORS["out"]
    vertex(
        f"n_{cid}",
        PX + 400,
        125 + 70 * k,
        110,
        34,
        lab,
        f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};"
        f"strokeColor={stroke};strokeWidth=1.5;fontSize=12;",
    )
physics = {("u4", "t2"), ("y1", "t2")}
for cid, _, _ in senders:
    for tid, _ in targets:
        strong = (cid, tid) in physics
        color = "#82b366" if strong else "#c8c8c8"
        width = 2.5 if strong else 0.8
        cells.append(
            f'<mxCell id="g_{cid}_{tid}" style="html=1;endArrow=classic;'
            f'endSize=4;strokeColor={color};strokeWidth={width};" edge="1" '
            f'parent="1" source="n_{cid}" target="n_{tid}">'
            '<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
text(
    "a_note",
    PX + 525,
    95,
    145,
    250,
    "<b>nó</b> = série (sequência s<sub>j</sub>)<br><br><b>aresta j → i</b> "
    "= filtro H<sub>ij</sub>(f)<br><br><b>mensagem</b> = s<sub>j</sub> "
    "filtrado<br><br><b>agregação</b> = Σ<sub>j</sub><br><br>"
    "<font color='#82b366'><b>verde</b></font>: o balanço de massa "
    "(u₄ → nível, y₁ → nível)",
    size=11,
    align="left",
)

# B: spectral edge
panel(
    "pB",
    PX,
    380,
    PW,
    260,
    "B · Aresta espectral j → i: função de transferência em 8 bandas",
)
box("b_sj", PX + 20, 420, 120, 50, "s̃<sub>j</sub>(t)<br>W+H amostras", "input")
box("b_fft", PX + 165, 420, 120, 50, "RFFT<br>n = 2(W+H)", "spec")
circle("b_mul", PX + 312, 428, "×")
box(
    "b_sum", PX + 375, 420, 120, 50, "Σ<sub>j</sub> sobre<br>emissores", "spec"
)
box("b_ifft", PX + 525, 420, 130, 50, "iRFFT → ŷ<sub>i</sub>", "spec")
box(
    "b_bands",
    PX + 20,
    505,
    300,
    70,
    "<b>8 coeficientes complexos</b> c<sub>ij,m</sub><br>centros "
    "log-espaçados (m = 0 em DC)<br>3 × 8 × 8 × 2 + 3 = 387 parâmetros",
    "norm",
)
box(
    "b_interp",
    PX + 345,
    505,
    310,
    70,
    "<b>Interpolação</b> H<sub>ij</sub>(f) = Σ<sub>m</sub> Λ<sub>m</sub>(f) "
    "c<sub>ij,m</sub><br>Λ<sub>m</sub>: chapéus lineares, "
    "Σ<sub>m</sub> Λ<sub>m</sub>(f) = 1",
    "spec",
)
text(
    "b_eq",
    PX + 20,
    585,
    PW - 40,
    45,
    "h<sub>ij</sub>(τ) = iRFFT(H<sub>ij</sub>)(τ): o kernel lido da aresta; "
    "o zero-padding torna a convolução linear, não circular",
)
edge("b1", "b_sj", "b_fft")
edge("b2", "b_fft", "b_mul")
edge("b3", "b_mul", "b_sum")
edge("b4", "b_sum", "b_ifft")
edge("b5", "b_bands", "b_interp")
edge(
    "b6",
    "b_interp",
    "b_mul",
    "H<sub>ij</sub>(f)",
    extra=port((0.05, 0), (0.5, 1)),
)

# C: recursion
panel(
    "pC",
    PX,
    660,
    PW,
    210,
    "C · Simulação recursiva de 1 passo (como se simula um ARX)",
)
box(
    "c_seq",
    PX + 20,
    705,
    250,
    62,
    "sequência no passo k:<br>[ passado 168 h | ŷ(t+1..t+k−1) | 0 … ]",
    "input",
)
box(
    "c_step",
    PX + 300,
    705,
    170,
    62,
    "filtro de 1 passo<br>na posição W+k",
    "spec",
)
box("c_out", PX + 500, 705, 160, 62, "ŷ(t+k)", "out")
text(
    "c_note",
    PX + 20,
    780,
    PW - 40,
    80,
    "• auto-arestas H<sub>ii</sub>: parte autorregressiva (polos); "
    "arestas das entradas: parte exógena<br>• entradas conhecidas entram "
    "passo a passo; saídas futuras recebem a própria previsão<br>• um "
    "filtro aplicado de uma vez dá R² negativo em 12 h; iterado, 0,79",
    size=11,
    align="left",
)
edge("c1", "c_seq", "c_step")
edge("c2", "c_step", "c_out")
edge(
    "c3",
    "c_out",
    "c_seq",
    "k ← k+1",
    ((PX + 580, 690), (PX + 145, 690)),
    port((0.5, 0), (0.5, 0)),
)

# D: residual
panel("pD", PX, 890, PW, 170, "D · Resíduo não linear (por saída)")
box(
    "d_in",
    PX + 20,
    935,
    150,
    60,
    "passado centrado<br>de y<sub>i</sub> (168)",
    "input",
)
box("d_l1", PX + 195, 935, 120, 60, "Linear<br>168 → 16", "res")
box("d_th", PX + 340, 935, 80, 60, "tanh", "res")
box("d_l2", PX + 445, 935, 120, 60, "Linear<br>16 → 12<br>(início = 0)", "res")
box("d_out", PX + 590, 935, 70, 60, "r<sub>i</sub>", "out")
text(
    "d_note",
    PX + 20,
    1005,
    PW - 40,
    45,
    "mesma MLP para as 3 saídas; não vê as outras séries: o acoplamento "
    "fica no grafo, onde há equação · 2.908 parâmetros",
    size=11,
)
edge("d1", "d_in", "d_l1")
edge("d2", "d_l1", "d_th")
edge("d3", "d_th", "d_l2")
edge("d4", "d_l2", "d_out")

# E: cascade + blend
panel("pE", PX, 1080, PW, 220, "E · Cascata física do nível e combinação")
box(
    "e_casc",
    PX + 20,
    1120,
    310,
    110,
    "<b>Balanço de massa</b> (integrador)<br>N(t+k) = N(t+k−1) + "
    "a·½(q̂<sub>k</sub> + q̂<sub>k−1</sub>) + b<br>q̂ = u₄ − ŷ₁, "
    "a = 0,00277 m/(l/s)<br>parte do nível medido",
    "phys",
)
box(
    "e_blend",
    PX + 350,
    1120,
    310,
    110,
    "<b>Combinação por passo</b><br>ŷ₂(t+h) = w<sub>h</sub>·N(t+h) + "
    "(1−w<sub>h</sub>)·ŷ₂<sup>direto</sup><br>12 pesos por mínimos "
    "quadrados na validação<br>full: 0,93 → 0,70 · pumps: 0,87 → 0,14",
    "phys",
)
text(
    "e_note",
    PX + 20,
    1240,
    PW - 40,
    50,
    "<i>perto da origem a cascata integrou pouco erro e domina; com o "
    "horizonte, o erro da vazão se acumula e a saída direta ganha peso</i>",
    size=11,
)
edge("e_1", "e_casc", "e_blend")

# F: loss
panel("pF", PX, 1320, PW, 150, "F · Função de perda (treino)")
text(
    "f_eq",
    PX + 20,
    1355,
    PW - 40,
    105,
    "ℒ = Σ<sub>i</sub> [ MSE<sub>i</sub> / 2σ<sub>i</sub>² + log "
    "σ<sub>i</sub> ] + λ<sub>c</sub>·MSE(nível<sup>cascata</sup>) + "
    "ρ·[ α Σ |c<sub>ij,m</sub>| + (1−α) Σ ‖c<sub>ij,·</sub>‖ ]<br><br>"
    "pesos por saída aprendidos (3 parâmetros) · cascata na perda: o erro "
    "do nível chega à vazão · lasso por grupo: zera bandas e arestas "
    "inteiras (ρ = 10⁻³)",
    size=12,
    align="left",
)

# dashed links main -> panels
edge(
    "z1",
    "filt",
    "pB",
    points=((950, 975), (950, 510)),
    extra=DASHED + port((1, 0.5), (0, 0.5)),
)
edge(
    "z0",
    "filt",
    "pA",
    points=((940, 960), (940, 210)),
    extra=DASHED + port((1, 0.3), (0, 0.5)),
)
edge("z2", "loop", "pC", extra=DASHED + port((1, 0.05), (0, 0.5)))
edge("z3", "resid", "pD", extra=DASHED + port((1, 0.5), (0, 0.5)))
edge(
    "z4",
    "casc",
    "pE",
    points=((965, 395), (965, 1190)),
    extra=DASHED + port((1, 0.25), (0, 0.5)),
)

model = (
    '<mxGraphModel grid="1" gridSize="10" page="1" pageWidth="1750" '
    'pageHeight="1500"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
    + "".join(cells)
    + "</root></mxGraphModel>"
)

if __name__ == "__main__":
    OUT.write_text(
        '<mxfile><diagram name="Arquitetura" id="arq">'
        + model
        + "</diagram></mxfile>"
    )
    print(OUT, len(model))
