# ruff: noqa: E501 - rótulos HTML das caixas do diagrama ficam numa linha
"""Gera docs/figures/gotas_ae_arquitetura.drawio (fluxo e painéis).

Largura ~720 px e fonte 16 px: a 16 cm no artigo, o texto fica com
cerca de 10 pt. Cores e letras seguem a figura antiga do GOTAS-ARX.
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).with_name("gotas_ae_arquitetura.drawio")
FS = 16
COL = {
    "cinza": ("#f5f5f5", "#666666"),
    "A": ("#ffe6cc", "#d79b00"),
    "B": ("#fff2cc", "#d6b656"),
    "C": ("#dae8fc", "#6c8ebf"),
    "D": ("#e1d5e7", "#9673a6"),
    "E": ("#d5e8d4", "#82b366"),
    "F": ("#f8cecc", "#b85450"),
}


class Page:
    def __init__(self, name: str, w: int, h: int) -> None:
        self.name, self.w, self.h = name, w, h
        self.cells: list[str] = []
        self.n = 2

    def _id(self) -> str:
        self.n += 1
        return f"{self.name[:3]}{self.n}"

    def box(self, x, y, w, h, html, key="cinza", fs=FS, extra="") -> str:
        fill, stroke = COL[key]
        i = self._id()
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};strokeWidth=1.5;fontSize={fs};{extra}"
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(html)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" '
            f'width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return i

    def frame(self, x, y, w, h, key, title="") -> str:
        fill, stroke = COL[key]
        i = self._id()
        style = (
            f"rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={fill};"
            f"opacity=45;strokeColor={stroke};strokeWidth=2;verticalAlign=top;"
            f"align=left;spacingLeft=8;spacingTop=2;fontSize={FS};fontStyle=1;"
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(title)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" '
            f'width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return i

    def text(self, x, y, w, h, html, fs=FS, align="center") -> str:
        i = self._id()
        style = (
            f"text;html=1;whiteSpace=wrap;align={align};verticalAlign=middle;"
            f"fontSize={fs};"
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(html)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" '
            f'width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return i

    def circle(self, x, y, d, label, key="cinza", fs=FS) -> str:
        fill, stroke = COL[key]
        i = self._id()
        style = (
            f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};strokeWidth=1.5;fontSize={fs};"
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" '
            f'width="{d}" height="{d}" as="geometry"/></mxCell>'
        )
        return i

    def badge(self, x, y, letter, key) -> None:
        _, stroke = COL[key]
        i = self._id()
        style = (
            f"ellipse;html=1;fillColor={stroke};strokeColor=none;"
            f"fontColor=#ffffff;fontStyle=1;fontSize=13;"
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{letter}" style="{style}" vertex="1" '
            f'parent="1"><mxGeometry x="{x}" y="{y}" width="22" height="22" '
            f'as="geometry"/></mxCell>'
        )

    def edge(
        self,
        a,
        b,
        label="",
        dashed=False,
        exit_=None,
        entry=None,
        points=(),
        width=1.8,
        color="#333333",
        straight=False,
    ) -> None:
        i = self._id()
        style = (
            ("" if straight else "edgeStyle=orthogonalEdgeStyle;")
            + "rounded=1;html=1;endArrow=block;"
            f"endFill=1;strokeWidth={width};strokeColor={color};fontSize=14;"
            "labelBackgroundColor=#ffffff;"
        )
        if dashed:
            style += "dashed=1;"
        for name, pt in (("exit", exit_), ("entry", entry)):
            if pt:
                style += (
                    f"{name}X={pt[0]};{name}Y={pt[1]};{name}Dx=0;{name}Dy=0;"
                )
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
        arr = f'<Array as="points">{pts}</Array>' if pts else ""
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label)}" style="{style}" '
            f'edge="1" parent="1" source="{a}" target="{b}">'
            f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>'
        )

    def xml(self) -> str:
        body = "".join(self.cells)
        return (
            f'<diagram id="{self.name}" name="{self.name}"><mxGraphModel '
            f'dx="{self.w}" dy="{self.h}" grid="1" gridSize="10" guides="1" '
            f'page="1" pageWidth="{self.w}" pageHeight="{self.h}" math="0">'
            f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}'
            f"</root></mxGraphModel></diagram>"
        )


S = lambda t: f"<sub>{t}</sub>"  # noqa: E731
P = lambda t: f"<sup>{t}</sup>"  # noqa: E731

# ---------------------------------------------------------------- fluxo
f = Page("fluxo", 720, 900)
prev = f.box(
    20, 10, 420, 50, "<b>Previsão de 12 passos</b><br>vazão, nível e pressão"
)
nivel = f.box(
    20,
    95,
    420,
    50,
    f"<b>Nível combinado</b> com a cascata<br>ŷ{S('2')} = w{S('k')} L{P('c')} + (1 − w{S('k')}) ŷ{S('2')}{P('direto')}",
    "E",
)
perda = f.box(
    480, 10, 220, 50, "<b>Perda de treino</b><br>com falhas simuladas", "F"
)
casc = f.box(
    480,
    95,
    220,
    70,
    f"<b>Cascata física</b><br>balanço de massa com<br>q̂ = u{S('4')} − ŷ{S('1')} → L{P('c')}",
    "E",
)
saidas = f.box(
    20,
    185,
    420,
    40,
    f"<b>Saídas diretas</b> ŷ{S('1')}, ŷ{S('3')} e ŷ{S('2')}{P('direto')}",
)
soma = f.circle(210, 255, 40, "<b>+</b>", fs=22)
fr = f.frame(10, 320, 440, 330, "B", "em cada banda b")
dec = f.box(
    40,
    360,
    380,
    50,
    f"<b>Decodificador</b> C{S('b')}<br>contexto → contribuição da banda",
    "B",
)
att = f.box(
    40,
    455,
    380,
    60,
    f"<b>Atenção</b> α{S('kτ')} sobre as horas da janela<br>contexto g{P('b')}{S('k')} de cada passo k",
    "B",
)
enc = f.box(
    40,
    560,
    380,
    60,
    f"<b>Codificador</b> E{S('b')}: os nós de cada<br>hora τ → estado latente z{P('b')}{S('τ')}",
    "B",
)
wave = f.box(
    20,
    680,
    420,
    50,
    "<b>Wavelet à trous</b><br>n<sub>w</sub> detalhes e uma aproximação por nó",
    "A",
)
centro = f.box(20, 760, 420, 40, "<b>Centralização</b> pela média da janela")
passado = f.box(
    20,
    830,
    420,
    50,
    f"<b>Passado medido</b> (W horas)<br>u{S('1')}, …, u{S('4')}, y{S('1')}, y{S('2')}, y{S('3')}",
)
cheb = f.box(
    480,
    330,
    220,
    80,
    f"<b>Resíduo de Chebyshev</b><br>polinômio do passado<br>de cada saída → r{S('i')}(k)",
    "D",
)
diret = f.box(
    480,
    470,
    220,
    80,
    f"<b>Ganhos diretos</b><br>D ũ(t+k) e referência<br>a{S('i')} + β{S('ik')}",
    "C",
)
futuro = f.box(
    480, 830, 220, 50, "<b>Entradas futuras</b><br>conhecidas", "cinza"
)
for box, (bx, by), letter, key in (
    (wave, (418, 683), "A", "A"),
    (enc, (398, 563), "B", "B"),
    (att, (398, 458), "B", "B"),
    (dec, (398, 363), "B", "B"),
    (diret, (678, 473), "C", "C"),
    (cheb, (678, 333), "D", "D"),
    (casc, (678, 98), "E", "E"),
    (nivel, (418, 98), "E", "E"),
    (perda, (678, 13), "F", "F"),
):
    f.badge(bx, by, letter, key)
f.edge(passado, centro)
f.edge(centro, wave)
f.edge(wave, enc)
f.edge(enc, att)
f.edge(att, dec)
f.edge(dec, soma)
f.edge(futuro, diret)
f.edge(
    futuro,
    att,
    "consulta",
    exit_=(0, 0.5),
    entry=(1, 0.5),
    points=((455, 855), (455, 485)),
)
f.edge(
    diret,
    soma,
    exit_=(0, 0.5),
    entry=(1, 0.5),
    points=((462, 510), (462, 275)),
)
f.edge(cheb, soma, exit_=(0.5, 0), entry=(1, 0.5), points=((590, 275),))
f.edge(soma, saidas)
f.edge(saidas, nivel, "ŷ" + S("2") + P("direto"))
f.edge(saidas, casc, "ŷ" + S("1"), exit_=(1, 0.5), entry=(0.5, 1))
f.edge(casc, nivel, "L" + P("c"), exit_=(0, 0.5), entry=(1, 0.5))
f.edge(nivel, prev)
f.edge(prev, perda, dashed=True, exit_=(1, 0.5), entry=(0, 0.5))

# --------------------------------------------------------------- painéis
g = Page("paineis", 760, 1000)
W2, H2 = 365, 310
pos = {
    "A": (10, 10),
    "B": (385, 10),
    "C": (10, 340),
    "D": (385, 340),
    "E": (10, 670),
    "F": (385, 670),
}
titles = {
    "A": "A. Decomposição wavelet",
    "B": "B. Autoencoder com atenção",
    "C": "C. Ganhos diretos e referência",
    "D": "D. Resíduo de Chebyshev",
    "E": "E. Nível com a cascata física",
    "F": "F. Treino",
}
for k, (x, y) in pos.items():
    g.frame(x, y, W2, H2, k, titles[k])

# A: cadeia de suavizações
x0, y0 = pos["A"]
c0 = g.box(x0 + 15, y0 + 70, 85, 40, f"x = c{S('0')}", "A")
c1 = g.box(x0 + 140, y0 + 70, 85, 40, f"c{S('1')}", "A")
c2 = g.box(x0 + 265, y0 + 70, 85, 40, f"c{S('2')}", "A")
w1 = g.box(x0 + 140, y0 + 140, 85, 40, f"w{S('1')}", "A")
w2 = g.box(x0 + 265, y0 + 140, 85, 40, f"w{S('2')}", "A")
g.edge(c0, c1)
g.edge(c1, c2)
g.text(x0 + 95, y0 + 42, 50, 26, "1 h", fs=14)
g.text(x0 + 220, y0 + 42, 50, 26, "2 h", fs=14)
g.edge(c1, w1)
g.edge(c2, w2)
g.text(
    x0 + 10,
    y0 + 190,
    W2 - 20,
    110,
    f"c{S('ℓ')}(t) = ½ [c{S('ℓ−1')}(t) + c{S('ℓ−1')}(t − 2{P('ℓ−1')})]<br>"
    f"w{S('ℓ')} = c{S('ℓ−1')} − c{S('ℓ')}<br>"
    f"x = Σ{S('ℓ')} w{S('ℓ')} + aproximação final<br><i>causal e exata</i>",
)

# B: latentes, pesos de atenção e contexto
x0, y0 = pos["B"]
lat = []
for j, (lab, hgt) in enumerate(
    (
        ("z<sub>t−W+1</sub>", 18),
        ("…", 8),
        ("z<sub>τ</sub>", 46),
        ("…", 12),
        ("z<sub>t</sub>", 30),
    )
):
    bx = x0 + 18 + j * 68
    g.box(bx + 10, y0 + 120 - hgt, 40, hgt, "", "B", extra="rounded=0;")
    lat.append(g.box(bx, y0 + 130, 62, 40, lab, "B", fs=14))
g.text(x0 + 15, y0 + 40, 200, 30, f"pesos α{S('kτ')}", fs=15, align="left")
q = g.box(x0 + 15, y0 + 195, 140, 40, f"consulta q{S('k')}", "B")
ctx = g.box(x0 + 210, y0 + 195, 140, 40, f"contexto g{S('k')}", "B")
g.edge(q, ctx)
g.text(
    x0 + 10,
    y0 + 245,
    W2 - 20,
    60,
    f"α{S('kτ')} = softmax(q{S('k')} · r{S('τ')} / √d{S('a')})<br>"
    f"g{S('k')} = Σ{S('τ')} α{S('kτ')} z{S('τ')}",
)

# C: grafo bipartido das entradas para as saídas
x0, y0 = pos["C"]
us = [
    g.circle(x0 + 40, y0 + 45 + 50 * j, 40, f"u{S(str(j + 1))}", "C")
    for j in range(4)
]
ys = [
    g.circle(x0 + 230, y0 + 70 + 55 * j, 40, f"y{S(str(j + 1))}", "C")
    for j in range(3)
]
for u in us:
    for yy in ys:
        g.edge(u, yy, width=1.0, color="#6c8ebf", straight=True)
g.text(
    x0 + 10,
    y0 + 245,
    W2 - 20,
    60,
    f"a{S('i')} + β{S('ik')} + Σ{S('q')} D{S('iq')} ũ{S('q')}(t+k)<br>"
    f"a{S('i')}: média da janela ou último valor",
)

# D: recorrência de Chebyshev
x0, y0 = pos["D"]
t1 = g.box(x0 + 20, y0 + 60, 95, 40, f"T{S('1')} = ξ", "D")
t2 = g.box(x0 + 135, y0 + 60, 95, 40, f"T{S('2')}", "D")
t3 = g.box(x0 + 250, y0 + 60, 95, 40, f"T{S('3')}", "D")
g.edge(t1, t2)
g.edge(t2, t3)
g.text(
    x0 + 10,
    y0 + 115,
    W2 - 20,
    190,
    f"T{S('d+1')} = 2ξ T{S('d')} − T{S('d−1')}<br>"
    f"r{S('i')}(k) = Σ{S('l')} Σ{S('d')} γ{S('ikld')} T{S('d')}(ξ{S('i')}(t−l+1))<br>"
    f"ξ{S('i')}: saída centrada, limitada a [−1, 1]<br><i>polinômio explícito, iniciado em zero</i>",
)

# E: cascata e combinação
x0, y0 = pos["E"]
lc = g.box(
    x0 + 20,
    y0 + 55,
    150,
    50,
    f"cascata L{P('c')}<br>κ ∫ (u{S('4')} − ŷ{S('1')})",
    "E",
    fs=15,
)
yd = g.box(
    x0 + 195,
    y0 + 55,
    150,
    50,
    f"saída direta<br>ŷ{S('2')}{P('direto')}",
    "E",
    fs=15,
)
mix = g.box(
    x0 + 95,
    y0 + 150,
    175,
    45,
    f"w{S('k')} L{P('c')} + (1−w{S('k')}) ŷ{S('2')}{P('direto')}",
    "E",
    fs=15,
)
g.edge(lc, mix, exit_=(0.5, 1), entry=(0.25, 0))
g.edge(yd, mix, exit_=(0.5, 1), entry=(0.75, 0))
g.text(
    x0 + 10,
    y0 + 210,
    W2 - 20,
    90,
    f"L{P('c')}(t+k) = L{P('c')}(t+k−1) + κ q̂ + δ<br>w{S('k')} ajustado na validação, por passo",
)

# F: perda e falhas simuladas
x0, y0 = pos["F"]
for j, lab in enumerate(("zero", "valor preso", "saturação")):
    g.box(x0 + 15 + j * 115, y0 + 60, 105, 40, lab, "F", fs=15)
g.text(
    x0 + 10,
    y0 + 105,
    W2 - 20,
    50,
    "em 30% das janelas, até 24 h,<br>com o valor real como alvo",
    fs=15,
)
g.text(
    x0 + 10,
    y0 + 170,
    W2 - 20,
    130,
    f"perda = L{S('K')} + λ{S('c')} MSE(L{P('c')}) + ρ P(E, C, D)<br>"
    f"P: <i>sparse-group lasso</i>, um grupo por aresta",
)

OUT.write_text(
    '<mxfile host="Electron" type="device">'
    + f.xml()
    + g.xml()
    + "</mxfile>\n",
    encoding="utf-8",
)
print("escrito", OUT)
