"""Separa a arquitetura do GOTAS em fluxograma e painéis.

Parte da imagem exportada do draw.io (`stf_arquitetura_drawio.png`), sem
redesenhar nada: recorta o fluxograma da esquerda e os painéis da direita e
apaga, no fluxograma, as setas tracejadas que ligavam as duas partes.

Uso (da raiz do repo):
    uv run --extra ml python \
        notebooks/training/scripts/figure_architecture_split.py
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from figure_style import ARTICLE_IMAGES
from PIL import Image

SOURCE = ARTICLE_IMAGES / "stf_arquitetura_drawio.png"
# Coordenadas medidas com a imagem em 1960 px de largura: (x0, y0, x1, y1).
REFERENCE_WIDTH = 1960
FLOW_BOX = (0, 35, 1019, 1765)
PANELS_BOX = (1040, 8, 1952, 1892)
# Setas sobre fundo branco: some o que é colorido; o preto fica.
ON_WHITE = (
    (695, 600, 1019, 622),
    (695, 622, 716, 1200),
    (700, 935, 1019, 966),
)
# Trechos sobre outros elementos: copia uma faixa limpa da mesma coluna.
# (x0, y0, x1, y1, y da faixa limpa)
COPY_ROWS = (
    (620, 1198, 716, 1222, 1226),  # saída do bloco do grafo
    (640, 770, 765, 800, 810),  # rótulo "painéis A e B"
    (680, 935, 700, 966, 970),  # saída da simulação recursiva
    (860, 358, 1019, 390, 395),  # saída do bloco da cascata
    (1005, 1238, 1019, 1258, 1262),  # saída do bloco do resíduo
)
# Canto coberto pelo rótulo: espelha o canto esquerdo, (x0, x1, y0, y1).
CORNER = (670, 690, 752, 800)
MIRROR_AXIS = 732
MIN_SATURATION = 0.08


def scaled(box: tuple[int, ...], factor: float) -> tuple[int, ...]:
    return tuple(round(v * factor) for v in box)


def clean_flow(image: npt.NDArray, factor: float) -> npt.NDArray:
    """Fluxograma sem as setas que iam para os painéis."""
    flow = image.copy()
    top = flow.max(axis=2).astype(float)
    saturation = (top - flow.min(axis=2)) / np.maximum(top, 1)
    for box in ON_WHITE:
        x0, y0, x1, y1 = scaled(box, factor)
        colored = saturation[y0:y1, x0:x1] > MIN_SATURATION
        flow[y0:y1, x0:x1][colored] = 255
    for *box, clean_y in COPY_ROWS:
        x0, y0, x1, y1 = scaled(tuple(box), factor)
        start = round(clean_y * factor)
        flow[y0:y1, x0:x1] = flow[start : start + y1 - y0, x0:x1]
    x0, x1, y0, y1 = scaled(CORNER, factor)
    axis = round(MIRROR_AXIS * factor)
    flow[y0:y1, x0:x1] = flow[y0:y1, axis - x1 : axis - x0][:, ::-1]
    return flow


def main() -> None:
    image = np.array(Image.open(SOURCE).convert("RGB"))
    factor = image.shape[1] / REFERENCE_WIDTH
    for name, array, box in (
        ("stf_arquitetura_fluxo.png", clean_flow(image, factor), FLOW_BOX),
        ("stf_arquitetura_paineis.png", image, PANELS_BOX),
    ):
        x0, y0, x1, y1 = scaled(box, factor)
        Image.fromarray(array[y0:y1, x0:x1]).save(ARTICLE_IMAGES / name)
        print(name, x1 - x0, y1 - y0)


if __name__ == "__main__":
    main()
