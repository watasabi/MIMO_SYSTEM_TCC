"""Recorta os painéis da arquitetura do GOTAS para uma figura própria.

Parte da imagem exportada do draw.io (`stf_arquitetura_drawio.png`), sem
redesenhar nada. O fluxo da previsão é um diagrama TikZ no `main.tex`, com as
cores e as letras destes painéis.

Uso (da raiz do repo):
    uv run --extra ml python \\
        notebooks/training/scripts/figure_architecture_split.py
"""

from __future__ import annotations

from figure_style import ARTICLE_IMAGES
from PIL import Image

SOURCE = ARTICLE_IMAGES / "stf_arquitetura_drawio.png"
# Coordenadas medidas com a imagem em 1960 px de largura.
REFERENCE_WIDTH = 1960
PANELS_BOX = (1040, 8, 1952, 1892)


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    factor = image.width / REFERENCE_WIDTH
    box = tuple(round(v * factor) for v in PANELS_BOX)
    image.crop(box).save(ARTICLE_IMAGES / "stf_arquitetura_paineis.png")


if __name__ == "__main__":
    main()
