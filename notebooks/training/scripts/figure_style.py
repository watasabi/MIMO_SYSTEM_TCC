"""Estilo das figuras do artigo.

As figuras são geradas no tamanho em que entram na página (16 cm de
largura), com texto de 10 pt, para que nada seja reduzido no LaTeX. O texto
explicativo fica na legenda do artigo, e não na imagem.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ARTICLE_IMAGES = Path.home() / "code/TCC_MIMO_SYS_overleaf/imagens"
WIDTH = 6.3  # polegadas: 16 cm
OUTPUT_LABELS = ["Vazão (l/s)", "Nível (m)", "Pressão (mca)"]
OUTPUT_NAMES = ["Vazão", "Nível", "Pressão"]
INPUT_NAMES = ["Bomba 1", "Bomba 2", "Bomba 3", "Vazão de entrada"]
COLORS = ["#1b1b1b", "#0072b2", "#d55e00", "#009e73"]
LINES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]


def apply_style() -> None:
    """Fonte serifada de 10 pt e traços legíveis na impressão."""
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_letter(ax: plt.Axes, index: int) -> None:
    """Letra do painel, citada na legenda do artigo."""
    ax.set_title(f"({chr(97 + index)})", loc="left", pad=3)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ARTICLE_IMAGES / name)
    plt.close(fig)
    print(name)
