"""Diagrama da arquitetura do modelo final (doc 13 e Overleaf).

Uso (da raiz do repo):
    uv run --extra ml python \
        notebooks/training/scripts/figure_final_architecture.py
"""

from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mimo_sys.training.data import PROJECT_ROOT

FIG = PROJECT_ROOT / "docs" / "figures" / "stf_final_arquitetura"
OVERLEAF = PROJECT_ROOT / "overleaf" / "imagens" / "stf_final_arquitetura.pdf"

COLORS = {
    "dados": "#e8eef7",
    "linear": "#fdf3e1",
    "fisica": "#e6f2e6",
    "saida": "#f3e8f5",
}


def box(ax, xy, w, h, title, lines, color) -> None:  # noqa: PLR0913
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=color,
            ec="#555555",
            lw=1,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.18,
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        weight="bold",
    )
    for k, line in enumerate(lines):
        ax.text(
            x + w / 2,
            y + h - 0.55 - 0.3 * k,
            line,
            ha="center",
            va="top",
            fontsize=8.6,
        )


def arrow(ax, start, end, text: str = "") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.3,
            color="#333333",
        )
    )
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.12,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            style="italic",
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    box(
        ax,
        (0.2, 4.6),
        3.3,
        2.3,
        "1. Janela de entrada",
        [
            "passado: 168 h das 7 séries",
            "+ balanço de massa acumulado",
            "horizonte: 12 h das entradas",
            "conhecidas (bombas; vazão de",
            "entrada no cenário full)",
            "saídas futuras = 0 (sem vazamento)",
        ],
        COLORS["dados"],
    )
    box(
        ax,
        (4.1, 4.6),
        4.6,
        2.3,
        "2. Grafo de funções de transferência",
        [
            "aresta j → i = filtro H̄ᵢⱼ(f), 8 bandas log-espaçadas",
            "H̄ = média das 5 seeds (uma equação só)",
            "simulação ARX recursiva: 1 h por vez,",
            "previsão realimentada, 12 passos",
            "+ resíduo MLP (média das 5 seeds)",
            "lasso por (aresta, banda)",
        ],
        COLORS["linear"],
    )
    box(
        ax,
        (9.3, 4.6),
        3.5,
        2.3,
        "3. Saídas diretas",
        [
            "ŷ vazão de distribuição",
            "ŷ pressão",
            "ŷ nível (direto)",
            "",
            "",
            "",
        ],
        COLORS["saida"],
    )
    box(
        ax,
        (4.1, 1.6),
        4.6,
        2.3,
        "4. Cascata física do nível",
        [
            "nível(t+1) = nível(t) + a·(q̄ent − q̄dist) + b",
            "a = 0,00277 m por l/s, b ajustados no treino",
            "vazão de distribuição: a prevista em (3)",
            "vazão de entrada: conhecida (full) ou",
            "a do dia anterior (pumps)",
            "também treinada junto, como loss",
        ],
        COLORS["fisica"],
    )
    box(
        ax,
        (9.3, 1.6),
        3.5,
        2.3,
        "5. Nível combinado",
        [
            "nível = wₕ·cascata + (1 − wₕ)·direto",
            "12 pesos por passo (validação)",
            "full: 0,93 → 0,71",
            "pumps: 0,87 → 0,14",
            "",
            "",
        ],
        COLORS["fisica"],
    )
    box(
        ax,
        (0.2, 1.6),
        3.3,
        2.3,
        "6. Intervalos",
        [
            "conformal por passo e saída",
            "(calibrado na validação)",
            "log σ explícito: passo, hora,",
            "movimento das bombas",
            "cobertura 0,78–0,94 no teste",
            "",
        ],
        COLORS["saida"],
    )
    arrow(ax, (3.5, 5.75), (4.1, 5.75))
    arrow(ax, (8.7, 5.75), (9.3, 5.75))
    arrow(ax, (10.3, 4.6), (7.6, 3.9), "vazão prevista")
    arrow(ax, (12.0, 4.6), (12.0, 3.9), "nível direto")
    arrow(ax, (8.7, 2.75), (9.3, 2.75), "nível cascata")
    # Final forecast goes under the boxes, back to the intervals.
    ax.plot([11.05, 11.05], [1.6, 1.2], color="#333333", lw=1.3)
    ax.plot([11.05, 1.85], [1.2, 1.2], color="#333333", lw=1.3)
    arrow(ax, (1.85, 1.2), (1.85, 1.6))
    ax.text(
        6.5,
        1.27,
        "previsão final (vazão, nível, pressão)",
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
    )
    ax.text(
        6.5,
        0.7,
        "Previsão em produção: forecast_at_origin recebe só o passado e "
        "as entradas futuras conhecidas (testes de perturbação e replay).",
        ha="center",
        fontsize=8.6,
        color="#444444",
    )
    ax.text(
        6.5,
        0.32,
        "Parâmetros: 387 lineares (H̄ + bias) + resíduo 5×2.908 + "
        "cascata 2 + combinação 12.",
        ha="center",
        fontsize=8.6,
        color="#444444",
    )
    ax.set_title(
        "Modelo final: SpectralTFNet com ensemble de H médio, "
        "cascata física e nível combinado",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG.with_suffix(".png"), dpi=150)
    fig.savefig(FIG.with_suffix(".pdf"))
    shutil.copy(FIG.with_suffix(".pdf"), OVERLEAF)
    print("ok", FIG)


if __name__ == "__main__":
    main()
