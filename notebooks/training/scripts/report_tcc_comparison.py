"""Números da comparação artigo × modelo final (doc 13 e Overleaf).

Junta, no mesmo teste e por saída: persistência e sazonal-ingênuo (as
referências mínimas que o artigo não reporta), o ARMAX/MOGWO publicado
em 1 passo e em free-run de 12 passos, e o modelo final em 1 passo e em
12 passos.

Uso (da raiz do repo):
    uv run --extra ml python \
        notebooks/training/scripts/report_tcc_comparison.py
"""

from __future__ import annotations

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
from report_final_model import final
from report_spectral_tf_net import Y, armax_horizon, armax_one_step

from mimo_sys.training.data import PROJECT_ROOT
from mimo_sys.training.evaluation import r2_per_output

OUT = PROJECT_ROOT / "outputs" / "tcc_comparison.json"
FIG = PROJECT_ROOT / "docs" / "figures" / "tcc_comparacao_r2"
OVERLEAF = PROJECT_ROOT / "overleaf" / "imagens" / "tcc_comparacao_r2.pdf"
# Tabela 2 do artigo e doc 8 §5.1 (SpectralTFNet linear no protocolo do
# artigo: ordem 3, 1 passo, sem u[k], mesmo recorte de treino/validação).
TABLE2_MOGWO = [0.9478, 0.9526, 0.8364]
TABLE2_MOCS = [0.9394, 0.9596, 0.8633]
STF_PAPER_PROTOCOL = [0.926, 0.976, 0.805]
NAMES = ["vazão", "nível", "pressão"]
PAPER_VAL = (1499, 2000)  # linhas MATLAB 1500:2000
TEST_START = 1876  # primeira linha do teste 70/15/15
H = 12


def r2(y: np.ndarray, p: np.ndarray) -> list[float]:
    return r2_per_output(y, p).round(4).tolist()


def main() -> None:
    y = Y.T  # (n, 3): vazão, nível, pressão
    out: dict = {}

    a, b = PAPER_VAL
    out["janela_artigo_1passo"] = {
        "persistencia": r2(y[a:b], y[a - 1 : b - 1]),
        "armax_mogwo_recalculado": r2(y[a:b], armax_one_step(a, b)),
    }

    t = TEST_START
    out["nosso_teste_1passo"] = {
        "persistencia": r2(y[t:], y[t - 1 : -1]),
        "armax_mogwo": r2(y[t:], armax_one_step(t, len(y))),
    }

    origins = np.arange(t, len(y) - H + 1)
    truth = np.stack([y[o : o + H] for o in origins])
    persist = np.stack([np.repeat(y[o - 1][None], H, 0) for o in origins])
    seasonal = np.stack([y[o - 24 : o - 24 + H] for o in origins])
    yt12, yp12 = armax_horizon(H)
    out["nosso_teste_12passos_origem_12h"] = {
        "persistencia": r2(truth[::H], persist[::H]),
        "sazonal_24h": r2(truth[::H], seasonal[::H]),
        "armax_mogwo_free_run": r2(yt12, yp12),
    }
    out["nosso_teste_12passos_todas_origens"] = {
        "persistencia": r2(truth, persist),
        "sazonal_24h": r2(truth, seasonal),
    }

    for scenario in ("full", "pumps"):
        yt, pred, w = final(scenario)
        assert yt.shape == truth.shape
        out[f"modelo_final_{scenario}"] = {
            "passo_1": r2(yt[:, 0], pred[:, 0]),
            "12_passos_origem_12h": r2(yt[::H], pred[::H]),
            "12_passos_todas_origens": r2(yt, pred),
            "por_passo_medio": [
                round(float(r2_per_output(yt[:, h], pred[:, h]).mean()), 4)
                for h in range(H)
            ],
        }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    plot(out)
    print(json.dumps(out, indent=1, ensure_ascii=False))


def plot(out: dict) -> None:
    """Grouped R² bars: the paper's protocol (left) and 12 h (right)."""
    panels = [
        (
            "1 passo à frente, janela de validação do artigo (1500–2000)",
            {
                "persistência": out["janela_artigo_1passo"]["persistencia"],
                "ARMAX MOGWO (Tab. 2)": TABLE2_MOGWO,
                "ARMAX MOCS (Tab. 2)": TABLE2_MOCS,
                "SpectralTFNet, protocolo do artigo": STF_PAPER_PROTOCOL,
            },
        ),
        (
            "12 passos à frente, nosso teste (origem a cada 12 h)",
            {
                "sazonal ingênuo (24 h)": out[
                    "nosso_teste_12passos_origem_12h"
                ]["sazonal_24h"],
                "ARMAX MOGWO (free-run)": out[
                    "nosso_teste_12passos_origem_12h"
                ]["armax_mogwo_free_run"],
                "modelo final, pumps": out["modelo_final_pumps"][
                    "12_passos_origem_12h"
                ],
                "modelo final, full": out["modelo_final_full"][
                    "12_passos_origem_12h"
                ],
            },
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, (title, bars) in zip(axes, panels, strict=True):
        width = 0.2
        for k, (label, vals) in enumerate(bars.items()):
            x = np.arange(3) + (k - 1.5) * width
            ax.bar(x, vals, width, label=label)
            for xi, v in zip(x, vals, strict=True):
                ax.text(
                    xi, max(v, 0) + 0.01, f"{v:.2f}", ha="center", fontsize=7
                )
        ax.set_xticks(range(3), NAMES)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(
            fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2
        )
    axes[0].set_ylabel("R² (teste)")
    fig.tight_layout()
    fig.savefig(FIG.with_suffix(".png"), dpi=150)
    fig.savefig(FIG.with_suffix(".pdf"))
    shutil.copy(FIG.with_suffix(".pdf"), OVERLEAF)


if __name__ == "__main__":
    main()
