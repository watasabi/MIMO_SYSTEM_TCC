"""Grafo por banda e significância das arestas do modelo final (I0_ref).

Duas figuras do modelo final do artigo (cenário `full`):

1. `stf_modelo_band_strength`: força `|c_ij,m|` de cada aresta em cada
   banda, média dos 5 treinos, em duas linhas de 4 bandas;
2. `stf_modelo_laplace_z`: significância `z = média / dp` do ganho de
   cada aresta nas 3 primeiras horas, em unidades físicas, pela
   aproximação de Laplace dos parâmetros lineares (seed 42), com o
   resíduo não linear fixo no valor treinado.

Uso (da raiz do repo):
    uv run --extra ml python \
        notebooks/training/scripts/figure_extracted_graph.py
"""

from __future__ import annotations

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from report_extracted_model import (
    FIG,
    OUTPUTS,
    OVERLEAF,
    SENDERS,
    VARIANT,
    load,
)
from run_improvements import SEEDS, config

from mimo_sys.training.data import (
    INPUT_IDX,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.train import _windows  # noqa: PLC2701
from mimo_sys.training.uncertainty import (
    coefficient_ci,
    impulse_response_graph,
    laplace_linear_posterior,
)

SUMMARY = PROJECT_ROOT / "outputs" / "extracted_model_laplace.json"
SCENARIO, SEED = "full", 42
LAST_LAG = 3  # ganho nas 3 primeiras horas, como no grafo extraído
Z_LIGHT = 6  # |z| acima do qual o texto da célula fica preto
Z_MAX = 10
Z_SIG = 3  # |z| acima disto: coeficiente significativo


def band_strength_figure() -> None:
    strength = np.mean(
        [load(SCENARIO, s)[0].band_strength().numpy() for s in SEEDS], 0
    )  # (saída, emissor, banda)
    n_bands = strength.shape[-1]
    cols = (n_bands + 1) // 2
    fig, axes = plt.subplots(
        2, cols, figsize=(2.9 * cols, 5.6), sharex=True, sharey=True
    )
    vmax = strength.max()
    for m, ax in enumerate(axes.flat):
        ax.imshow(strength[:, :, m], vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"banda {m}", fontsize=9)
        ax.set_xticks(range(strength.shape[1]))
        ax.set_xticklabels(
            SENDERS[: strength.shape[1]], rotation=90, fontsize=7
        )
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(OUTPUTS)))
        ax.set_yticklabels(OUTPUTS, fontsize=8)
    fig.suptitle(
        "Modelo final (full): |c_ij,m| por banda (0 = DC, log-espaçadas; "
        "média de 5 treinos)"
    )
    fig.tight_layout()
    name = "stf_modelo_band_strength.png"
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)
    shutil.copy(FIG / name, OVERLEAF / name)


def laplace_figure() -> dict:  # noqa: PLR0914 - linear on purpose
    model, pp, _ = load(SCENARIO, SEED)
    cfg = config(VARIANT, SCENARIO, SEED)
    _, windows, _, _ = _windows(cfg, load_splits())
    x = torch.as_tensor(windows["train"][0], dtype=torch.float32)
    y = torch.as_tensor(windows["train"][1][..., :3], dtype=torch.float32)
    _, sigma = laplace_linear_posterior(model, x, y)
    span = np.asarray(pp.scaler.data_range_)
    to_phys = span[TARGET_IDX][:, None] / span[None, : model.n_nodes]

    def gain(i: int, j: int):  # noqa: ANN202
        first = 0 if j in INPUT_IDX else 1  # u[k] é conhecida, y[k] não

        def coef(m: torch.nn.Module) -> torch.Tensor:
            h = impulse_response_graph(m, i, j)
            return h[first : LAST_LAG + 1].sum() * float(to_phys[i, j])

        return coef

    rows, z = [], np.zeros((len(OUTPUTS), model.n_nodes))
    for i, out in enumerate(OUTPUTS):
        for j in range(model.n_nodes):
            mean, sd = coefficient_ci(model, sigma, gain(i, j))
            z[i, j] = mean / sd if sd > 0 else 0.0
            rows.append(
                {
                    "saida": out,
                    "emissor": SENDERS[j],
                    "ganho": mean,
                    "dp": sd,
                    "z": z[i, j],
                }
            )
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(
        np.clip(np.abs(z), 0, Z_MAX), cmap="viridis", vmin=0, vmax=Z_MAX
    )
    for (r, c), v in np.ndenumerate(z):
        ax.text(
            c,
            r,
            f"{v:+.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if abs(v) < Z_LIGHT else "black",
        )
    ax.set_xticks(
        range(model.n_nodes),
        SENDERS[: model.n_nodes],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(OUTPUTS)), OUTPUTS, fontsize=8)
    fig.colorbar(im, ax=ax, label=f"|z| (limitado a {Z_MAX})")
    ax.set_title(
        "Modelo final (full): significância do ganho de cada "
        "aresta nas 3 primeiras horas (Laplace)",
        fontsize=10,
    )
    fig.tight_layout()
    name = "stf_modelo_laplace_z.png"
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)
    shutil.copy(FIG / name, OVERLEAF / name)
    return {"arestas": rows, "significativas": int((np.abs(z) > Z_SIG).sum())}


if __name__ == "__main__":
    band_strength_figure()
    out = laplace_figure()
    SUMMARY.write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=float)
    )
    print("arestas com |z| > 3:", out["significativas"])
    for r in out["arestas"]:
        if abs(r["z"]) > Z_SIG:
            print(
                f"  {r['emissor']:20s} -> {r['saida']:20s} "
                f"ganho {r['ganho']:+.5f}  z {r['z']:+.1f}"
            )
