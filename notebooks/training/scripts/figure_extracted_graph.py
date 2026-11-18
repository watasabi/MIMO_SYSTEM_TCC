"""Grafo por banda e significância das arestas do modelo final.

Duas figuras do modelo final do artigo (cenário `full`):

1. `stf_modelo_band_strength`: força `|c_ij,m|` de cada aresta em cada
   banda, média dos 5 treinos, um painel por saída;
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

import matplotlib.pyplot as plt
import numpy as np
import torch
from figure_style import (
    OUTPUT_NAMES,
    WIDTH,
    apply_style,
    panel_letter,
    save,
)
from report_extracted_model import (
    LABEL,
    OUTPUTS,
    SENDERS,
    VARIANT,
    band_periods,
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
    cfg = config(VARIANT, SCENARIO, SEED)
    periods = band_periods(
        2 * (cfg.input_window + cfg.output_window), strength.shape[-1]
    )
    names = [LABEL[s] for s in SENDERS[: strength.shape[1]]]
    ticks = ["DC" if np.isinf(p) else f"{p:.0f}" for p in periods]
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 7.2), sharex=True)
    vmax = strength.max()
    for k, ax in enumerate(axes):
        im = ax.imshow(
            strength[k], vmin=0, vmax=vmax, cmap="viridis", aspect="auto"
        )
        ax.set_yticks(range(len(names)), names)
        panel_letter(ax, k)
    axes[-1].set_xticks(range(len(ticks)), ticks)
    axes[-1].set_xlabel("Período central da banda (h)")
    fig.tight_layout()
    fig.colorbar(im, ax=axes, label=r"$|c_{ij,m}|$", shrink=0.6)
    save(fig, "stf_modelo_band_strength.pdf")


def laplace_figure() -> dict:  # noqa: PLR0914 - linear on purpose
    model, pp, _ = load(SCENARIO, SEED)
    cfg = config(VARIANT, SCENARIO, SEED)
    _, windows, _, _ = _windows(cfg, load_splits(cfg.data_prefix))
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
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    im = ax.imshow(
        np.clip(np.abs(z), 0, Z_MAX),
        cmap="viridis",
        vmin=0,
        vmax=Z_MAX,
        aspect="auto",
    )
    for (r, c), v in np.ndenumerate(z):
        ax.text(
            c,
            r,
            f"{v:+.1f}".replace(".", ","),
            ha="center",
            va="center",
            color="white" if abs(v) < Z_LIGHT else "black",
        )
    names = [LABEL[s] for s in SENDERS[: model.n_nodes]]
    ax.set_xticks(range(model.n_nodes), names, rotation=35, ha="right")
    ax.set_yticks(range(len(OUTPUTS)), OUTPUT_NAMES)
    fig.colorbar(im, ax=ax, label=f"$|z|$ (limitado a {Z_MAX})")
    fig.tight_layout()
    save(fig, "stf_modelo_laplace_z.pdf")
    return {"arestas": rows, "significativas": int((np.abs(z) > Z_SIG).sum())}


if __name__ == "__main__":
    apply_style()
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
