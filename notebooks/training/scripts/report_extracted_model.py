"""O modelo final extraído: equação, força de cada aresta e grafo.

Para cada seed do modelo final (`I0_ref`, doc 13), lê o `H_ij(f)` da
parte linear e mede, nas janelas de teste:

- a **contribuição** de cada aresta j → i, `Σ_τ h_ij(τ) s̃_j(t − τ)`
  nas 12 posições do horizonte (sequência medida, sem realimentação),
  em unidades físicas da saída; a força da aresta é o desvio padrão
  dessa contribuição, e o sinal é o do ganho do kernel (soma de
  `h_ij(τ)` em τ = −12..24 h, o futuro programado incluído);
- os coeficientes `h_ij(τ)` dos primeiros atrasos, em unidades físicas;
- o desvio padrão do resíduo não linear, para comparar.

Figuras: o grafo aprendido (espessura = força, cor = sinal) e as
respostas ao impulso das arestas mais fortes.

Uso (da raiz do repo):
    uv run --extra ml python \
        notebooks/training/scripts/report_extracted_model.py
"""

from __future__ import annotations

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
from run_improvements import OUT, SEEDS, config, load_weights

from mimo_sys.training.data import (
    FEATURE_COLS,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.train import (
    _windows,  # noqa: PLC2701
    build_model,
    known_columns,
    target_layout,
)

VARIANT = "I0_ref"
FIG = PROJECT_ROOT / "docs" / "figures"
OVERLEAF = PROJECT_ROOT / "overleaf" / "imagens"
SUMMARY = PROJECT_ROOT / "outputs" / "extracted_model.json"
W, H = 168, 12
SENDERS = [*FEATURE_COLS, "balanco_massa"]
LABEL = {
    "freq_b1": "bomba 1",
    "freq_b2": "bomba 2",
    "freq_b3": "bomba 3",
    "vazao_entrada": "vazão\nentrada",
    "vazao_distribuicao": "vazão\ndistr.",
    "nivel_res": "nível",
    "pressao": "pressão",
    "balanco_massa": "Σ(u₄−y₁)",
}
OUTPUTS = ["vazao_distribuicao", "nivel_res", "pressao"]
UNITS = ["l/s", "m", "mca"]
MIN_SHARE = 0.05  # arestas abaixo de 5% da saída não entram no grafo


def load(scenario: str, seed: int):  # noqa: ANN201
    c = config(VARIANT, scenario, seed)
    layout = target_layout(c)
    _, windows, _, _ = _windows(c, load_splits())
    n_nodes = windows["test"][0].shape[2]
    kwargs = dict(c.model_kwargs)
    kwargs.setdefault("known_future_idx", tuple(known_columns(c)))
    m = build_model(
        c.model,
        c.input_window + c.output_window,
        c.output_window,
        n_nodes,
        target_idx=list(layout.model_cols),
        **kwargs,
    )
    m.load_state_dict(load_weights(c.run_name))
    pp = torch.load(
        OUT / "weights" / f"{c.run_name}_pp.pt", weights_only=False
    )
    return m.eval(), pp, windows["test"]


def kernels(m: torch.nn.Module) -> np.ndarray:
    """`(i, j, n_fft)` impulse responses; τ ≤ 0 zeroed for unknown senders.

    In the recursion, a sender whose future is unknown is zero from the
    current position on, so only its strictly past lags (τ ≥ 1) act.
    """
    with torch.no_grad():
        kern = torch.stack(
            [
                torch.stack(
                    [m.impulse_response(i, j) for j in range(m.n_nodes)]
                )
                for i in range(3)
            ]
        ).numpy()
    unknown = m.future_mask.numpy() == 0
    kern[:, unknown, 0] = 0.0
    kern[:, unknown, m.n_fft // 2 + 1 :] = 0.0
    return kern


def analyse(scenario: str, seed: int) -> dict:
    m, pp, (x, _) = load(scenario, seed)
    span = np.asarray(pp.scaler.data_range_)
    out_span = span[TARGET_IDX]
    to_phys = out_span[:, None] / span[None, : m.n_nodes]
    kern = kernels(m)
    xt = torch.as_tensor(x, dtype=torch.float32)
    seq = (xt - xt[:, :W].mean(1, keepdim=True)).numpy()
    spec = np.fft.rfft(seq, n=m.n_fft, axis=1)  # (b, f, j)
    resp = np.fft.rfft(kern, n=m.n_fft, axis=2)  # (i, j, f)
    contrib = np.fft.irfft(
        np.einsum("ijf,bfj->bijf", resp, spec), n=m.n_fft, axis=3
    )[..., W : W + H]  # (b, i, j, H)
    with torch.no_grad():
        m(xt)
    lags = kern[..., :25] * to_phys[..., None]
    ahead = kern[..., -12:][..., ::-1] * to_phys[..., None]  # τ = −1..−12
    return {
        "strength": (contrib * out_span[None, :, None, None]).std(axis=(0, 3)),
        # Doc 8's reading: sum of the first lags (τ = 0..3; τ = 0 is
        # already zero for unknown senders).
        "gain": lags[..., :4].sum(-1),
        "resid_std": m._last_residual.numpy().std(axis=(0, 1)) * out_span,
        "lags": lags,
        "ahead": ahead,
    }


def graph_figure(
    s: np.ndarray, sign: np.ndarray, resid: np.ndarray, scenario: str
) -> None:
    """Grafo aprendido: emissores à esquerda, saídas à direita."""
    fig, ax = plt.subplots(figsize=(12, 8.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.4, 9.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ys = np.linspace(8, 0.4, len(SENDERS))
    pos_s = {j: (1.4, ys[j]) for j in range(len(SENDERS))}
    pos_t = {0: (9.6, 6.9), 1: (9.6, 4.2), 2: (9.6, 1.5)}
    share = s / (s.sum(1, keepdims=True) + resid[:, None])
    for i in range(3):
        for j in range(len(SENDERS)):
            if share[i, j] < MIN_SHARE:
                continue
            color = "#b85450" if sign[i, j] > 0 else "#6c8ebf"
            ax.add_patch(
                FancyArrowPatch(
                    pos_s[j],
                    pos_t[i],
                    arrowstyle="-|>",
                    mutation_scale=14,
                    lw=0.8 + 20 * share[i, j],
                    color=color,
                    alpha=0.8,
                    shrinkA=24,
                    shrinkB=34,
                    connectionstyle="arc3,rad=0.04",
                )
            )
            # rótulos de cada saída em x diferente
            frac = (0.62, 0.48, 0.34)[i]
            mx = frac * pos_t[i][0] + (1 - frac) * pos_s[j][0]
            my = frac * pos_t[i][1] + (1 - frac) * pos_s[j][1]
            ax.text(
                mx,
                my,
                f"{100 * share[i, j]:.0f}%",
                fontsize=8,
                color=color,
                ha="center",
                va="center",
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.8),
            )
    for j, name in enumerate(SENDERS):
        fc = (
            "#e1d5e7"
            if name in OUTPUTS
            else ("#fff2cc" if name == "balanco_massa" else "#f8cecc")
        )
        ax.add_patch(
            plt.Circle(pos_s[j], 0.42, fc=fc, ec="#333", lw=1.2, zorder=5)
        )
        ax.text(
            *pos_s[j],
            LABEL[name],
            ha="center",
            va="center",
            fontsize=8,
            zorder=6,
        )
    for i, name in enumerate(OUTPUTS):
        ax.add_patch(
            plt.Circle(
                pos_t[i], 0.62, fc="#e1d5e7", ec="#333", lw=1.6, zorder=5
            )
        )
        ax.text(
            *pos_t[i],
            LABEL[name] + "\n(t+k)",
            ha="center",
            va="center",
            fontsize=9,
            zorder=6,
        )
        rs = 100 * resid[i] / (s[i].sum() + resid[i])
        ax.text(
            pos_t[i][0] + 0.85,
            pos_t[i][1],
            f"+ resíduo\n{rs:.0f}%",
            fontsize=8,
            va="center",
            color="#555",
        )
    ax.text(
        1.4,
        8.6,
        "emissores (passado + futuro conhecido)",
        ha="center",
        fontsize=10,
        weight="bold",
    )
    ax.text(9.6, 8.0, "saídas", ha="center", fontsize=10, weight="bold")
    ax.text(
        5.6,
        -0.3,
        "espessura e % = parcela da variação da saída explicada pela "
        "aresta (desvio da contribuição)  ·  vermelho: ganho positivo nas "
        "3 primeiras horas  ·  azul: negativo  ·  arestas < 5% omitidas",
        ha="center",
        fontsize=8,
        color="#444",
    )
    ax.set_title(
        f"Modelo final extraído ({scenario}): grafo de "
        "interações aprendido (média de 5 treinos)",
        fontsize=12,
    )
    name = f"stf_grafo_extraido_{scenario}"
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    shutil.copy(FIG / f"{name}.pdf", OVERLEAF / f"{name}.pdf")


def kernel_figure(
    lags: np.ndarray,
    lags_sd: np.ndarray,
    ahead: np.ndarray,
    top: list[tuple[int, int]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (i, j) in zip(axes.flat, top, strict=False):
        tau = np.arange(-12, 25)
        h = np.concatenate([ahead[i, j][::-1], lags[i, j]])
        sd = np.concatenate([np.zeros(12), lags_sd[i, j]])
        ax.axvspan(-12.5, -0.5, color="#f3f3f3")
        ax.bar(tau, h, color="tab:red", width=0.7)
        ax.errorbar(
            tau[12:], h[12:], yerr=sd[12:], fmt="none", ecolor="#555", lw=0.7
        )
        ax.axhline(0, color="#999", lw=0.6)
        unit = f"{UNITS[i]} por unid. de {LABEL[SENDERS[j]]}".replace(
            "\n", " "
        )
        ax.set_title(
            f"{LABEL[SENDERS[j]]} → {LABEL[OUTPUTS[i]]}".replace("\n", " "),
            fontsize=10,
        )
        ax.set_xlabel("atraso τ (h)  ·  τ < 0: futuro programado")
        ax.set_ylabel(unit, fontsize=7)
    fig.suptitle(
        "Respostas ao impulso h_ij(τ) das arestas mais fortes "
        "(média de 5 treinos; barras de erro: desvio entre "
        "treinos)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "stf_kernels_extraidos.png", dpi=140)
    plt.close(fig)
    shutil.copy(
        FIG / "stf_kernels_extraidos.png",
        OVERLEAF / "stf_kernels_extraidos.png",
    )


def main() -> None:
    summary = {}
    for scenario in ("full", "pumps"):
        runs = [analyse(scenario, s) for s in SEEDS]
        st = np.stack([r["strength"] for r in runs])
        gain = np.stack([r["gain"] for r in runs]).mean(0)
        sign = np.sign(gain)
        resid = np.mean([r["resid_std"] for r in runs], 0)
        lags = np.stack([r["lags"] for r in runs])
        ahead = np.stack([r["ahead"] for r in runs]).mean(0)
        s_mean = st.mean(0)
        graph_figure(s_mean, sign, resid, scenario)
        share = s_mean / (s_mean.sum(1, keepdims=True) + resid[:, None])
        rows = []
        for i, out in enumerate(OUTPUTS):
            order = np.argsort(-s_mean[i])
            for j in order:
                rows.append(
                    {
                        "saida": out,
                        "emissor": SENDERS[j],
                        "parcela": round(float(share[i, j]), 3),
                        "forca": round(float(s_mean[i, j]), 4),
                        "forca_cv_seeds": round(
                            float(st[:, i, j].std() / (s_mean[i, j] + 1e-12)),
                            2,
                        ),
                        "ganho": round(float(gain[i, j]), 5),
                        "h_tau_0_3": lags.mean(0)[i, j, :4].round(5).tolist(),
                        "h_tau_0_3_desvio": lags.std(0)[i, j, :4]
                        .round(5)
                        .tolist(),
                        "h_futuro_1_3": ahead[i, j, :3].round(5).tolist(),
                    }
                )
            rows.append(
                {
                    "saida": out,
                    "emissor": "residuo",
                    "parcela": round(
                        float(resid[i] / (s_mean[i].sum() + resid[i])), 3
                    ),
                    "forca": round(float(resid[i]), 4),
                }
            )
        summary[scenario] = rows
        if scenario == "full":
            flat = [(i, j) for i in range(3) for j in range(len(SENDERS))]
            top = sorted(flat, key=lambda ij: -share[ij])[:6]
            kernel_figure(lags.mean(0), lags.std(0), ahead, top)
        print(f"\n### {scenario}")
        for r in rows:
            if r["parcela"] >= MIN_SHARE or r["emissor"] == "residuo":
                print(r)
    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
