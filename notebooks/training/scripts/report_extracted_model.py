"""O modelo final extraído: equação, força de cada aresta e grafo.

Para cada seed do modelo final (`F5_final`), lê o `H_ij(f)` da
parte linear e mede, nas janelas de teste:

- a **contribuição** de cada aresta j → i, `Σ_τ h_ij(τ) s̃_j(t − τ)`
  nas 12 posições do horizonte (sequência medida, sem realimentação),
  em unidades físicas da saída; a força da aresta é o desvio padrão
  dessa contribuição, e o sinal é o do ganho do kernel (soma de
  `h_ij(τ)` em τ = −12..24 h, o futuro programado incluído);
- os coeficientes `h_ij(τ)` dos primeiros atrasos, em unidades físicas;
- o desvio padrão do resíduo não linear, para comparar.

Figuras: o grafo aprendido (espessura = força, cor = sinal) e as respostas
ao impulso das arestas mais fortes. O resumo traz também os coeficientes
complexos `c_ij,m` dessas arestas (módulo e fase), em unidades físicas.

Uso (da raiz do repo):
    uv run --extra ml python \
        notebooks/training/scripts/report_extracted_model.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from figure_style import COLORS, WIDTH, apply_style, save
from matplotlib.patches import FancyArrowPatch
from run_improvements import (
    FINAL_VARIANT,
    OUT,
    SEEDS,
    config,
    load_weights,
)

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

VARIANT = FINAL_VARIANT
SUMMARY = PROJECT_ROOT / "outputs" / "extracted_model.json"
H = 12
SENDERS = [*FEATURE_COLS, "balanco_massa"]
LABEL = {
    "freq_b1": "Bomba 1",
    "freq_b2": "Bomba 2",
    "freq_b3": "Bomba 3",
    "vazao_entrada": "Vazão de entrada",
    "vazao_distribuicao": "Vazão",
    "nivel_res": "Nível",
    "pressao": "Pressão",
    "balanco_massa": r"$\Sigma(u_4-y_1)$",
}
SENDER_UNITS = ["Hz", "Hz", "Hz", "l/s", "l/s", "m", "mca", "l/s"]
OUTPUTS = ["vazao_distribuicao", "nivel_res", "pressao"]
UNITS = ["l/s", "m", "mca"]
MIN_SHARE = 0.05  # arestas abaixo de 5% da saída não entram no grafo


def load(scenario: str, seed: int):  # noqa: ANN201
    c = config(VARIANT, scenario, seed)
    layout = target_layout(c)
    _, windows, _, _ = _windows(c, load_splits(c.data_prefix))
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


def analyse(scenario: str, seed: int) -> dict:  # noqa: PLR0914
    m, pp, (x, _) = load(scenario, seed)
    W = config(VARIANT, scenario, seed).input_window
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
        # Coeficientes das bandas, em unidades físicas.
        "bands": torch.view_as_complex(
            m.response[0].detach().contiguous()
        ).numpy()
        * to_phys[..., None],
    }


def band_periods(n_fft: int, n_bands: int) -> np.ndarray:
    """Período (h) do centro de cada banda (`_band_interpolation`)."""
    n_bins = n_fft // 2 + 1
    centres = np.concatenate(
        [[0.0], np.logspace(0, np.log10(n_bins - 1), n_bands - 1)]
    )
    centres[-1] = n_bins - 1
    with np.errstate(divide="ignore"):
        return n_fft / centres


def graph_figure(s: np.ndarray, sign: np.ndarray, resid: np.ndarray) -> None:
    """Grafo aprendido: emissores à esquerda, saídas à direita."""
    fig, ax = plt.subplots(figsize=(WIDTH, 4.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ys = np.linspace(8.3, 0.5, len(SENDERS))
    pos_s = {j: (2.9, ys[j]) for j in range(len(SENDERS))}
    pos_t = {0: (9.2, 7.0), 1: (9.2, 4.4), 2: (9.2, 1.8)}
    share = s / (s.sum(1, keepdims=True) + resid[:, None])
    for i in range(3):
        for j in range(len(SENDERS)):
            if share[i, j] < MIN_SHARE:
                continue
            color = COLORS[2] if sign[i, j] > 0 else COLORS[1]
            ax.add_patch(
                FancyArrowPatch(
                    pos_s[j],
                    pos_t[i],
                    arrowstyle="-|>",
                    mutation_scale=9,
                    lw=0.5 + 9 * share[i, j],
                    color=color,
                    alpha=0.75,
                    shrinkA=6,
                    shrinkB=12,
                    connectionstyle="arc3,rad=0.04",
                )
            )
            frac = (0.68, 0.52, 0.36)[i]
            ax.text(
                frac * pos_t[i][0] + (1 - frac) * pos_s[j][0],
                frac * pos_t[i][1] + (1 - frac) * pos_s[j][1],
                f"{100 * share[i, j]:.0f}%",
                color=color,
                ha="center",
                va="center",
                bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85),
            )
    for j, name in enumerate(SENDERS):
        ax.add_patch(
            plt.Circle(pos_s[j], 0.14, fc="0.25", ec="none", zorder=5)
        )
        ax.text(
            pos_s[j][0] - 0.3,
            pos_s[j][1],
            LABEL[name],
            ha="right",
            va="center",
        )
    for i, name in enumerate(OUTPUTS):
        ax.add_patch(plt.Circle(pos_t[i], 0.2, fc="0.25", ec="none", zorder=5))
        rs = 100 * resid[i] / (s[i].sum() + resid[i])
        ax.text(
            pos_t[i][0] + 0.4,
            pos_t[i][1],
            f"{LABEL[name]}\nresíduo {rs:.0f}%",
            va="center",
        )
    save(fig, "stf_grafo_extraido_full.pdf")


def kernel_figure(
    lags: np.ndarray,
    lags_sd: np.ndarray,
    ahead: np.ndarray,
    top: list[tuple[int, int]],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.4), sharex=True)
    for ax, (i, j) in zip(axes.flat, top, strict=False):
        tau = np.arange(-H, 25)
        h = np.concatenate([ahead[i, j][::-1], lags[i, j]])
        sd = np.concatenate([np.zeros(H), lags_sd[i, j]])
        ax.axvspan(-H - 0.5, -0.5, color="0.93")
        ax.bar(tau, h, color=COLORS[2], width=0.7)
        ax.errorbar(
            tau[H:], h[H:], yerr=sd[H:], fmt="none", ecolor="0.3", lw=0.6
        )
        ax.axhline(0, color="0.5", lw=0.6)
        sender = LABEL[SENDERS[j]]
        ax.set_title(f"{sender} → {LABEL[OUTPUTS[i]].lower()}", loc="left")
        ax.set_ylabel(f"{UNITS[i]} por {SENDER_UNITS[j]}")
    for ax in axes[-1]:
        ax.set_xlabel(r"Atraso $\tau$ (h)")
    fig.align_ylabels(axes)
    fig.tight_layout()
    save(fig, "stf_kernels_extraidos.pdf")


def main() -> None:  # noqa: PLR0914 - linear on purpose
    apply_style()
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
        if scenario == "full":
            graph_figure(s_mean, sign, resid)
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
            bands = np.stack([r["bands"] for r in runs])
            cfg = config(VARIANT, scenario, SEEDS[0])
            periods = band_periods(
                2 * (cfg.input_window + cfg.output_window), bands.shape[-1]
            )
            summary["bandas_periodo_h"] = periods.round(1).tolist()
            summary["filtros_full"] = [
                {
                    "aresta": f"{SENDERS[j]} -> {OUTPUTS[i]}",
                    "modulo": np.abs(bands.mean(0)[i, j]).round(5).tolist(),
                    "modulo_desvio": np.abs(bands)[:, i, j]
                    .std(0)
                    .round(5)
                    .tolist(),
                    "fase_graus": np.degrees(np.angle(bands.mean(0)[i, j]))
                    .round(0)
                    .tolist(),
                }
                for i, j in top
            ]
            print("períodos das bandas (h):", summary["bandas_periodo_h"])
            for f in summary["filtros_full"]:
                print(f)
        print(f"\n### {scenario}")
        for r in rows:
            if r["parcela"] >= MIN_SHARE or r["emissor"] == "residuo":
                print(r)
    SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
