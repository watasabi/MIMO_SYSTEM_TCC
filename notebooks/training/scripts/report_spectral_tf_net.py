"""Tabelas (R², RMSE, SMAPE, espectro) e gráficos do SpectralTFNet.

Compara com o ARMAX do artigo nas mesmas janelas de teste.

Uso:
    uv run --extra ml python \
        notebooks/training/scripts/report_spectral_tf_net.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mimo_sys.training.data import (
    DATA_PROCESSED,
    INPUT_COLS,
    OUTPUT_COLS,
    PROJECT_ROOT,
)
from mimo_sys.training.evaluation import amplitude_spectrum

PREDS = PROJECT_ROOT / "outputs" / "spectral_tf_net_preds"
RESULTS = PROJECT_ROOT / "outputs" / "spectral_tf_net_results.jsonl"
FIG = PROJECT_ROOT / "docs" / "figures"
SEEDS = (42, 43, 44)
NAMES = ["vazão", "nível", "pressão"]
OUR_TEST_START = 1876  # first test row of the 70/15/15 split
PAPER_TEST = (1499, 2000)  # rows of the paper's validation window

# ------------------------------------------------------------- metrics


def time_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """R², RMSE, SMAPE (%) per output."""
    err = y - p
    return {
        "r2": 1 - (err**2).sum(0) / ((y - y.mean(0)) ** 2).sum(0),
        "rmse": np.sqrt((err**2).mean(0)),
        "smape": 100 * np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(p)), 0),
    }


def spectral_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """R² and SMAPE (%) between amplitude spectra, DC dropped."""
    a, b = amplitude_spectrum(y)[1:], amplitude_spectrum(p)[1:]
    return {
        "r2_esp": 1 - ((a - b) ** 2).sum(0) / ((a - a.mean(0)) ** 2).sum(0),
        "smape_esp": 100 * np.mean(2 * np.abs(a - b) / (a + b + 1e-12), 0),
    }


def all_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    return {**time_metrics(y, p), **spectral_metrics(y, p)}


def our_model(prefix: str) -> dict[str, np.ndarray]:
    """Mean over seeds of every metric, for one saved config."""
    runs = []
    for seed in SEEDS:
        d = np.load(PREDS / f"{prefix}_s{seed}.npz")
        runs.append(all_metrics(d["y_true"], d["y_pred"]))
    return {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}


# --------------------------------------------- published ARMAX (Eq. 15)

full = pd.concat(
    [
        pd.read_parquet(DATA_PROCESSED / f"mimo_full_sample_{n}.parquet")
        for n in ("train", "val", "test")
    ],
    ignore_index=True,
)
U = full[INPUT_COLS].to_numpy().T
Y = full[OUTPUT_COLS].to_numpy().T
P = pd.read_csv(
    PROJECT_ROOT / "data" / "external" / "tabela3_parametros_armax_papper.csv"
)["MOGWO"].to_numpy()


def armax_base(t: int, out: int, y: np.ndarray) -> float:
    """AR + exogenous part of output `out` at row `t` (lags 1-3)."""
    base = out * 24
    others = [o for o in range(3) if o != out]
    b_u = P[base + 9 : base + 21].reshape(4, 3)
    s = 0.0
    for k in range(3):
        s -= P[base + k] * y[out, t - 1 - k]
        s -= P[base + 3 + k] * y[others[0], t - 1 - k]
        s -= P[base + 6 + k] * y[others[1], t - 1 - k]
        s += b_u[:, k] @ U[:, t - 1 - k]
    return s


def armax_residuals() -> np.ndarray:
    """One-step prediction errors over the whole series (MA terms)."""
    e = np.zeros_like(Y)
    for t in range(3, Y.shape[1]):
        for out in range(3):
            c = P[out * 24 + 21 : out * 24 + 24]
            pred = armax_base(t, out, Y)
            pred += sum(c[k] * e[out, t - 1 - k] for k in range(3))
            e[out, t] = Y[out, t] - pred
    return e


E = armax_residuals()


def armax_one_step(start: int, stop: int) -> np.ndarray:
    """`(stop - start, 3)` one-step predictions, measured lags."""
    return (Y[:, start:stop] - E[:, start:stop]).T


def armax_free_run(t0: int, horizon: int) -> np.ndarray:
    """`(horizon, 3)` free-run from `t0` (no measured y inside)."""
    y_sim, e_sim = Y.copy(), E.copy()
    for t in range(t0, t0 + horizon):
        for out in range(3):
            c = P[out * 24 + 21 : out * 24 + 24]
            pred = armax_base(t, out, y_sim)
            pred += sum(c[k] * e_sim[out, t - 1 - k] for k in range(3))
            y_sim[out, t], e_sim[out, t] = pred, 0.0
    return y_sim[:, t0 : t0 + horizon].T


def armax_horizon(horizon: int, offset: int = 0) -> tuple:
    """Free-run on our test with the same non-overlapping anchors."""
    anchors = range(OUR_TEST_START + offset, Y.shape[1] - horizon + 1, horizon)
    pred = np.concatenate([armax_free_run(t, horizon) for t in anchors])
    first = OUR_TEST_START + offset
    return Y[:, first : first + len(pred)].T, pred


# --------------------------------------------------------------- tables


def row(label: str, m: dict[str, np.ndarray]) -> str:
    cells = []
    for k in range(3):
        cells.append(
            f"{m['r2'][k]:.3f} / {m['rmse'][k]:.3g} / {m['smape'][k]:.1f}%"
        )
    esp = " / ".join(f"{m['r2_esp'][k]:.2f}" for k in range(3))
    esp_s = " / ".join(f"{m['smape_esp'][k]:.0f}%" for k in range(3))
    return (
        f"| {label} | "
        + " | ".join(cells)
        + f" | **{m['r2'].mean():.3f}** | {esp} | {esp_s} |"
    )


HEADER = (
    "| Modelo | vazão: R² / RMSE (l/s) / SMAPE | nível: R² / RMSE (m)"
    " / SMAPE | pressão: R² / RMSE (mca) / SMAPE | R² médio"
    " | R² espectral (v/n/p) | SMAPE espectral (v/n/p) |\n"
    "|---|---|---|---|---|---|---|"
)


def main() -> None:  # noqa: PLR0912, PLR0914, PLR0915 - linear on purpose
    """Print every table, save every figure."""
    out: dict[str, dict] = {}

    # 1. One step, paper's window.
    y_pw = Y[:, PAPER_TEST[0] : PAPER_TEST[1]].T
    arm_pw = all_metrics(y_pw, armax_one_step(*PAPER_TEST))
    out["1passo_recorte"] = {
        "ARMAX MOGWO (recalculado)": arm_pw,
        "SpectralTFNet linear": our_model("stf_paper_recorte_w3_h1_lin"),
        "SpectralTFNet + resíduo": our_model("stf_paper_recorte_w3_h1_res"),
    }
    # 2. One step, our test.
    y_ot = Y[:, OUR_TEST_START:].T
    out["1passo_nosso_teste"] = {
        "ARMAX MOGWO": all_metrics(
            y_ot, armax_one_step(OUR_TEST_START, Y.shape[1])
        ),
        "SpectralTFNet linear": our_model("stf_paper_w3_h1_lin"),
        "SpectralTFNet + resíduo": our_model("stf_paper_w3_h1_res"),
    }
    # 3-4. Trained with H=6 and H=12, our test.
    for h in (6, 12):
        out[f"h{h}"] = {
            "ARMAX MOGWO (free-run)": all_metrics(*armax_horizon(h)),
            "SpectralTFNet linear": our_model(f"stf_h{h}_lin_w168_b128"),
            "SpectralTFNet + resíduo": our_model(f"stf_h{h}_res_w168_b128"),
        }
    for name, block in out.items():
        print(f"\n### {name}\n\n{HEADER}")
        for label, m in block.items():
            print(row(label, m))

    # ---------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 10})

    # (a) One step vs paper, on the paper's window.
    table2 = {
        "ARMAX MOGWO\n(Tab. 2 do paper)": [0.9478, 0.9526, 0.8364],
        "ARMAX MOCS\n(Tab. 2 do paper)": [0.9394, 0.9596, 0.8633],
    }
    bars = {
        **table2,
        "SpectralTFNet\nlinear": out["1passo_recorte"]["SpectralTFNet linear"][
            "r2"
        ],
        "SpectralTFNet\n+ resíduo": out["1passo_recorte"][
            "SpectralTFNet + resíduo"
        ]["r2"],
    }
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.2
    for k, (label, vals) in enumerate(bars.items()):
        x = np.arange(3) + (k - 1.5) * width
        ax.bar(x, vals, width, label=label)
        for xi, v in zip(x, vals, strict=True):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(3), NAMES)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("R²")
    ax.set_title("1 passo à frente, recorte do paper (teste 1500–2000)")
    ax.legend(
        fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1)
    )
    fig.tight_layout()
    fig.savefig(FIG / "stf_1passo_vs_paper.png", dpi=130)

    # (b) R² per horizon step, all offsets.
    rows = {
        json.loads(line)["run"]: json.loads(line) for line in RESULTS.open()
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    for run, label, style in (
        ("stf_perstep_h12_res_w168_b128_s42", "treinado H=12 + resíduo", "-o"),
        ("stf_perstep_h12_lin_w168_b128_s42", "treinado H=12 linear", "--o"),
        ("stf_perstep_h6_res_w168_b128_s42", "treinado H=6 + resíduo", "-s"),
        ("stf_perstep_h6_lin_w168_b128_s42", "treinado H=6 linear", "--s"),
    ):
        p = rows[run]["por_passo_todos_offsets"]
        r2 = np.mean(
            [p[f"r2_{o}"] for o in ("vazao", "nivel_cascata", "pressao")], 0
        )
        ax.plot(range(1, len(r2) + 1), r2, style, label=label, ms=4)
    arm = []
    for h in range(12):
        sq, ys = [], []
        for off in range(12):
            yt, yp = armax_horizon(12, off)
            yt, yp = yt.reshape(-1, 12, 3), yp.reshape(-1, 12, 3)
            sq.append((yt[:, h] - yp[:, h]) ** 2)
            ys.append(yt.reshape(-1, 3))
        mse = np.concatenate(sq).mean(0)
        arm.append(np.mean(1 - mse / np.concatenate(ys).var(0)))
    ax.plot(range(1, 13), arm, ":^", color="gray", label="ARMAX do paper")
    ax.axhline(0.85, color="red", lw=0.8, ls=":", label="meta 0,85")
    ax.set_xlabel("passo do horizonte (h)")
    ax.set_ylabel("R² médio (vazão, nível, pressão)")
    ax.set_title("Qualidade por passo do horizonte (nosso teste)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stf_por_horizonte.png", dpi=130)

    # (c) Series and (d) spectra, + residual, seed 42, H=6 and H=12.
    for h in (6, 12):
        d = np.load(PREDS / f"stf_h{h}_res_w168_b128_s42.npz")
        yt, yp = d["y_true"], d["y_pred"]
        units = ["l/s", "m", "mca"]
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for k, ax in enumerate(axes):
            ax.plot(yt[:, k], color="black", lw=1, label="medido")
            ax.plot(yp[:, k], color="tab:red", lw=1, label=f"previsto ({h} h)")
            for start in range(0, len(yt), h):
                ax.axvline(start, color="#dddddd", lw=0.5, zorder=0)
            ax.set_ylabel(f"{NAMES[k]} ({units[k]})")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel(
            "hora do período de teste (linhas cinza = nova janela)"
        )
        fig.suptitle(f"SpectralTFNet + resíduo, previsão encadeada de {h} h")
        fig.tight_layout()
        fig.savefig(FIG / f"stf_serie_h{h}.png", dpi=130)

        freqs = np.fft.rfftfreq(len(yt))[1:]
        period = 1 / freqs
        a_t, a_p = amplitude_spectrum(yt)[1:], amplitude_spectrum(yp)[1:]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for k, ax in enumerate(axes):
            ax.loglog(period, a_t[:, k], color="black", lw=1, label="medido")
            ax.loglog(
                period, a_p[:, k], color="tab:red", lw=1, label="previsto"
            )
            ax.axvline(24, color="gray", ls=":", lw=0.8)
            ax.axvline(12, color="gray", ls=":", lw=0.8)
            ax.set_title(NAMES[k])
            ax.set_xlabel("período (h)")
            ax.invert_xaxis()
        axes[0].set_ylabel("amplitude")
        axes[0].legend(fontsize=8)
        fig.suptitle("Espectro de amplitude, teste (linhas: 24 h e 12 h)")
        fig.tight_layout()
        fig.savefig(FIG / f"stf_espectro_h{h}.png", dpi=130)

    # (e) Ablations.
    abl = {
        "completo\n(linear)": "stf_abl_full_w168_b128",
        "+ resíduo": "stf_abl_residual_w168_b128",
        "− gate de\nregime": "stf_abl_no_regime_w168_b128",
        "− loss da\ncascata": "stf_abl_no_cascade_loss_w168_b128",
        "− entradas\nfuturas": "stf_abl_no_future_w168_b128",
    }
    vals = [
        np.mean([rows[f"{v}_s{s}"]["r2_medio_cascata"] for s in SEEDS])
        for v in abl.values()
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = ["tab:blue", "tab:green", "tab:gray", "tab:orange", "tab:red"]
    ax.bar(list(abl), vals, color=colors)
    for k, v in enumerate(vals):
        ax.text(k, v + 0.005, f"{v:.3f}", ha="center")
    ax.axhline(0.819, color="black", ls=":", lw=0.8)
    ax.text(4.4, 0.822, "ensemble de 5 (0,819)", ha="right", fontsize=8)
    ax.set_ylim(0.6, 0.86)
    ax.set_ylabel("R² médio, 12 passos")
    ax.set_title("O que cada peça vale (3 seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "stf_ablacoes.png", dpi=130)
    print("\nfiguras salvas em", FIG)


if __name__ == "__main__":
    main()
