"""Desempenho por frequência do modelo final (I0_ref, cenário `full`).

Usa as mesmas previsões da figura da série (`report_lean_model.py`): a
simulação livre de 12 h encadeada no teste, sem retreinar nada.

1. `stf_modelo_espectro_full`: densidade espectral (Welch) do medido e do
   previsto, seed 42 (a mesma da figura da série);
2. `stf_modelo_r2_banda_full`: R² em cada banda de período, média e
   desvio dos 5 treinos. A banda é recortada na FFT; o erro de cada banda
   soma o erro total (Parseval), então o R² global é a média dos R² das
   bandas ponderada pela variância de cada banda. O viés (DC do erro)
   entra na banda mais lenta.

Uso (da raiz do repo):
    uv run --extra ml python \
        notebooks/training/scripts/figure_spectral_performance.py
"""

from __future__ import annotations

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from report_lean_model import FIG_SEED, H, seed_preds
from report_spectral_tf_net import FIG, NAMES
from run_improvements import SEEDS
from scipy.signal import welch

from mimo_sys.training.data import PROJECT_ROOT

OVERLEAF = PROJECT_ROOT.parent / "TCC_MIMO_SYS_overleaf" / "imagens"
SUMMARY = PROJECT_ROOT / "outputs" / "spectral_performance.json"
SCENARIO = "full"
UNITS = ["l/s", "m", "mca"]
NYQUIST = 2  # período mínimo (horas) com amostragem horária
# Bandas de período (horas), as mesmas da decomposição do artigo.
BANDS = [(NYQUIST, 4), (4, 8), (8, 16), (16, 32), (32, np.inf)]
BAND_LABELS = ["2–4", "4–8", "8–16", "16–32", ">32"]
COLORS = ["tab:blue", "tab:green", "tab:purple"]
COMMA = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def chained(a: np.ndarray) -> np.ndarray:
    """Free-run of 12 h with non-overlapping origins, as one series."""
    return a[::H].reshape(-1, 3)


def band_r2(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R² and variance share of each band, per output: (bands, 3)."""
    n = len(t)
    period = np.full(n // 2 + 1, np.inf)
    period[1:] = n / np.arange(1, n // 2 + 1)
    yf = np.fft.rfft(t - t.mean(0), axis=0)
    ef = np.fft.rfft(t - p, axis=0)
    r2, share = [], []
    for lo, hi in BANDS:
        mask = (period > lo) & (period <= hi)
        if lo == NYQUIST:  # o Nyquist fica na banda mais rápida
            mask |= period == NYQUIST
        y_b = np.fft.irfft(yf * mask[:, None], n, axis=0)
        e_b = np.fft.irfft(ef * mask[:, None], n, axis=0)
        sy = (y_b**2).sum(0)
        r2.append(1 - (e_b**2).sum(0) / sy)
        share.append(sy / ((t - t.mean(0)) ** 2).sum(0))
    return np.array(r2), np.array(share)


def spectrum_figure(t: np.ndarray, p: np.ndarray) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 4.4), sharex=True)
    for k, ax in enumerate(axes):
        f, st = welch(t[:, k], fs=1, nperseg=len(t) // 2)
        _, sp = welch(p[:, k], fs=1, nperseg=len(t) // 2)
        ax.loglog(1 / f[1:], st[1:], color="black", lw=1.2, label="medido")
        ax.loglog(1 / f[1:], sp[1:], color="tab:red", lw=1.2, label="previsto")
        for edge in (4, 8, 16, 32):
            ax.axvline(edge, color="#cccccc", lw=0.6, zorder=0)
        ax.set_ylabel(f"{NAMES[k]}\n({UNITS[k]})²·h", fontsize=9)
        ax.grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].invert_xaxis()
    axes[-1].set_xticks([2, 4, 8, 16, 32, 64, 128], minor=False)
    axes[-1].set_xticklabels(["2", "4", "8", "16", "32", "64", "128"])
    axes[-1].set_xlabel("período (horas)")
    fig.tight_layout()
    name = f"stf_modelo_espectro_{SCENARIO}.png"
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    shutil.copy(FIG / name, OVERLEAF / name)


def r2_figure(r2: np.ndarray) -> None:
    mean, std = r2.mean(0), r2.std(0)  # (bands, 3)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    x = np.arange(len(BANDS))
    width = 0.26
    for k in range(3):
        xs = x + (k - 1) * width
        ax.bar(
            xs,
            mean[:, k],
            width,
            yerr=std[:, k],
            capsize=2,
            color=COLORS[k],
            label=NAMES[k],
        )
        for xi, v, sd in zip(xs, mean[:, k], std[:, k], strict=True):
            txt = f"{v:.2f}".replace(".", ",")
            y_txt = v + sd + 0.02 if v >= 0 else v - sd - 0.02
            va = "bottom" if v >= 0 else "top"
            ax.text(xi, y_txt, txt, ha="center", va=va, fontsize=7)
    ax.set_xticks(x, BAND_LABELS)
    ax.set_xlabel("banda (período em horas)")
    ax.set_ylabel("R² na banda (teste)")
    ax.set_ylim(min(0, mean.min() - 0.12), 1.08)
    ax.yaxis.set_major_formatter(COMMA)
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    name = f"stf_modelo_r2_banda_{SCENARIO}.png"
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    shutil.copy(FIG / name, OVERLEAF / name)


def main() -> None:
    runs = {s: seed_preds(SCENARIO, s) for s in SEEDS}
    truth = chained(runs[FIG_SEED]["true"])
    preds = {s: chained(r["final"]) for s, r in runs.items()}

    spectrum_figure(truth, preds[FIG_SEED])
    per_seed = [band_r2(truth, p) for p in preds.values()]
    r2 = np.array([r for r, _ in per_seed])  # (seeds, bands, 3)
    share = per_seed[0][1]
    r2_figure(r2)

    global_r2 = 1 - ((truth - preds[FIG_SEED]) ** 2).sum(0) / (
        (truth - truth.mean(0)) ** 2
    ).sum(0)
    out = {
        "cenario": SCENARIO,
        "horas_teste": len(truth),
        "bandas_h": BAND_LABELS,
        "saidas": NAMES,
        "r2_media": r2.mean(0).round(3).tolist(),
        "r2_desvio": r2.std(0).round(3).tolist(),
        "fracao_variancia": share.round(3).tolist(),
        "r2_global_seed42": global_r2.round(3).tolist(),
        "r2_ponderado_seed42": (per_seed[SEEDS.index(FIG_SEED)][0] * share)
        .sum(0)
        .round(3)
        .tolist(),
    }
    SUMMARY.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
