"""Figuras de resultado da validação cruzada para o artigo.

Usa as previsões gravadas (`outputs/cv/preds`) e as mesmas origens e
combinação do nível do relatório (`report_cv_extended.model_runs`):

- `cv_r2_por_passo_<cenario>.pdf`: R² médio das 3 saídas em cada passo
  do horizonte, média dos 5 folds de teste (cada fold na média das
  sementes);
- `cv_serie_full_f5.pdf`: o teste do fold 5 no cenário `full`, medido e
  previsto pelo GOTAS em previsões de 12 h sem sobreposição.

Uso (da raiz do repo):
    .venv/bin/python notebooks/training/scripts/figure_cv_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from report_cv_extended import BAD, FOLDS, N_FOLDS, H, model_runs

from mimo_sys.training.benchmarks import valid_origins

FIGURES = Path(__file__).resolve().parents[3] / "outputs" / "figures"
WIDTH_IN = 6.3
# Modelo -> (rótulo, cor, estilo).
CURVES = {
    "gotas_aex": ("GOTAS", "#c0392b", "-"),
    "xgboost": ("XGBoost", "#1f4e79", "-"),
    "catboost": ("CatBoost", "#2e86c1", "--"),
    "gru": ("GRU", "#27ae60", ":"),
    "armax": ("ARMAX", "#7f7f7f", "-."),
}
OUTPUTS = (
    ("Vazão de distribuição (l/s)", 0),
    ("Nível (m)", 1),
    ("Pressão (mca)", 2),
)

mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
    }
)


def r2_steps(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """`(H,)` R² médio das saídas em cada passo, sobre as origens."""
    err = ((true - pred) ** 2).sum(axis=0)  # (H, saídas)
    var = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return (1 - err / var).mean(axis=1)


def horizon_curve(model: str, scenario: str) -> np.ndarray | None:
    """R² por passo, média dos folds de teste e das sementes."""
    per_fold = []
    for k in range(1, N_FOLDS + 1):
        runs = model_runs(model, scenario, k)
        if runs is None:
            return None
        n = len(runs[0]["test_true"])
        keep = valid_origins(BAD, FOLDS[k - 1].val_end, n, H)
        per_fold.append(
            np.mean(
                [
                    r2_steps(r["test_true"][keep], r["test_pred"][keep])
                    for r in runs
                ],
                axis=0,
            )
        )
    return np.mean(per_fold, axis=0)


def figure_horizon(scenario: str) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
    steps = np.arange(1, H + 1)
    for model, (label, color, style) in CURVES.items():
        curve = horizon_curve(model, scenario)
        if curve is not None:
            ax.plot(steps, curve, style, color=color, lw=1.8, label=label)
    ax.set_xlabel("Passo do horizonte (horas)")
    ax.set_ylabel("R² médio das três saídas")
    ax.set_xticks(steps)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"cv_r2_por_passo_{scenario}.pdf")
    plt.close(fig)


def figure_series(scenario: str = "full", fold: int = 5) -> None:
    runs = model_runs("gotas_aex", scenario, fold)
    true = runs[0]["test_true"]
    pred = np.mean([r["test_pred"] for r in runs], axis=0)
    start = FOLDS[fold - 1].val_end
    origins = np.arange(0, len(true), H)
    hours = start + np.arange(len(origins) * H)
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH_IN, 6.0), sharex=True)
    for ax, (label, i), letter in zip(axes, OUTPUTS, "abc", strict=True):
        ax.plot(
            hours,
            true[origins, :, i].ravel(),
            color="black",
            lw=1.2,
        )
        ax.plot(
            hours,
            pred[origins, :, i].ravel(),
            color="#c0392b",
            lw=1.2,
        )
        for o in origins:
            ax.axvline(start + o, color="0.85", lw=0.6, zorder=0)
        ax.set_ylabel(label)
        ax.text(
            0.01, 0.95, f"({letter})", transform=ax.transAxes, va="top"
        )
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Hora da série")
    fig.tight_layout()
    fig.savefig(FIGURES / f"cv_serie_{scenario}_f{fold}.pdf")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for scenario in ("full", "pumps"):
        figure_horizon(scenario)
    figure_series()
    print("figuras em", FIGURES)


if __name__ == "__main__":
    main()
