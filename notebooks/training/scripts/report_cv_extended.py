"""Relatório estendido da validação cruzada: mais métricas e pós-hoc.

Lê só as previsões já salvas por `run_cv_benchmark.py` (não retreina
nada). Acrescenta ao `report()` oficial:

- métricas de validação e teste (mean ± desvio entre folds). O erro de
  treino não é salvo pelo `final`, então não entra;
- MAPE, além de R², RMSE, MAE, sMAPE, MASE e KGE;
- R² no espectro de amplitude (`evaluation.amplitude_spectrum`), sem a
  frequência zero;
- pós-hoc: bootstrap pareado em blocos e Diebold-Mariano de cada
  modelo contra o GOTAS, nas mesmas origens, por cenário.

As métricas usam o protocolo de 12 h do artigo: origens a cada 12 h,
sem sobreposição, para que as janelas concatenadas formem uma série no
tempo. Os testes pós-hoc usam todas as origens horárias, e a variância
de Newey-West do Diebold-Mariano trata a sobreposição.

Uso (da raiz do repo):
    S=notebooks/training/scripts/report_cv_extended.py
    uv run --extra ml --extra tune --extra boost python $S
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_cv_all import FINAL_SEEDS, MODELS, N_FOLDS, OUT, pred_path  # noqa: E402
from run_cv_benchmark import (  # noqa: E402
    BAD,
    FOLDS,
    RAW,
    SCENARIOS,
    H,
    reference_forecasts,
)
from run_improvements import (  # noqa: E402
    _blend_weights,  # noqa: PLC2701
    _split,  # noqa: PLC2701
)

from mimo_sys.training.benchmarks import (  # noqa: E402
    seasonal_scale,
    valid_origins,
)
from mimo_sys.training.data import TARGET_IDX  # noqa: E402
from mimo_sys.training.evaluation import (  # noqa: E402
    amplitude_spectrum,
    diebold_mariano,
    paired_block_bootstrap,
)
from mimo_sys.training.mass_balance_cascade import apply_level_blend  # noqa: E402

OUTPUTS = ["vazao_distribuicao", "nivel_res", "pressao"]
REPORT = OUT / "report_extended.json"
TABLE = OUT / "report_extended.txt"
ZERO_GUARD = 1e-9
VARIANCE_GUARD = 1e-12


def mape(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """MAPE (%); undefined (`y == 0`) rows are dropped per output."""
    out = np.full(y.shape[-1], np.nan)
    err = np.abs((y - pred) / y)
    for j in range(y.shape[-1]):
        mask = np.abs(y[..., j]) > ZERO_GUARD
        if mask.any():
            out[j] = 100 * err[..., j][mask].mean()
    return out


def spectral_r2(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """R² of the one-sided amplitude spectrum, per output, without DC.

    `y`/`pred`: `(n_origins, horizon, n_outputs)`, non-overlapping
    origins spaced `horizon` apart (`split_scores`), flattened back into
    one sequence per output before the FFT. The zero frequency (the
    mean) is dropped: it is orders of magnitude larger than the other
    bins and would give R² ≈ 1 to any forecast with the right mean.
    """
    at = amplitude_spectrum(y.reshape(-1, y.shape[-1]))[1:]
    ap = amplitude_spectrum(pred.reshape(-1, pred.shape[-1]))[1:]
    sse = ((at - ap) ** 2).sum(axis=0)
    sst = ((at - at.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sst > VARIANCE_GUARD, 1 - sse / sst, np.nan)


def extended_metrics(
    y3: np.ndarray, pred3: np.ndarray, scale: np.ndarray
) -> dict[str, np.ndarray]:
    """`benchmarks.forecast_metrics` plus MAPE and spectral R².

    Args:
        y3, pred3: `(n_origins, horizon, n_outputs)`, already filtered
            to valid origins. Every metric but the spectral one pools
            origins and horizon together (`reshape(-1, n_outputs)`);
            the spectral one needs the time order, so it keeps the 3D
            shape.
    """
    spectral = spectral_r2(y3, pred3)
    y, pred = y3.reshape(-1, y3.shape[-1]), pred3.reshape(-1, pred3.shape[-1])
    err = y - pred
    mae = np.abs(err).mean(axis=0)
    y_c, p_c = y - y.mean(axis=0), pred - pred.mean(axis=0)
    y_std, p_std = y.std(axis=0), pred.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (y_c * p_c).mean(axis=0) / (y_std * p_std)
        kge = 1 - np.sqrt(
            (corr - 1) ** 2
            + (p_std / y_std - 1) ** 2
            + (pred.mean(axis=0) / y.mean(axis=0) - 1) ** 2
        )
    return {
        "r2": 1 - (err**2).sum(axis=0) / (y_c**2).sum(axis=0),
        "rmse": np.sqrt((err**2).mean(axis=0)),
        "mae": mae,
        "mape": mape(y, pred),
        "smape": 100
        * np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(pred)), axis=0),
        "mase": mae / scale,
        "kge": kge,
        "r2_espectral": spectral,
    }


def split_scores(
    true: np.ndarray, pred: np.ndarray, fold: int, split: str
) -> dict[str, np.ndarray]:
    """Metrics on `split` ('val'/'test'): one origin every `H` hours.

    Same 12 h protocol as `run_cv_benchmark.scores`: non-overlapping
    origins whose horizon touches no sensor failure.
    """
    f = FOLDS[fold - 1]
    starts = {"val": f.train_end, "test": f.val_end}
    scale = seasonal_scale(RAW[: f.train_end][:, TARGET_IDX])
    idx = np.arange(0, len(true), H)
    idx = idx[valid_origins(BAD, starts[split], len(true), H)[idx]]
    return extended_metrics(true[idx], pred[idx], scale)


def model_runs(model: str, scenario: str, fold: int) -> list[dict] | None:
    """Raw arrays of every available seed, blended level included."""
    f = FOLDS[fold - 1]
    seeds = FINAL_SEEDS[:1] if model == "armax" else FINAL_SEEDS
    runs = []
    for seed in seeds:
        path = pred_path(model, scenario, fold, seed)
        if not path.exists():
            return None
        arrays = dict(np.load(path))
        val = _split(arrays, "val")
        keep = valid_origins(BAD, f.train_end, len(val["true"]), H)
        weights = _blend_weights({k: v[keep] for k, v in val.items()})
        test_pred = arrays["test_pred"].copy()
        test_pred[..., 1] = apply_level_blend(
            weights, arrays["test_pred"][..., 1], arrays["test_nivel_cascata"]
        )
        # Na validação, os pesos do nível de cada metade vêm da outra
        # metade, como no objetivo da busca (`_crossfit_val`): ajustar e
        # avaliar nas mesmas origens deixaria a validação otimista.
        val_pred = val["pred"].copy()
        n = len(val["true"])
        halves = (np.arange(n) < n // 2, np.arange(n) >= n // 2)
        for fit_on, apply_on in (halves, halves[::-1]):
            part = {k: v[fit_on & keep] for k, v in val.items()}
            val_pred[apply_on, :, 1] = apply_level_blend(
                _blend_weights(part),
                val["pred"][apply_on, :, 1],
                val["nivel_cascata"][apply_on],
            )
        runs.append(
            {
                "val_true": val["true"],
                "val_pred": val_pred,
                "test_true": arrays["test_true"],
                "test_pred": test_pred,
            }
        )
    return runs


def summarize_folds(per_fold: list[dict]) -> dict:
    out = {}
    for name in per_fold[0]:
        values = np.stack([m[name] for m in per_fold])  # (folds, outputs)
        out[name] = {
            "media_saidas": round(float(np.nanmean(values)), 5),
            "desvio_saidas": round(float(np.nanmean(values, axis=1).std()), 5),
            "media_por_saida": np.nanmean(values, axis=0).round(5).tolist(),
            "desvio_por_saida": values.std(axis=0).round(5).tolist(),
        }
    out["folds"] = len(per_fold)
    return out


def _paired_test_arrays(
    base_runs: dict[int, list[dict]], runs: dict[int, list[dict]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatena as origens de teste válidas dos 5 folds, sem sobrepor.

    Pareia por semente (até o menor número de sementes dos dois lados)
    em cada fold antes de concatenar, para o bootstrap e o
    Diebold-Mariano compararem a mesma origem e a mesma semente.
    """
    true_all, base_all, other_all = [], [], []
    for k in range(1, N_FOLDS + 1):
        f = FOLDS[k - 1]
        n = len(base_runs[k][0]["test_true"])
        keep = valid_origins(BAD, f.val_end, n, H)
        n_seeds = min(len(base_runs[k]), len(runs[k]))
        true_all.append(base_runs[k][0]["test_true"][keep])
        base_all.append(
            np.stack(
                [base_runs[k][s]["test_pred"][keep] for s in range(n_seeds)]
            )
        )
        other_all.append(
            np.stack([runs[k][s]["test_pred"][keep] for s in range(n_seeds)])
        )
    return (
        np.concatenate(true_all, axis=0),
        np.concatenate(base_all, axis=1),
        np.concatenate(other_all, axis=1),
    )


def posthoc(scenario: str) -> dict:
    """Bootstrap pareado + Diebold-Mariano de cada modelo contra o GOTAS."""
    out = {}
    base_runs = {
        k: model_runs("gotas", scenario, k) for k in range(1, N_FOLDS + 1)
    }
    if any(r is None for r in base_runs.values()):
        return out
    for model in MODELS:
        if model == "gotas":
            continue
        runs = {
            k: model_runs(model, scenario, k) for k in range(1, N_FOLDS + 1)
        }
        if any(r is None for r in runs.values()):
            continue
        true, gotas, other = _paired_test_arrays(base_runs, runs)
        boot = paired_block_bootstrap(true, gotas, other)
        stat, p = diebold_mariano(true, gotas, other)
        out[model] = {
            "delta_r2_vs_gotas": boot["delta"].round(5).tolist(),
            "delta_r2_media_vs_gotas": round(boot["delta_mean"], 5),
            "ic95_media": [
                round(boot["ci_low_mean"], 5),
                round(boot["ci_high_mean"], 5),
            ],
            "diebold_mariano_stat": round(stat, 4),
            "diebold_mariano_p": round(p, 4),
            "origens": int(len(true)),
        }
    return out


ROW_METRICS = (
    ("r2", "R2", 3),
    ("r2_espectral", "R2_espec", 3),
    ("rmse", "RMSE", 3),
    ("mae", "MAE", 3),
    ("mape", "MAPE%", 1),
    ("smape", "sMAPE%", 1),
    ("mase", "MASE", 2),
    ("kge", "KGE", 3),
)
NO_VAL_BASELINE = {"persistencia", "sazonal"}


def mean_scores(runs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Média, entre sementes, de cada métrica por saída."""
    return {name: np.mean([r[name] for r in runs], axis=0) for name in runs[0]}


def scenario_table(scenario: str) -> dict:
    """Val/test summary of every model, plus the naive references."""
    table: dict[str, dict] = {}
    refs = [reference_forecasts(k) for k in range(1, N_FOLDS + 1)]
    for name in NO_VAL_BASELINE:
        per_fold = [
            split_scores(r["true"], r[name], k, "test")
            for k, r in enumerate(refs, start=1)
        ]
        table[name] = {"test": summarize_folds(per_fold)}
    for model in MODELS:
        per_val, per_test = [], []
        for k in range(1, N_FOLDS + 1):
            runs = model_runs(model, scenario, k)
            if runs is None:
                break
            # Métrica de cada semente, depois a média, como no `report()`.
            per_val.append(
                mean_scores(
                    [
                        split_scores(r["val_true"], r["val_pred"], k, "val")
                        for r in runs
                    ]
                )
            )
            per_test.append(
                mean_scores(
                    [
                        split_scores(r["test_true"], r["test_pred"], k, "test")
                        for r in runs
                    ]
                )
            )
        if len(per_test) == N_FOLDS:
            table[model] = {
                "val": summarize_folds(per_val),
                "test": summarize_folds(per_test),
            }
    return table


def render_scenario(scenario: str, data: dict) -> list[str]:
    """Text lines: test metrics table, val->test and pós-hoc."""
    lines = [
        f"\n## {scenario}: métricas de teste, 12 h, mean ± std entre folds",
        f"{'modelo':24s} "
        + " ".join(f"{label:>14s}" for _, label, _ in ROW_METRICS),
    ]
    ranked = sorted(
        data["modelos"],
        key=lambda m: -data["modelos"][m]["test"]["r2"]["media_saidas"],
    )
    for model in ranked:
        m = data["modelos"][model]["test"]
        cells = [
            f"{m[key]['media_saidas']:.{prec}f}±{m[key]['desvio_saidas']:.{prec}f}"
            for key, _, prec in ROW_METRICS
        ]
        lines.append(f"{model:24s} " + " ".join(cells))
    lines.append(f"\n## {scenario}: validação → teste (R² médio das 3 saídas)")
    for model in ranked:
        if model in NO_VAL_BASELINE:
            continue
        d = data["modelos"][model]
        lines.append(
            f"{model:24s} val {d['val']['r2']['media_saidas']:.3f} "
            f"-> teste {d['test']['r2']['media_saidas']:.3f}"
        )
    lines.append(
        f"\n## {scenario}: pós-hoc vs. GOTAS "
        "(bootstrap pareado + Diebold-Mariano)"
    )
    for model, p in data["posthoc_vs_gotas"].items():
        lines.append(
            f"{model:24s} Δ R² = {p['delta_r2_media_vs_gotas']:+.3f} "
            f"IC95 [{p['ic95_media'][0]:+.3f}, {p['ic95_media'][1]:+.3f}] "
            f"| DM stat={p['diebold_mariano_stat']:+.2f} "
            f"p={p['diebold_mariano_p']:.3f} (n={p['origens']})"
        )
    return lines


def main() -> None:
    summary = {
        scenario: {
            "modelos": scenario_table(scenario),
            "posthoc_vs_gotas": posthoc(scenario),
        }
        for scenario in SCENARIOS
    }
    REPORT.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    lines = [
        line
        for scenario, data in summary.items()
        for line in render_scenario(scenario, data)
    ]
    TABLE.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
