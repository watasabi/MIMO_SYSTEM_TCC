"""Retrain the Optuna winners (full / pumps) and report final metrics.

Rebuilds each variant's best trial with `FixedTrial`, trains with
`save=True` (persists weights to `models/`), and reports R²/MAE/RMSE
per output on the test set — `nivel_res` through the mass-balance
cascade, same protocol as `run_spectral_tf_net.run`.

Uso (da raiz do repo):
    uv run --extra ml --extra tune python \
        notebooks/training/scripts/retrain_best_spectral_tf_net.py
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import optuna
import torch
from tune_spectral_tf_net import STORAGE_DIR, base_config

from mimo_sys.training.data import (
    INPUT_COLS,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.evaluation import metrics_table
from mimo_sys.training.mass_balance_cascade import (
    fit_mass_balance,
    mass_balance_cascade_forecast,
)
from mimo_sys.training.train import add_derived_features, train
from mimo_sys.training.tuning import suggest_config

VAZAO, NIVEL = 0, 1
REPORT = PROJECT_ROOT / "outputs" / "spectral_tf_net_tuning_report.json"


def cascade_level(cfg, y_pred: np.ndarray) -> np.ndarray:
    """`nivel_res` from the physics cascade fed by the predicted flow."""
    splits = add_derived_features(cfg, load_splits(cfg.data_prefix))
    raw = np.concatenate([splits["train"], splits["val"], splits["test"]])
    val_end = len(splits["train"]) + len(splits["val"])
    seg = raw[val_end - cfg.input_window :]
    inflow = INPUT_COLS.index("vazao_entrada")
    vz, nv = TARGET_IDX[VAZAO], TARGET_IDX[NIVEL]
    tr = splits["train"]
    a, b = fit_mass_balance(tr[:, nv], tr[:, inflow], tr[:, vz])
    h = cfg.output_window
    anchors = np.arange(cfg.input_window, len(seg) - h + 1, h)[
        : len(y_pred) // h
    ]
    return mass_balance_cascade_forecast(
        seg[anchors - 1, nv],
        np.stack(
            [
                seg[t : t + h, inflow]
                if "vazao_entrada" in cfg.future_known
                else seg[t - 24 : t - 24 + h, inflow]
                for t in anchors
            ]
        ),
        y_pred[:, VAZAO].reshape(len(anchors), h),
        a,
        b,
        last_imbalance=seg[anchors - 1, inflow] - seg[anchors - 1, vz],
    ).reshape(-1)


def retrain(variant: str) -> dict:
    storage = f"sqlite:///{STORAGE_DIR / 'optuna_spectral_tf_net.db'}"
    study = optuna.load_study(
        study_name=f"tune_spectral_tf_net_{variant}", storage=storage
    )
    trial = optuna.trial.FixedTrial(study.best_params)
    base = base_config(variant)
    cfg = suggest_config(trial, "spectral_tf_net", base)
    cfg = dataclasses.replace(
        cfg,
        save=True,
        run_name=f"stf_tuned_{variant}_best",
        experiment="mimo_sys_spectral_tf_net",
    )
    result = train(cfg)
    y_true, y_pred = result.y_true, result.y_pred
    final = y_pred.copy()
    final[:, NIVEL] = cascade_level(cfg, y_pred)
    r2_raw = metrics_table(y_true, y_pred)
    r2_final = metrics_table(y_true, final)
    # The cascade only helps when it beats the model's own direct
    # `nivel_res` head; with `vazao_entrada` not known in the future
    # (`pumps`), its seasonal-naive inflow can make the cascade worse.
    nivel_direct_r2 = r2_raw.loc["nivel_res", "r2"]
    nivel_cascata_r2 = r2_final.loc["nivel_res", "r2"]
    cascade_helps = nivel_cascata_r2 > nivel_direct_r2
    nivel_best = r2_final if cascade_helps else r2_raw
    return {
        "variant": variant,
        "best_val_mse": study.best_value,
        "best_params": study.best_params,
        "test_metrics_raw": r2_raw.to_dict(orient="index"),
        "test_metrics_nivel_cascata": {
            "nivel_res": r2_final.loc["nivel_res"].to_dict()
        },
        "cascade_helps_nivel": bool(cascade_helps),
        "r2_medio": float(
            np.mean(
                [
                    r2_raw.loc["vazao_distribuicao", "r2"],
                    nivel_best.loc["nivel_res", "r2"],
                    r2_raw.loc["pressao", "r2"],
                ]
            )
        ),
        "run_id": result.run_id,
    }


def main() -> None:
    torch.set_float32_matmul_precision("high")
    rows = [retrain(v) for v in ("full", "pumps")]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    for row in rows:
        print(
            f"{row['variant']}: médio {row['r2_medio']:.3f} "
            f"(val_mse {row['best_val_mse']:.5f}, "
            f"cascata ajuda nível: {row['cascade_helps_nivel']}, "
            f"run {row['run_id']})"
        )
    print(f"relatório completo em {REPORT}")


if __name__ == "__main__":
    main()
