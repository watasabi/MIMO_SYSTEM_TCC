"""Optuna (TPE + median pruning) hyperparameter search for SpectralTFNet.

Two variants, same search space otherwise:

- `full`: known future inputs are the pump frequencies *and* the
  inflow (`future_known=INPUT_COLS`, the current default).
- `pumps`: only the pump frequencies are known ahead of time; the
  inflow is not (`future_known=PUMP_COLS`), so the cascade falls back
  to the seasonal-naive inflow guess (`Forecaster.cascade_inflow_known`).

Resumable: the study is persisted to a sqlite file, so killing and
re-running this script continues the same study instead of restarting.

Uso (da raiz do repo):
    S=notebooks/training/scripts/tune_spectral_tf_net.py
    uv run --extra ml --extra tune python $S full 200
    uv run --extra ml --extra tune python $S pumps 200
"""

from __future__ import annotations

import sys

import torch

from mimo_sys.training.data import INPUT_COLS, PROJECT_ROOT
from mimo_sys.training.train import TrainConfig
from mimo_sys.training.tuning import PUMP_COLS, tune

STORAGE_DIR = PROJECT_ROOT / "outputs"
OUTPUT_WINDOW = 12

VARIANTS: dict[str, tuple[str, ...]] = {
    "full": tuple(INPUT_COLS),
    "pumps": PUMP_COLS,
}


def base_config(variant: str) -> TrainConfig:
    return TrainConfig(
        model="spectral_tf_net",
        output_window=OUTPUT_WINDOW,
        future_exog=True,
        future_known=VARIANTS[variant],
        mass_balance_feature=True,
        uncertainty_weighted=True,
        max_epochs=150,
        patience=20,
        accelerator="auto",
        precision="bf16-mixed",
        progress=True,
        experiment="mimo_sys_spectral_tf_net_tuning",
        save=False,
    )


def main(variant: str, n_trials: int) -> None:
    # Tensor Cores on the 3060; silences the per-trial Lightning warning.
    torch.set_float32_matmul_precision("high")
    STORAGE_DIR.mkdir(exist_ok=True)
    storage = f"sqlite:///{STORAGE_DIR / 'optuna_spectral_tf_net_r2.db'}"
    study_name = f"tune_spectral_tf_net_{variant}_r2"
    study = tune(
        model="spectral_tf_net",
        n_trials=n_trials,
        base=base_config(variant),
        study_name=study_name,
        storage=storage,
        metric="val_r2",
    )
    print(f"{study_name}: best val_r2 {study.best_value:.5f}")
    print(study.best_params)


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = args[0] if args else "full"
    n_trials = int(args[1]) if len(args) > 1 else 100
    if variant not in VARIANTS:
        raise SystemExit(f"variant must be one of {tuple(VARIANTS)}")
    main(variant, n_trials)
