"""Optuna search over architecture and training knobs (Nixtla `Auto*`-style).

Each architecture has a search space for its own kwargs; a shared space
covers the training techniques (loss, RevIN, augmentations, spectral
term, scheduler, EMA). Every trial is a nested MLflow run under one
parent `tune_<model>`; unpromising trials are pruned on `val_mse`.

Needs the `tune` extra: `uv run --extra ml --extra tune ...`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import mlflow
import optuna
from optuna_integration import PyTorchLightningPruningCallback

from mimo_sys.training.data import INPUT_COLS
from mimo_sys.training.train import TrainConfig, train

Space = Callable[[optuna.Trial], dict[str, Any]]

# Pump frequencies only (no `vazao_entrada`): used for the
# `future_known="pumps"` tuning variant, see `tune_spectral_tf_net`.
PUMP_COLS = tuple(c for c in INPUT_COLS if c != "vazao_entrada")

MODEL_SPACES: dict[str, Space] = {
    "mpnn": lambda t: {
        "hidden_size": t.suggest_categorical("hidden_size", [32, 64, 128]),
        "message_dim": t.suggest_int("message_dim", 1, 4),
        "n_layers": t.suggest_int("n_layers", 1, 2),
    },
    "sgn": lambda t: {
        "hidden_size": t.suggest_categorical("hidden_size", [32, 64, 128]),
        "message_dim": t.suggest_int("message_dim", 1, 4),
        "n_layers": t.suggest_int("n_layers", 1, 2),
        "noise_scale": t.suggest_float("noise_scale", 0.0, 2.0),
    },
    "stemgnn": lambda t: {
        "hidden_size": t.suggest_categorical("hidden_size", [32, 64, 128]),
        "cheb_order": t.suggest_int("cheb_order", 2, 5),
        "n_blocks": t.suggest_int("n_blocks", 1, 3),
    },
    "seq2seq_latent_gnn": lambda t: {
        "hidden_size": t.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": t.suggest_int("num_layers", 1, 2),
        "top_k": t.suggest_int("top_k", 2, 6),
        "k_high": t.suggest_int("k_high", 4, 12),
    },
    "spectral_tf_net": lambda t: {
        "n_bands": t.suggest_int("n_bands", 2, 24),
        "n_regimes": t.suggest_int("n_regimes", 1, 4),
        "residual_hidden": t.suggest_categorical(
            "residual_hidden", [0, 8, 16, 32, 64]
        ),
    },
}


def suggest_config(
    trial: optuna.Trial, model: str, base: TrainConfig
) -> TrainConfig:
    """Training config for one trial (also rebuilds the best one).

    Pass an `optuna.trial.FixedTrial(study.best_params)` to turn the
    winning parameters back into a `TrainConfig`.
    """
    cfg = replace(
        base,
        model=model,
        model_kwargs=MODEL_SPACES[model](trial),
        lr=trial.suggest_float("lr", 1e-5, 3e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True),
        optimizer=trial.suggest_categorical(
            "optimizer", ["adam", "adamw", "rmsprop", "sgd_nesterov"]
        ),
        momentum=trial.suggest_float("momentum", 0.0, 0.99),
        loss=trial.suggest_categorical("loss", ["mse", "huber"]),
        revin=trial.suggest_categorical("revin", [None, "standard"]),
        augment=trial.suggest_categorical("augment", [False, True]),
        spectral_weight=trial.suggest_float(
            "spectral_weight", 1e-4, 1.0, log=True
        )
        * trial.suggest_categorical("use_spectral_loss", [0.0, 1.0]),
        scheduler=trial.suggest_categorical(
            "scheduler", ["plateau", "cosine_warmup"]
        ),
        ema_decay=trial.suggest_categorical("ema_decay", [None, 0.999]),
    )
    if model == "spectral_tf_net":
        cfg = replace(
            cfg,
            input_window=trial.suggest_int("input_window", 24, 336),
            batch_size=trial.suggest_int("batch_size", 16, 512, log=True),
            cascade_weight=trial.suggest_float(
                "cascade_weight", 1e-3, 5.0, log=True
            )
            * trial.suggest_categorical("use_cascade_loss", [0.0, 1.0]),
            backcast_weight=trial.suggest_float(
                "backcast_weight", 1e-4, 2.0, log=True
            )
            * trial.suggest_categorical("use_backcast_loss", [0.0, 1.0]),
            message_l1_weight=trial.suggest_float(
                "message_l1_weight", 1e-5, 1e-1, log=True
            )
            * trial.suggest_categorical("use_message_l1", [0.0, 1.0]),
            model_kwargs={
                **cfg.model_kwargs,
                "future_in_input": base.future_exog,
            },
        )
    return cfg


def tune(
    model: str,
    n_trials: int = 30,
    base: TrainConfig | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    metric: str = "val_mse",
) -> optuna.Study:
    """Run the search and log the best trial on the parent run.

    Args:
        model: Key of `MODEL_SPACES`.
        n_trials: Trials to run.
        base: Fixed settings (epochs, patience, preprocessor, seed,
            experiment). Trials never save weights to `models/`; retrain
            the winner with `suggest_config(FixedTrial(...), ...)`.
        study_name: Optuna study name (default `tune_<model>`).
        storage: Optuna storage URL, e.g. `sqlite:///optuna.db`, to
            resume a study.
        metric: `val_mse` (minimised) or `val_r2` (maximised).

    Returns:
        The finished study, optimising the best `metric` per trial.
    """
    if metric not in {"val_mse", "val_r2"}:
        raise ValueError(f"unknown metric: {metric!r}")
    direction = "minimize" if metric == "val_mse" else "maximize"
    result_attr = "best_val_mse" if metric == "val_mse" else "best_val_r2"

    base = base or TrainConfig(max_epochs=100)
    name = study_name or f"tune_{model}"
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=storage is not None,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=base.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10
        ),
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial, model, base)
        cfg = replace(cfg, run_name=f"{name}_trial_{trial.number}", save=False)
        pruning = PyTorchLightningPruningCallback(trial, monitor=metric)
        return getattr(train(cfg, extra_callbacks=[pruning]), result_attr)

    mlflow.set_experiment(base.experiment)
    with mlflow.start_run(run_name=name):
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params(
            {f"best.{k}": v for k, v in study.best_params.items()}
        )
        mlflow.log_metric(f"best_{metric}", study.best_value)
    return study
