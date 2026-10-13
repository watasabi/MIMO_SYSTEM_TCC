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

from mimo_sys.training.train import TrainConfig, train

Space = Callable[[optuna.Trial], dict[str, Any]]

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
}


def suggest_config(
    trial: optuna.Trial, model: str, base: TrainConfig
) -> TrainConfig:
    """Training config for one trial (also rebuilds the best one).

    Pass an `optuna.trial.FixedTrial(study.best_params)` to turn the
    winning parameters back into a `TrainConfig`.
    """
    return replace(
        base,
        model=model,
        model_kwargs=MODEL_SPACES[model](trial),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        loss=trial.suggest_categorical("loss", ["mse", "huber"]),
        revin=trial.suggest_categorical("revin", [None, "standard"]),
        augment=trial.suggest_categorical("augment", [False, True]),
        spectral_weight=trial.suggest_categorical(
            "spectral_weight", [0.0, 0.05, 0.2]
        ),
        scheduler=trial.suggest_categorical(
            "scheduler", ["plateau", "cosine_warmup"]
        ),
        ema_decay=trial.suggest_categorical("ema_decay", [None, 0.999]),
    )


def tune(
    model: str,
    n_trials: int = 30,
    base: TrainConfig | None = None,
    study_name: str | None = None,
    storage: str | None = None,
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

    Returns:
        The finished study; minimises the best `val_mse` per trial.
    """
    base = base or TrainConfig(max_epochs=100)
    name = study_name or f"tune_{model}"
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=storage is not None,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10
        ),
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial, model, base)
        cfg = replace(cfg, run_name=f"{name}_trial_{trial.number}", save=False)
        pruning = PyTorchLightningPruningCallback(trial, monitor="val_mse")
        return train(cfg, extra_callbacks=[pruning]).best_val_mse

    mlflow.set_experiment(base.experiment)
    with mlflow.start_run(run_name=name):
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params(
            {f"best.{k}": v for k, v in study.best_params.items()}
        )
        mlflow.log_metric("best_val_mse", study.best_value)
    return study
