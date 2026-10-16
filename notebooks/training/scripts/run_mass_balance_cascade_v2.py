"""Log the ensemble + physics-cascade result to MLflow, with plots.

Extends `run_mass_balance_cascade.py`: besides printing metrics, this
logs two MLflow runs with `log_series`/`log_spectrum` artifacts (PNG +
HTML, same convention as `train(..., log_plots=True)`), so
`notebooks/eda/05_mass_balance_cascade_eda.ipynb` has something to
show:

- `ensemble_ml`: the 5-member ML ensemble alone (R² médio 0.726).
- `ensemble_cascade`: same `vazao_distribuicao`/`pressao`, `nivel_res`
  replaced by the physics cascade (R² médio 0.809).

Uso:
    cd notebooks/training
    uv run --extra ml python scripts/run_mass_balance_cascade_v2.py
"""

from __future__ import annotations

import mlflow
import numpy as np

from mimo_sys.training import (
    TrainConfig,
    add_derived_features,
    average_predictions,
    fit_mass_balance,
    mass_balance_cascade_forecast,
    train,
)
from mimo_sys.training.data import (
    INPUT_COLS,
    OUTPUT_COLS,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.evaluation import (
    band_table,
    chained_forecast,
    log_figure,
    metrics_table,
    plot_series,
    plot_spectrum,
)

EXPERIMENT = "mimo_sys_mass_balance_cascade"
INPUT_WINDOW, OUTPUT_WINDOW = 24, 12
VAZAO_POS, NIVEL_POS = 0, 1

COMMON = dict(
    per_output_heads=True,
    per_output_heads_hidden=32,
    uncertainty_weighted=True,
    revin="standard",
    mass_balance_feature=True,
    max_epochs=150,
    patience=25,
    accelerator="cpu",
    experiment=EXPERIMENT,
    save=False,
)

MEMBERS = {
    "mpnn_md8_s42": dict(
        model="mpnn",
        model_kwargs=dict(hidden_size=128, message_dim=8, n_layers=2),
        seed=42,
    ),
    "mpnn_base_s42": dict(
        model="mpnn",
        model_kwargs=dict(hidden_size=128, message_dim=16, n_layers=2),
        seed=42,
    ),
    "stemgnn_s43": dict(
        model="stemgnn", model_kwargs=dict(hidden_size=64), seed=43
    ),
    "seq2seq_s42": dict(
        model="seq2seq_latent_gnn",
        model_kwargs=dict(hidden_size=64, top_k=5, k_high=20),
        seed=42,
    ),
    "seq2seq_s43": dict(
        model="seq2seq_latent_gnn",
        model_kwargs=dict(hidden_size=64, top_k=5, k_high=20),
        seed=43,
    ),
}


def _log_run(
    run_name: str, y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray
) -> None:
    """Log metrics, band table and plots for one combined forecast."""
    with mlflow.start_run(run_name=run_name):
        table = metrics_table(y_true, y_pred)
        bands = band_table(y_true, y_pred)
        for out in OUTPUT_COLS:
            for name, value in table.loc[out].items():
                key = name.replace("%", "pct")
                mlflow.log_metric(f"{key}_{out}", float(value))
        for (band, out), row in bands.iterrows():
            mlflow.log_metric(
                f"energia_rel_pct_{band}_{out}", float(row["energia_rel_%"])
            )
        log_figure(plot_series(y_true, y_pred, run_name, idx, table), "serie")
        log_figure(plot_spectrum(y_true, y_pred, run_name, bands), "espectro")
        print(f"  logged {run_name}: {table['r2'].round(3).tolist()}")


def main() -> None:  # noqa: PLR0914 - linear on purpose
    """Train the members, log the ML ensemble and the cascade result."""
    test_preds = []
    y_test = None
    test_seg_for_physics = None
    for name, extra in MEMBERS.items():
        cfg = TrainConfig(run_name=name, **COMMON, **extra)
        print(f"training {name}...", flush=True)
        result = train(cfg)
        splits = add_derived_features(cfg, load_splits(cfg.data_prefix))
        raw = np.concatenate([splits["train"], splits["val"], splits["test"]])
        val_end = len(splits["train"]) + len(splits["val"])
        test_seg = raw[val_end - cfg.input_window :]
        test_seg_for_physics = test_seg
        model = result.module.model.eval()
        y_pred = chained_forecast(
            model,
            test_seg,
            result.preprocessor,
            start=cfg.input_window,
            stop=len(test_seg),
            device="cpu",
            input_window=cfg.input_window,
            output_window=cfg.output_window,
        )
        if y_test is None:
            y_test = test_seg[
                cfg.input_window : cfg.input_window + len(y_pred)
            ][:, TARGET_IDX]
        test_preds.append(y_pred)

    ensemble = average_predictions(test_preds)
    idx = np.arange(len(ensemble))
    mlflow.set_experiment(EXPERIMENT)
    _log_run("ensemble_ml", y_test, ensemble, idx)

    train_splits = load_splits()
    a, b = fit_mass_balance(
        train_splits["train"][:, TARGET_IDX[NIVEL_POS]],
        train_splits["train"][:, INPUT_COLS.index("vazao_entrada")],
        train_splits["train"][:, TARGET_IDX[VAZAO_POS]],
    )

    inflow_col = INPUT_COLS.index("vazao_entrada")
    anchors = np.arange(
        INPUT_WINDOW,
        len(test_seg_for_physics) - OUTPUT_WINDOW + 1,
        OUTPUT_WINDOW,
    )
    level_anchor = test_seg_for_physics[anchors - 1, TARGET_IDX[NIVEL_POS]]
    inflow_windows = np.stack(
        [
            test_seg_for_physics[t0 : t0 + OUTPUT_WINDOW, inflow_col]
            for t0 in anchors
        ]
    )
    last_imbalance = (
        test_seg_for_physics[anchors - 1, inflow_col]
        - test_seg_for_physics[anchors - 1, TARGET_IDX[VAZAO_POS]]
    )
    predicted_outflow = ensemble[:, VAZAO_POS].reshape(
        len(anchors), OUTPUT_WINDOW
    )
    cascade_nivel = mass_balance_cascade_forecast(
        level_anchor,
        inflow_windows,
        predicted_outflow,
        a,
        b,
        last_imbalance=last_imbalance,
    ).reshape(-1)

    final = ensemble.copy()
    final[:, NIVEL_POS] = cascade_nivel
    _log_run("ensemble_cascade", y_test, final, idx)
    print(
        "mean R2 (ensemble_cascade):",
        metrics_table(y_test, final)["r2"].mean().round(3),
    )


if __name__ == "__main__":
    main()
