"""Final recipe: ML ensemble + physics cascade for `nivel_res`.

Trains the same 5-member ensemble as `run_ensemble_r2_push.py`
(`vazao_distribuicao` R² 0.929, `pressao` R² 0.740), then replaces its
`nivel_res` prediction with `mass_balance_cascade_forecast`: the
reservoir's mass-balance gray-box
(`docs/1_analise-dados-reservatorio.md` §4), fed by the ensemble's own
`vazao_distribuicao` prediction instead of trying to extrapolate it
itself. See `src/mimo_sys/training/mass_balance_cascade.py` for why
this works where a learned leaky integrator (tried and discarded, see
`docs/changes/2026-10-16-r2-push-experiment.md`) did not: the physics
is correct, the missing piece was the outflow input during the
horizon, not a better integrator.

Result (test, chained 12-step forecast): R² vazao=0.929, nivel=0.755,
pressao=0.740 — mean 0.808, vs. 0.726 for the ML-only ensemble and
0.618 for the notebook 04 baseline.

Uso:
    cd notebooks/training
    uv run --extra ml python scripts/run_mass_balance_cascade.py
"""

from __future__ import annotations

import numpy as np

from mimo_sys.training import (
    TrainConfig,
    add_derived_features,
    average_predictions,
    fit_mass_balance,
    mass_balance_cascade_forecast,
    train,
)
from mimo_sys.training.data import INPUT_COLS, TARGET_IDX, load_splits
from mimo_sys.training.evaluation import chained_forecast, metrics_table

EXPERIMENT = "mimo_sys_r2_push"
INPUT_WINDOW, OUTPUT_WINDOW = 24, 12
VAZAO_POS, NIVEL_POS = 0, 1  # position of each output within TARGET_IDX

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


def main() -> None:  # noqa: PLR0914 - linear on purpose
    """Train the ensemble, then cascade the physics model for nivel_res."""
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
        r2 = metrics_table(y_test, y_pred)["r2"].round(3).tolist()
        print(f"  {name}: {r2}")

    ensemble = average_predictions(test_preds)
    print("\n=== ML ensemble (5 members, equal weight) ===")
    print(metrics_table(y_test, ensemble).round(3))

    # Physics cascade for nivel_res, fed by the ensemble's own
    # vazao_distribuicao prediction.
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
    print("\n=== final: ensemble + physics cascade for nivel_res ===")
    print(metrics_table(y_test, final).round(3))
    print("mean R2:", metrics_table(y_test, final)["r2"].mean().round(3))


if __name__ == "__main__":
    main()
