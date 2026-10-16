"""Train the ensemble members behind the R²-push experiment, reproducibly.

`docs/changes/2026-10-16-r2-push-experiment.md` summarises the whole
branch's findings. This script only trains the final 5-member pool
that the branch's greedy-forward-selection landed on, logs each
member's metrics, and prints the ensemble's honest test R² (members
selected on validation, averaged on test — never selected on test).

Winning recipe per member: MPNN/StemGNN/Seq2SeqLatentGNN with
`per_output_heads` + `uncertainty_weighted` + RevIN + the explicit
mass-balance integral feature (`cfg.mass_balance_feature`).

Uso:
    cd notebooks/training
    uv run --extra ml python scripts/run_ensemble_r2_push.py
"""

from __future__ import annotations

import numpy as np

from mimo_sys.training import (
    TrainConfig,
    add_derived_features,
    average_predictions,
    train,
)
from mimo_sys.training.data import TARGET_IDX, load_splits
from mimo_sys.training.evaluation import chained_forecast, metrics_table

EXPERIMENT = "mimo_sys_r2_push"

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


def main() -> None:
    """Train every member, then report the ensemble's test metrics."""
    test_preds = []
    y_test = None
    for name, extra in MEMBERS.items():
        cfg = TrainConfig(run_name=name, **COMMON, **extra)
        print(f"training {name}...", flush=True)
        result = train(cfg)
        splits = add_derived_features(cfg, load_splits(cfg.data_prefix))
        raw = np.concatenate([splits["train"], splits["val"], splits["test"]])
        val_end = len(splits["train"]) + len(splits["val"])
        test_seg = raw[val_end - cfg.input_window :]
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
    print("\n=== ensemble (5 members, equal weight) ===")
    print(metrics_table(y_test, ensemble).round(3))


if __name__ == "__main__":
    main()
