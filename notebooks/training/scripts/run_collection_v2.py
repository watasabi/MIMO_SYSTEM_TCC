"""Retreina o baseline e a melhor variante de cada modelo, com plots.

Uso:
    cd notebooks/training
    uv run --extra ml python scripts/run_collection_v2.py
"""

from __future__ import annotations

from dataclasses import replace

from mimo_sys.training import TrainConfig, train

EXPERIMENT = "mimo_sys_collection_v2"

MODEL_KWARGS = {
    "mpnn": {"hidden_size": 64, "message_dim": 8, "n_layers": 1},
    "sgn": {"hidden_size": 64, "message_dim": 8, "n_layers": 1},
    "stemgnn": {"hidden_size": 64},
    "seq2seq_latent_gnn": {"hidden_size": 64},
}
BEST_VARIANT = {
    "mpnn": {"scheduler": "cosine_warmup", "gradient_clip_val": 1.0},
    "sgn": {"augment": True},
    "stemgnn": {"loss": "huber"},
    "seq2seq_latent_gnn": {
        "revin": "standard",
        "augment": True,
        "spectral_weight": 0.1,
        "ema_decay": 0.999,
    },
}


def configs() -> list[TrainConfig]:
    """Baseline + melhor variante, para cada um dos 4 modelos."""
    base = TrainConfig(
        input_window=24,
        output_window=12,
        experiment=EXPERIMENT,
        save=False,
        log_plots=True,
    )
    out = []
    for model, kwargs in MODEL_KWARGS.items():
        out.append(
            replace(base, model=model, model_kwargs=kwargs, run_name=model)
        )
        out.append(
            replace(
                base,
                model=model,
                model_kwargs=kwargs,
                run_name=f"{model}_otimizado",
                **BEST_VARIANT[model],
            )
        )
    return out


def main() -> None:
    """Train every config, printing progress."""
    cfgs = configs()
    for i, cfg in enumerate(cfgs, 1):
        print(f"[{i}/{len(cfgs)}] {cfg.run_name}", flush=True)
        result = train(cfg)
        print("  r2_medio:", result.metrics["r2"].mean().round(3))


if __name__ == "__main__":
    main()
