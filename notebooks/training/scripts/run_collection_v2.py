"""Retreina baseline + melhor variante de cada modelo, com plots.

A grade de `run_explainability_grid.py` (rota B, `input_window=24`) só
tinha `save=False` e nenhum plot — boa para comparar métricas, inútil
para olhar série/espectro. Este script retreina, com `log_plots=True`,
a config baseline e a melhor variante por R² médio de cada um dos 4
modelos (`docs/changes/2026-10-14-explainability-training.md`):

    mpnn               -> cosine_clip
    sgn                -> augment
    stemgnn            -> huber
    seq2seq_latent_gnn -> combinado

Experimento MLflow `mimo_sys_collection_v2`, SEED=42 (uma seed — o
objetivo é comparar visualmente, não repetir o estudo estatístico já
feito na grade).

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
