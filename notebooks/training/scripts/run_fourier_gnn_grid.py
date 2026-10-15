"""Grade de técnicas para o FourierGNN, mesma receita da rota B.

FourierGNN não expõe `encode()` (sem corte limpo de encoder — o FGO é
um multiplicador compartilhado, não um encoder/decoder separável),
então fica fora da variante `contrastive`, igual ao StemGNN
(`run_explainability_grid.py`). Mesmas 8 variantes, `input_window=24`
(piora em 48/72 — ver `docs/changes/2026-10-15-...`), 3 seeds,
`save=False`, experimento MLflow `mimo_sys_explainability` (mesmo das
outras 4 arquiteturas, para comparar lado a lado).

Uso:
    cd notebooks/training
    uv run --extra ml python scripts/run_fourier_gnn_grid.py
"""

from __future__ import annotations

from mimo_sys.training import TrainConfig, train

EXPERIMENT = "mimo_sys_explainability"
SEEDS = [42, 43, 44]
MODEL_KWARGS = {"embed_dim": 32, "n_layers": 3, "ffn_hidden": 64}

VARIANTS = {
    "baseline": {},
    "revin": {"revin": "standard"},
    "augment": {"augment": True},
    "mixup": {"mixup_alpha": 0.2},
    "huber": {"loss": "huber"},
    "spectral": {"spectral_weight": 0.1},
    "ema": {"ema_decay": 0.999},
    "cosine_clip": {"scheduler": "cosine_warmup", "gradient_clip_val": 1.0},
    "combinado": {
        "revin": "standard",
        "augment": True,
        "spectral_weight": 0.1,
        "ema_decay": 0.999,
    },
}


def jobs() -> list[TrainConfig]:
    """One `TrainConfig` per (variant, seed)."""
    return [
        TrainConfig(
            model="fourier_gnn",
            model_kwargs=MODEL_KWARGS,
            input_window=24,
            output_window=12,
            seed=seed,
            experiment=EXPERIMENT,
            run_name=f"B_fourier_gnn__{variant}__seed{seed}",
            save=False,
            **overrides,
        )
        for variant, overrides in VARIANTS.items()
        for seed in SEEDS
    ]


def main() -> None:
    """Run every job, printing progress."""
    cfgs = jobs()
    print(f"{len(cfgs)} treinos")
    for i, cfg in enumerate(cfgs, 1):
        print(f"[{i}/{len(cfgs)}] {cfg.run_name}", flush=True)
        result = train(cfg)
        print("  r2_medio:", result.metrics["r2"].mean().round(3))


if __name__ == "__main__":
    main()
