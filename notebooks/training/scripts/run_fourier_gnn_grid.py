"""Grade de técnicas de treino para o FourierGNN (W=24, 3 seeds).

Uso:
    cd notebooks/training
    uv run --extra ml python scripts/run_fourier_gnn_grid.py
"""

from __future__ import annotations

import mlflow

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


def done_run_names() -> set[str]:
    """Run names already logged in `EXPERIMENT` (completed or not).

    Lets a rerun skip what an earlier, interrupted pass already did.
    """
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return set()
    df = mlflow.search_runs([exp.experiment_id], max_results=5000)
    if "tags.mlflow.runName" not in df:
        return set()
    return set(df["tags.mlflow.runName"].dropna())


def main() -> None:
    """Run pending jobs, printing progress.

    Skips `run_name`s already present in `EXPERIMENT`, so re-running
    after an interruption resumes instead of retraining everything.
    """
    all_jobs = jobs()
    done = done_run_names()
    pending = [cfg for cfg in all_jobs if cfg.run_name not in done]
    print(
        f"{len(all_jobs)} treinos, {len(done)} já feitos, "
        f"{len(pending)} pendentes"
    )
    for i, cfg in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {cfg.run_name}", flush=True)
        result = train(cfg)
        print("  r2_medio:", result.metrics["r2"].mean().round(3))


if __name__ == "__main__":
    main()
