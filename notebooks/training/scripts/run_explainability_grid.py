"""Grade de treinos das rotas de extração (A: W=3, B: W=24), 3 seeds.

Uso:
    cd notebooks/training
    uv run --extra ml python scripts/run_explainability_grid.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlflow

EXPERIMENT = "mimo_sys_explainability"
SEEDS = [42, 43, 44]
ROUTE_A_INPUT_WINDOW = 3
ROUTE_B_INPUT_WINDOW = 24

ROUTE_A = {
    "mpnn": {"hidden_size": 64, "message_dim": 2, "n_layers": 1},
    "sgn": {
        "hidden_size": 64,
        "message_dim": 2,
        "n_layers": 1,
        "noise_scale": 1.0,
    },
}
ROUTE_B = {
    "mpnn": {"hidden_size": 64, "message_dim": 8, "n_layers": 1},
    "sgn": {"hidden_size": 64, "message_dim": 8, "n_layers": 1},
    "stemgnn": {"hidden_size": 64},
    "seq2seq_latent_gnn": {"hidden_size": 64},
}

VARIANTS = {
    "baseline": {},
    "revin": {"revin": "standard"},
    "augment": {"augment": True},
    "mixup": {"mixup_alpha": 0.2},
    "huber": {"loss": "huber"},
    "spectral": {"spectral_weight": 0.1},
    "ema": {"ema_decay": 0.999},
    "cosine_clip": {"scheduler": "cosine_warmup", "gradient_clip_val": 1.0},
    "contrastive": {"pretrain_epochs": 30, "freeze_encoder_epochs": 10},
    "combinado": {
        "revin": "standard",
        "augment": True,
        "spectral_weight": 0.1,
        "ema_decay": 0.999,
    },
}
# MPNN/SGN só têm encoder pretreinável com n_layers >= 2 (ver
# encode() em mimo_sys.architectures); ambos usam n_layers=1 nesta
# grade. StemGNN nunca expõe encode() (sem corte limpo de encoder —
# ver src/mimo_sys/architectures/README.md), em qualquer configuração.
NO_CONTRASTIVE = {"mpnn", "sgn", "stemgnn"}

SCRIPT = """
import json, sys
from mimo_sys.training import TrainConfig, train
cfg = TrainConfig(**json.loads(sys.argv[1]))
print("RUN_ID", train(cfg).run_id)
"""


def jobs() -> list[dict]:
    """One `TrainConfig` dict per (rota, modelo, variante, seed)."""
    out = []
    for route_name, route, input_window in (
        ("A", ROUTE_A, ROUTE_A_INPUT_WINDOW),
        ("B", ROUTE_B, ROUTE_B_INPUT_WINDOW),
    ):
        for model, kwargs in route.items():
            for variant, overrides in VARIANTS.items():
                if variant == "contrastive" and model in NO_CONTRASTIVE:
                    continue
                for seed in SEEDS:
                    cfg = {
                        "model": model,
                        "model_kwargs": kwargs,
                        "input_window": input_window,
                        "output_window": (
                            1 if input_window == ROUTE_A_INPUT_WINDOW else 12
                        ),
                        "seed": seed,
                        "experiment": EXPERIMENT,
                        "run_name": (
                            f"{route_name}_{model}__{variant}__seed{seed}"
                        ),
                        "save": False,
                        **overrides,
                    }
                    if input_window == ROUTE_A_INPUT_WINDOW and model in {
                        "mpnn",
                        "sgn",
                    }:
                        cfg["message_l1_weight"] = 1e-2
                    out.append(cfg)
    return out


def done_run_names() -> set[str]:
    """Run names already logged in `EXPERIMENT` (completed or not).

    Lets a rerun skip what an earlier, interrupted pass already did —
    `main()` filters jobs against this before training anything.
    """
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return set()
    df = mlflow.search_runs([exp.experiment_id], max_results=5000)
    if "tags.mlflow.runName" not in df:
        return set()
    return set(df["tags.mlflow.runName"].dropna())


def run_job(cfg: dict) -> str:
    """Train one config in a subprocess; return its MLflow run_id."""
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(cfg)],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return result.stdout.split("RUN_ID")[-1].strip()


def main() -> None:
    """Run pending jobs, printing progress and collecting failures.

    Skips `run_name`s already present in `EXPERIMENT`, so re-running
    this script after an interruption (e.g. the machine was shut down)
    resumes instead of retraining everything.
    """
    all_jobs = jobs()
    done = done_run_names()
    pending = [cfg for cfg in all_jobs if cfg["run_name"] not in done]
    print(
        f"{len(all_jobs)} treinos, {len(done)} já feitos, "
        f"{len(pending)} pendentes"
    )
    failures = []
    for i, cfg in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {cfg['run_name']}", flush=True)
        try:
            run_id = run_job(cfg)
            print("  ok:", run_id)
        except subprocess.CalledProcessError as exc:
            print("  FALHOU:", exc.stderr[-2000:])
            failures.append(cfg["run_name"])
    print(f"\nfeito. falhas: {failures}")


if __name__ == "__main__":
    main()
