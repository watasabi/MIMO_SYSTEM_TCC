"""Lança toda a validação cruzada temporal, com etapas em paralelo.

Só usa a biblioteca padrão: o lançador fica leve, e a RAM vai para os
treinos. Cada modelo, cenário e fold roda `tune` e depois `final` em
processos próprios de `run_cv_benchmark.py`. É retomável: o que já terminou
é pulado, e etapas interrompidas são refeitas no final, uma por vez.

Uso (da raiz do repo):
    uv run --extra ml --extra tune --extra boost python \
        notebooks/training/scripts/run_cv_all.py [processos] [tentativas]
"""

from __future__ import annotations

import fcntl
import os
import subprocess  # noqa: S404
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT = Path(__file__).with_name("run_cv_benchmark.py")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# `MIMO_CV_OUT` separa testes curtos da rodada oficial.
OUT = Path(os.environ.get("MIMO_CV_OUT", PROJECT_ROOT / "outputs" / "cv"))
SCENARIOS = ("full", "pumps")
N_FOLDS = 5
N_TRIALS = 30
WORKERS = 2
FINAL_SEEDS = (42, 43, 44)
MIN_AVAILABLE_MB = 1500
MODELS = (
    "gotas",
    "mlp",
    "lstm",
    "gru",
    "xgboost",
    "catboost",
    "mpnn",
    "sgn",
    "stemgnn",
    "seq2seq_latent_gnn",
    "fourier_gnn",
    "armax",
)
# Menos arenas de memória e poucas threads por processo: o paralelismo
# vem de rodar vários processos.
CHILD_ENV = {
    "MALLOC_ARENA_MAX": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}
Job = tuple[str, str, int]


def pred_path(model: str, scenario: str, fold: int, seed: int) -> Path:
    return OUT / "preds" / f"{model}_{scenario}_f{fold}_s{seed}.npz"


def seeds_of(model: str) -> tuple[int, ...]:
    return FINAL_SEEDS[:1] if model == "armax" else FINAL_SEEDS


def finished(job: Job) -> bool:
    model, scenario, fold = job
    return all(
        pred_path(model, scenario, fold, seed).exists()
        for seed in seeds_of(model)
    )


def available_mb() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable not found")


def step(name: str, job: Job, n_trials: int) -> int:
    """Roda uma etapa quando houver memória para ela."""
    model, scenario, fold = job
    while available_mb() < MIN_AVAILABLE_MB:
        time.sleep(10)
    cmd = [sys.executable, str(SCRIPT), name, model, scenario, str(fold)]
    if name == "tune":
        cmd.append(str(n_trials))
    label = f"{name}_{model}_{scenario}_f{fold}"
    began = time.time()
    with (OUT / "logs" / f"{label}.log").open("a", encoding="utf-8") as log:
        code = subprocess.run(  # noqa: S603
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, **CHILD_ENV},
        ).returncode
    minutes = (time.time() - began) / 60
    print(f"{label}: código {code}, {minutes:.1f} min", flush=True)
    return code


def run_job(job: Job, n_trials: int) -> bool:
    """Busca e avaliação final de um modelo, cenário e fold."""
    if step("tune", job, n_trials) != 0:
        return False
    return step("final", job, n_trials) == 0 and finished(job)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else WORKERS
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else N_TRIALS  # noqa: PLR2004
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    # Duas rodadas ao mesmo tempo esgotariam a memória da máquina.
    lock = (OUT / "run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Já existe uma rodada em andamento.") from None
    jobs = [
        (model, scenario, fold)
        for model in MODELS
        for scenario in SCENARIOS
        for fold in range(1, N_FOLDS + 1)
    ]
    pending = [job for job in jobs if not finished(job)]
    print(f"{len(pending)} de {len(jobs)} etapas pendentes", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda j: run_job(j, n_trials), pending))
    failed = [j for j, ok in zip(pending, results, strict=True) if not ok]
    for job in failed:
        print("refazendo", job, flush=True)
        run_job(job, n_trials)
    missing = [job for job in jobs if not finished(job)]
    print(f"concluído; faltam {len(missing)}: {missing}", flush=True)


if __name__ == "__main__":
    main()
