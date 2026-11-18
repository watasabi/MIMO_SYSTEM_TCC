"""Mostra o andamento da validação cruzada, sem carregar bibliotecas pesadas.

Uso (da raiz do repo):
    watch -n 30 .venv/bin/python notebooks/training/scripts/run_cv_status.py
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import datetime, timedelta

from run_cv_all import (
    MODELS,
    N_FOLDS,
    N_TRIALS,
    OUT,
    SCENARIOS,
    available_mb,
    finished,
)

RECENT_MINUTES = 20
DONE = "('COMPLETE', 'PRUNED', 'FAIL')"


def main() -> None:
    n_studies = len(SCENARIOS) * N_FOLDS
    total = len(MODELS) * n_studies * N_TRIALS
    db = OUT / "optuna.db"
    per_study: Counter[str] = Counter()
    recent = 0
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        since = datetime.now() - timedelta(minutes=RECENT_MINUTES)
        rows = con.execute(
            "SELECT s.study_name, t.datetime_complete FROM trials t "
            "JOIN studies s ON s.study_id = t.study_id "
            f"WHERE t.state IN {DONE}"
        ).fetchall()
        con.close()
        for name, when in rows:
            per_study[name] += 1
            recent += when is not None and when >= str(since)
    print(f"{'Modelo':20s} {'Tentativas':>12s} {'Etapas finais':>14s}")
    done_trials = 0
    for model in MODELS:
        count = 0
        final = 0
        for scenario in SCENARIOS:
            for fold in range(1, N_FOLDS + 1):
                name = f"{model}_{scenario}_f{fold}"
                count += min(per_study[name], N_TRIALS)
                final += finished((model, scenario, fold))
        done_trials += count
        print(
            f"{model:20s} {count:5d}/{n_studies * N_TRIALS:<6d} "
            f"{final:8d}/{n_studies}"
        )
    rate = recent / RECENT_MINUTES
    print(f"\nTotal: {done_trials}/{total} ({100 * done_trials / total:.1f}%)")
    print(f"Ritmo nos últimos {RECENT_MINUTES} min: {rate:.1f} tentativas/min")
    if rate > 0:
        hours = (total - done_trials) / rate / 60
        print(f"Tempo restante aproximado: {hours:.1f} h")
    print(f"Memória disponível: {available_mb() / 1024:.1f} GB")


if __name__ == "__main__":
    main()
