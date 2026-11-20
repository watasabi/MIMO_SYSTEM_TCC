"""Andamento da triagem do GOTAS v2, sem carregar bibliotecas pesadas.

Lê `outputs/cv/gotas_v2_screen.jsonl` (gravado por `run_gotas_v2.py`) e
mostra, por variante e cenário, a validação normal, a validação sob
estresse, a parte explicável (sem o resíduo MLP; `-` nas linhas
antigas, sem essa medida) e a diferença pareada contra a referência,
nos mesmos folds e sementes.

Uso (da raiz do repo):
    S=notebooks/training/scripts/run_gotas_v2_status.py
    watch -n 60 .venv/bin/python $S
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "cv"
    / "gotas_v2_screen.jsonl"
)
SEEDS_PER_FOLD = 3
N_FOLDS = 5


def report() -> None:
    """Média por variante e cenário, e diferença pareada contra a base.

    A diferença usa só os (fold, semente) presentes nos dois lados, para
    uma variante incompleta não ser comparada com a base inteira. A
    base de cada variante nova (GOTAS-W, -LPV, -A) é comparada com o GOTAS
    atual (`gotas base`).
    """
    if not RESULTS.exists():
        print("Nenhum resultado ainda.")
        return
    runs: dict = {}
    for line in RESULTS.open(encoding="utf-8"):
        r = json.loads(line)
        key = (r["base"], r["variante"], r["cenario"])
        runs.setdefault(key, {})[(r["fold"], r["semente"])] = (
            r["val_r2"],
            r["estresse_r2"],
            r.get("explicavel_r2"),
        )
    total = N_FOLDS * SEEDS_PER_FOLD
    print(
        f"{'base':8s} {'variante':20s} {'cenário':7s} {'treinos':>8s} "
        f"{'val':>6s} {'estresse':>8s} {'explic.':>7s}  diferença pareada"
    )
    for (base, variant, scenario), d in sorted(runs.items()):
        if base != "gotas" and variant == "base":
            ref_key = ("gotas", "base", scenario)
        else:
            ref_key = (base, "base", scenario)
        ref = (
            runs.get(ref_key) if ref_key != (base, variant, scenario) else None
        )
        val = np.mean([v[0] for v in d.values()])
        stress = np.mean([v[1] for v in d.values()])
        explained = [v[2] for v in d.values() if v[2] is not None]
        explic = f"{np.mean(explained):7.3f}" if explained else f"{'-':>7s}"
        delta = ""
        if ref:
            common = sorted(set(d) & set(ref))
            dv = np.mean([d[c][0] - ref[c][0] for c in common])
            ds = np.mean([d[c][1] - ref[c][1] for c in common])
            delta = (
                f"vs {ref_key[0]} {ref_key[1]}: val {dv:+.3f}, "
                f"estresse {ds:+.3f} (n={len(common)})"
            )
        print(
            f"{base:8s} {variant:20s} {scenario:7s} {len(d):>3d}/{total:<4d} "
            f"{val:6.3f} {stress:8.3f} {explic}  {delta}"
        )


if __name__ == "__main__":
    report()
