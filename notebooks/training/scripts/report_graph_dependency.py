"""Evidência de dependência entre as séries (justifica o grafo).

Só no treino. Para cada saída i e cada série j ≠ i:
- Granger condicional: VAR com as 7 séries (ordem por AIC) e teste de
  Wald de que todos os lags de j somem da equação de i;
- correlação cruzada após pré-branqueamento (AR por série, ordem por
  AIC), com o lag do pico em 0..24 h.

Uso (da raiz do repo):
    uv run python notebooks/training/scripts/report_graph_dependency.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import ar_select_order

from mimo_sys.training.data import (
    DATA_PROCESSED,
    INPUT_COLS,
    OUTPUT_COLS,
    PROJECT_ROOT,
)

OUT = PROJECT_ROOT / "outputs" / "graph_dependency.json"
MAX_LAG = 24

df = pd.read_parquet(DATA_PROCESSED / "mimo_full_sample_train.parquet")
cols = list(INPUT_COLS) + list(OUTPUT_COLS)
x = df[cols].astype(float)
x = (x - x.mean()) / x.std()

var = VAR(x).fit(maxlags=MAX_LAG, ic="aic")
granger = {}
for i in OUTPUT_COLS:
    for j in cols:
        if j == i:
            continue
        t = var.test_causality(i, [j], kind="wald")
        granger[f"{j} -> {i}"] = float(t.pvalue)


def whiten(s: pd.Series) -> np.ndarray:
    sel = ar_select_order(s.to_numpy(), maxlag=MAX_LAG, ic="aic")
    return sel.model.fit().resid


white = {c: whiten(x[c]) for c in cols}
ccf = {}
for i in OUTPUT_COLS:
    for j in cols:
        if j == i:
            continue
        a, b = white[j], white[i]
        n = min(len(a), len(b))
        a, b = a[-n:], b[-n:]
        r = [np.corrcoef(a[: n - k], b[k:])[0, 1] for k in range(MAX_LAG + 1)]
        k = int(np.argmax(np.abs(r)))
        ccf[f"{j} -> {i}"] = {
            "lag_h": k,
            "r": round(float(r[k]), 3),
            "significativo": bool(abs(r[k]) > 1.96 / np.sqrt(n)),
        }

out = {"var_lag_order": int(var.k_ar), "granger_p": granger, "ccf": ccf}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"VAR ordem {var.k_ar}, n = {len(x)}")
for k, p in granger.items():
    c = ccf[k]
    print(f"{k:40s} p={p:.2e}  ccf r={c['r']:+.3f} lag={c['lag_h']}h")
