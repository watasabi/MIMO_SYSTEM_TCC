"""Splits do dataset MIMO completo (2208 amostras, SANEPAR_COMPLETO.txt).

Ao contrário de `01_mimo_dataset_from_papper.ipynb` (que reproduz
apenas o recorte de 1102 amostras usado no artigo), este script usa a
série inteira e gera dois esquemas de split para treino/avaliação:

- K-fold de série temporal (`TimeSeriesSplit`) + um holdout de teste
  final, para validação cruzada respeitando a ordem temporal.
- Split simples treino/validação/teste cronológico (sem embaralhar),
  no mesmo espírito do que já era usado em
  `notebooks/training/01_seq2seq_attention.ipynb`.

Ver `docs/changes/2026-10-02-full-dataset-splits.md`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

DATA_RAW = Path("../../data/raw")
DATA_PROCESSED = Path("../../data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

SANEPAR_PATH = (
    DATA_RAW
    / "WRSO and NSWRSO_draft_all"
    / "Multi_objective_for_MIMO_systems"
    / "MOGWO"
    / "SANEPAR_COMPLETO.txt"
)

COLUMNS = [
    "freq_b1",
    "freq_b2",
    "freq_b3",
    "vazao_entrada",
    "vazao_distribuicao",
    "nivel_res",
    "pressao",
]

N_SPLITS = 5
TEST_SIZE = 200  # amostras finais reservadas como holdout de teste
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # restante (0.15) vira teste

df = pd.DataFrame(np.loadtxt(SANEPAR_PATH), columns=COLUMNS)
df.index.name = "sample_idx"
print("dataset completo:", df.shape)

# --- Esquema 1: TimeSeriesSplit (k-fold) + holdout de teste ---------------
# O holdout final nunca entra nos folds, só é usado na avaliação final.
holdout_test = df.iloc[-TEST_SIZE:].reset_index()
cv_pool = df.iloc[:-TEST_SIZE]

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
fold_records = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(cv_pool)):
    fold_records.append(
        pd.DataFrame(
            {
                "fold": fold,
                "sample_idx": cv_pool.index[train_idx],
                "role": "train",
            }
        )
    )
    fold_records.append(
        pd.DataFrame(
            {
                "fold": fold,
                "sample_idx": cv_pool.index[val_idx],
                "role": "val",
            }
        )
    )
    print(
        f"fold {fold}: treino={len(train_idx)} amostras "
        f"({cv_pool.index[train_idx[0]]}:{cv_pool.index[train_idx[-1]]}), "
        f"val={len(val_idx)} amostras "
        f"({cv_pool.index[val_idx[0]]}:{cv_pool.index[val_idx[-1]]})"
    )

kfold_assignments = pd.concat(fold_records, ignore_index=True)

kfold_path = DATA_PROCESSED / "mimo_full_kfold_assignments.parquet"
holdout_path = DATA_PROCESSED / "mimo_full_kfold_test.parquet"
kfold_assignments.to_parquet(kfold_path)
holdout_test.to_parquet(holdout_path)
print("salvo:", kfold_path)
print("salvo:", holdout_path)
print("holdout de teste:", holdout_test.shape)

# --- Esquema 2: split simples treino/validação/teste cronológico ---------
n = len(df)
train_end = int(n * TRAIN_FRAC)
val_end = train_end + int(n * VAL_FRAC)

train = df.iloc[:train_end].reset_index()
val = df.iloc[train_end:val_end].reset_index()
test = df.iloc[val_end:].reset_index()

print(
    "split simples -> treino:",
    train.shape,
    "val:",
    val.shape,
    "teste:",
    test.shape,
)

train_path = DATA_PROCESSED / "mimo_full_sample_train.parquet"
val_path = DATA_PROCESSED / "mimo_full_sample_val.parquet"
test_path = DATA_PROCESSED / "mimo_full_sample_test.parquet"
train.to_parquet(train_path)
val.to_parquet(val_path)
test.to_parquet(test_path)
print("salvo:", train_path)
print("salvo:", val_path)
print("salvo:", test_path)
