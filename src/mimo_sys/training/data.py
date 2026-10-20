"""Dataset constants, split loading and DataLoaders.

Signal preparation (scaling, detrend, windowing) lives in
`mimo_sys.preprocessors.TimeSeriesPreprocessor`, so that every forecast
can be reconstructed with its `inverse_transform`. This module only
loads the splits and wraps the windows the preprocessor returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

INPUT_COLS = ["freq_b1", "freq_b2", "freq_b3", "vazao_entrada"]
OUTPUT_COLS = ["vazao_distribuicao", "nivel_res", "pressao"]
FEATURE_COLS = INPUT_COLS + OUTPUT_COLS
N_NODES = len(FEATURE_COLS)
TARGET_IDX = [FEATURE_COLS.index(c) for c in OUTPUT_COLS]
INPUT_IDX = [FEATURE_COLS.index(c) for c in INPUT_COLS]
N_TARGETS = len(TARGET_IDX)

INPUT_WINDOW = 24
OUTPUT_WINDOW = 12
SEED = 42


def load_splits(
    prefix: str = "mimo_full_sample",
) -> dict[str, npt.NDArray]:
    """Raw `FEATURE_COLS` arrays of the chronological 70/15/15 split.

    Args:
        prefix: File prefix in `data/processed/`; reads
            `<prefix>_{train,val,test}.parquet`.

    Returns:
        `{"train", "val", "test"}` -> `(n_rows, N_NODES)` float arrays.
    """
    return {
        name: pd.read_parquet(DATA_PROCESSED / f"{prefix}_{name}.parquet")[
            FEATURE_COLS
        ].to_numpy(dtype=float)
        for name in ("train", "val", "test")
    }


def with_future_exog(past: npt.NDArray, future: npt.NDArray) -> npt.NDArray:
    """Append the known future exogenous inputs to input windows.

    Only `INPUT_IDX` (`freq_b1..3`, `vazao_entrada`) is known ahead of
    time; every other column of `future` — the outputs, and any derived
    feature built from them (e.g. the mass-balance integral) — is
    zeroed, so the horizon never leaks into the input.

    Args:
        past: `(n, input_window, n_cols)` input windows.
        future: `(n, output_window, n_cols)` the same rows' horizon,
            all columns, in the same (preprocessed) space.

    Returns:
        `(n, input_window + output_window, n_cols)`.
    """
    known = np.zeros_like(future)
    known[..., INPUT_IDX] = future[..., INPUT_IDX]
    return np.concatenate([past, known], axis=1)


class WindowDataset(Dataset):
    """`(x, y)` windows already produced by the preprocessor.

    Args:
        X: `(n, input_window, n_nodes)`.
        y: `(n, output_window, n_targets)`.
    """

    def __init__(self, X: npt.NDArray, y: npt.NDArray) -> None:
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_loaders(
    windows: dict[str, tuple[npt.NDArray, npt.NDArray]],
    batch_size: int = 32,
) -> dict[str, DataLoader]:
    """One DataLoader per split; only `train` is shuffled."""
    return {
        name: DataLoader(
            WindowDataset(X, y),
            batch_size=batch_size,
            shuffle=name == "train",
        )
        for name, (X, y) in windows.items()
    }
