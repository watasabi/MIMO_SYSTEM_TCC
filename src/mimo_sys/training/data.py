"""Dataset constants, split loading and DataLoaders.

Signal preparation (scaling, detrend, windowing) lives in
`mimo_sys.preprocessors.TimeSeriesPreprocessor`, so that every forecast
can be reconstructed with its `inverse_transform`. This module only
loads the splits and wraps the windows the preprocessor returns.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def with_future_exog(
    past: npt.NDArray,
    future: npt.NDArray,
    known_idx: list[int] | None = None,
) -> npt.NDArray:
    """Append the known future exogenous inputs to input windows.

    Only `known_idx` (default `INPUT_IDX`: `freq_b1..3`,
    `vazao_entrada`) is known ahead of time; every other column of
    `future` — the outputs, and any derived feature built from them
    (e.g. the mass-balance integral) — is zeroed, so the horizon never
    leaks into the input.

    Args:
        past: `(n, input_window, n_cols)` input windows.
        future: `(n, output_window, n_cols)` the same rows' horizon,
            all columns, in the same (preprocessed) space.
        known_idx: Columns whose future values are real inputs.

    Returns:
        `(n, input_window + output_window, n_cols)`.
    """
    idx = INPUT_IDX if known_idx is None else known_idx
    known = np.zeros_like(future)
    known[..., idx] = future[..., idx]
    return np.concatenate([past, known], axis=1)


@dataclass(frozen=True)
class TargetLayout:
    """How the model's outputs add up to the `OUTPUT_COLS`.

    Output `i` (original units) is the sum of the predicted columns
    with `pred_output == i` plus the known-future columns with
    `known_output == i` (e.g. the seasonal base `y(t-24)`, already
    observed at the forecast origin). The baseline predicts the raw
    outputs directly: one predicted column per output, nothing known.

    Attributes:
        pred_cols: Columns the model predicts, in output order.
        pred_output: Output index of each predicted column.
        known_cols: Columns whose horizon values are known at the
            origin and are added back, not predicted.
        known_output: Output index of each known column.
        aux_cols: Columns the model also predicts that are not part of
            any output (e.g. the unknown future `vazao_entrada` that
            feeds the mass-balance cascade); returned separately.
    """

    pred_cols: tuple[int, ...] = tuple(TARGET_IDX)
    pred_output: tuple[int, ...] = tuple(range(N_TARGETS))
    known_cols: tuple[int, ...] = ()
    known_output: tuple[int, ...] = ()
    aux_cols: tuple[int, ...] = ()

    @property
    def model_cols(self) -> tuple[int, ...]:
        """Every column the model outputs, in order."""
        return self.pred_cols + self.aux_cols

    @property
    def decomposed(self) -> bool:
        """The outputs are rebuilt from components, not predicted."""
        return self.pred_cols != tuple(TARGET_IDX) or bool(self.known_cols)


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
