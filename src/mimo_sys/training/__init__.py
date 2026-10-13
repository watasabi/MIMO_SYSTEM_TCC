"""Training toolkit for the MIMO forecasters.

Signal preparation and reconstruction live in
`mimo_sys.preprocessors.TimeSeriesPreprocessor`; this package holds
what only happens while training (augmentations, losses, RevIN,
callbacks, contrastive pretraining) and the `train` entry point.
Methodology: `docs/6_tecnicas-de-treino.md`.
"""

from mimo_sys.training.train import (
    ARCHITECTURES,
    BASELINE_PREPROCESSOR,
    TrainConfig,
    TrainResult,
    build_model,
    train,
)

__all__ = [
    "ARCHITECTURES",
    "BASELINE_PREPROCESSOR",
    "TrainConfig",
    "TrainResult",
    "build_model",
    "train",
]
