"""Training toolkit for the MIMO forecasters.

Signal preparation and reconstruction live in
`mimo_sys.preprocessors.TimeSeriesPreprocessor`; this package holds
what only happens while training (augmentations, losses, RevIN,
callbacks, contrastive pretraining) and the `train` entry point.
Methodology: `docs/6_tecnicas-de-treino.md`.
"""

from mimo_sys.training.ensemble import (
    average_predictions,
    fit_ensemble_weights,
    greedy_forward_selection,
    select_ensemble_subset,
    weighted_predictions,
)
from mimo_sys.training.explainability import (
    attention_by_lag,
    gradient_saliency,
    top_lags,
)
from mimo_sys.training.train import (
    ARCHITECTURES,
    BASELINE_PREPROCESSOR,
    TrainConfig,
    TrainResult,
    add_derived_features,
    build_model,
    train,
)

__all__ = [
    "ARCHITECTURES",
    "BASELINE_PREPROCESSOR",
    "TrainConfig",
    "TrainResult",
    "add_derived_features",
    "attention_by_lag",
    "average_predictions",
    "build_model",
    "fit_ensemble_weights",
    "gradient_saliency",
    "greedy_forward_selection",
    "select_ensemble_subset",
    "top_lags",
    "train",
    "weighted_predictions",
]
