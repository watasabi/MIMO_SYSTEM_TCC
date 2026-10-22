"""One-call training of an architecture on the 70/15/15 split.

`train(TrainConfig(...))`:

1. seeds everything;
2. fits the `TimeSeriesPreprocessor` on the training rows only and
   windows train/val/test with it;
3. optionally pretrains the encoder contrastively;
4. trains with the chosen loss, augmentations, regularisers, EMA/SWA;
   checkpointing and early stopping watch `val_mse` (preprocessor
   space), not `val_loss`, so runs with different losses compare;
5. evaluates the best checkpoint on the test split, both per window
   (preprocessor space) and as a chained forecast reconstructed to the
   original scale with `inverse_transform`;
6. logs everything to one MLflow run and saves the `state_dict` plus the
   fitted preprocessor under `models/`.

Train one model per process when looping over architectures: training
all of them in the same process ran out of RAM before.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import joblib
import lightning as L
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from ewtpy import EWT1D
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from mimo_sys.architectures import (
    FourierGNN,
    LinearEdgeNet,
    MPNNForecaster,
    Seq2SeqLatentGNN,
    SpectralTFNet,
    StemGNN,
    SymbolicGraphNetwork,
)
from mimo_sys.preprocessors import TimeSeriesPreprocessor
from mimo_sys.training.augmentations import (
    ChannelMask,
    Compose,
    Jitter,
    MagnitudeWarp,
    Mixup,
    Scaling,
    SpanMask,
)
from mimo_sys.training.callbacks import (
    EMA,
    EncoderFreeze,
    MLflowModelCheckpoint,
)
from mimo_sys.training.contrastive import ContrastivePretrainer
from mimo_sys.training.data import (
    INPUT_COLS,
    INPUT_IDX,
    INPUT_WINDOW,
    MODELS_DIR,
    N_NODES,
    N_TARGETS,
    OUTPUT_COLS,
    OUTPUT_WINDOW,
    SEED,
    TARGET_IDX,
    TargetLayout,
    load_splits,
    make_loaders,
    with_future_exog,
)
from mimo_sys.training.evaluation import (
    band_table,
    chained_forecast,
    log_figure,
    metrics_table,
    plot_series,
    plot_spectrum,
)
from mimo_sys.training.features import (
    add_ewma_feature,
    add_mass_balance_feature,
    atrous_components,
    bad_sample_mask,
    ewt_online_components,
    seasonal_components,
    seasonal_median,
)
from mimo_sys.training.forecaster import Forecaster, Scheduler
from mimo_sys.training.integrator import LeakyIntegratorWrapper
from mimo_sys.training.losses import (
    build_loss,
    decaying_freq_weights,
    seasonal_naive_scale,
)
from mimo_sys.training.mass_balance_cascade import fit_mass_balance
from mimo_sys.training.output_heads import PerOutputHeadWrapper
from mimo_sys.training.scalers import RevINMode, RevINWrapper

ARCHITECTURES: dict[str, type[nn.Module]] = {
    "mpnn": MPNNForecaster,
    "sgn": SymbolicGraphNetwork,
    "stemgnn": StemGNN,
    "seq2seq_latent_gnn": Seq2SeqLatentGNN,
    "fourier_gnn": FourierGNN,
    "linear_edge_net": LinearEdgeNet,
    "spectral_tf_net": SpectralTFNet,
}

Decomposition = Literal["atrous", "ewt", "seasonal24"]
ComponentScale = Literal["unit", "parent"]
SEASON = 24

# Baseline of notebooks 04/05: MinMax fitted on train, nothing else.
BASELINE_PREPROCESSOR: dict[str, Any] = {
    "scaler": "minmax",
    "remove_outliers": False,
    "apply_ewt": False,
    "detrend": False,
    "apply_filter": False,
}


def build_model(
    name: str,
    input_window: int = INPUT_WINDOW,
    output_window: int = OUTPUT_WINDOW,
    n_nodes: int = N_NODES,
    target_idx: list[int] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate an architecture with the project geometry.

    Args:
        input_window: Lag window. The symbolic-regression route
            (`MPNNForecaster`/`SymbolicGraphNetwork` + `message_l1`)
            needs this small (e.g. 3): `message_fn` sees
            `2 * input_window` scalars, and the search degrades past a
            handful — see `docs/4_rotas-de-extracao.md`.
        output_window: Forecast horizon.
        n_nodes: Number of input series; `N_NODES + 1` when
            `TrainConfig.mass_balance_feature` adds the integral
            column (see `features.add_mass_balance_feature`).
        target_idx: Nodes the model predicts (default the 3 outputs;
            their components with `TrainConfig.target_decomposition`).
    """
    geometry = {
        "n_nodes": n_nodes,
        "input_window": input_window,
        "output_window": output_window,
        "target_idx": list(TARGET_IDX if target_idx is None else target_idx),
    }
    return ARCHITECTURES[name](**geometry, **kwargs)


@dataclass
class TrainConfig:
    """Everything a training run depends on; logged as MLflow params."""

    model: str = "seq2seq_latent_gnn"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    input_window: int = INPUT_WINDOW
    output_window: int = OUTPUT_WINDOW
    preprocessor: dict[str, Any] = field(
        default_factory=lambda: dict(BASELINE_PREPROCESSOR)
    )
    mass_balance_feature: bool = False
    mass_balance_window: int | None = None
    mass_balance_ewma_span: float | None = None
    revin: RevINMode | None = None
    leaky_integrator_idx: list[int] | None = None
    leaky_integrator_init: float = 0.98
    per_output_heads: bool = False
    per_output_heads_hidden: int = 16
    uncertainty_weighted: bool = False
    output_weights: tuple[float, ...] | None = None
    loss: str = "mse"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    momentum: float = 0.9
    scheduler: Scheduler = "plateau"
    warmup_epochs: int = 5
    max_epochs: int = 200
    patience: int = 20
    batch_size: int = 32
    gradient_clip_val: float | None = None
    augment: bool = False
    augment_p: float = 0.5
    mixup_alpha: float = 0.0
    spectral_weight: float = 0.0
    spectral_cutoff_period: float | None = None
    spectral_decay: float = 2.0
    flow_integral_weight: float = 0.0
    flow_integral_idx: int = 0
    cascade_weight: float = 0.0
    future_exog: bool = False
    future_known: tuple[str, ...] = tuple(INPUT_COLS)
    # Output decomposition (`docs/changes/2026-10-22-decomposicao-saidas.md`):
    # causal components of each output appended as nodes; the model
    # predicts them and the outputs are their exact sum.
    target_decomposition: Decomposition | None = None
    decomposition_levels: int = 4
    ewt_window: int = 168
    component_scale: ComponentScale = "unit"
    component_loss_weight: float = 1.0
    components_as_targets: bool = True
    # Sensor failures in the training split (`features.bad_sample_mask`):
    # interpolated for the scaler/derived features, and every training
    # window whose target horizon touches one is dropped. Val/test are
    # left untouched.
    clean_train: bool = False
    # `pumps` scenario: the graph also forecasts the unknown future
    # `vazao_entrada` (an extra target node), and the cascade uses it
    # instead of the seasonal-naive guess.
    predict_inflow: bool = False
    # Causal estimate of the time-of-day pressure setpoint (median of
    # the same hour over the last 7 days, `features.seasonal_median`),
    # appended as a node known over the whole horizon.
    setpoint_feature: bool = False
    accumulate_grad_batches: int = 1
    message_l1_weight: float = 0.0
    backcast_weight: float = 1.0
    teacher_forcing_ratio: float = 0.0
    ema_decay: float | None = None
    swa_lr: float | None = None
    pretrain_epochs: int = 0
    pretrain_temperature: float = 0.2
    freeze_encoder_epochs: int = 0
    seed: int = SEED
    experiment: str = "mimo_sys_training"
    run_name: str | None = None
    save: bool = True
    log_plots: bool = False
    accelerator: str = "auto"
    precision: str = "32-true"
    progress: bool = False
    data_prefix: str = "mimo_full_sample"

    def flat_params(self) -> dict[str, Any]:
        """Config as MLflow-friendly scalars."""
        out = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                out.update({f"{key}.{k}": v for k, v in value.items()})
            else:
                out[key] = value
        return out


@dataclass
class TrainResult:
    """What a run produced."""

    module: Forecaster
    preprocessor: TimeSeriesPreprocessor
    metrics: pd.DataFrame
    bands: pd.DataFrame
    y_true: npt.NDArray
    y_pred: npt.NDArray
    best_val_mse: float
    best_val_r2: float
    run_id: str


def _augmentations(cfg: TrainConfig) -> Compose | None:
    augs: list = []
    if cfg.augment:
        # `y`'s own columns are always `0..len(TARGET_IDX)-1` (plus a
        # trailing cascade-only column when `cascade_weight > 0`, see
        # `_windows`) — not `TARGET_IDX`, which indexes `FEATURE_COLS`.
        y_target_idx = list(range(len(TARGET_IDX)))
        augs += [
            Jitter(0.02),
            Scaling(y_target_idx, 0.1),
            MagnitudeWarp(y_target_idx, 0.1),
            SpanMask(0.2),
            ChannelMask(INPUT_IDX, 0.1),
        ]
    if cfg.mixup_alpha > 0:
        augs.append(Mixup(cfg.mixup_alpha))
    return Compose(augs, p=cfg.augment_p) if augs else None


def add_derived_features(
    cfg: TrainConfig, splits: dict[str, npt.NDArray]
) -> dict[str, npt.NDArray]:
    """Re-split after appending `cfg`'s configured extra feature columns.

    Features are computed on the full, chronologically concatenated
    series (so running sums/averages don't reset at a split boundary)
    then sliced back — exposed separately from `_windows` so callers
    evaluating a trained model (e.g. `chained_forecast` on a raw
    segment) can reproduce the exact same columns.
    """
    if cfg.clean_train:
        splits = {**splits, "train": clean_rows(splits["train"])}
    if not (
        cfg.mass_balance_feature
        or cfg.target_decomposition is not None
        or cfg.setpoint_feature
    ):
        return splits
    full = derive_columns(
        cfg,
        np.concatenate([splits["train"], splits["val"], splits["test"]]),
        splits["train"],
    )
    train_end = len(splits["train"])
    val_end = train_end + len(splits["val"])
    return {
        "train": full[:train_end],
        "val": full[train_end:val_end],
        "test": full[val_end:],
    }


def clean_rows(raw: npt.NDArray) -> npt.NDArray:
    """Failed rows (`bad_sample_mask`) replaced by linear interpolation."""
    bad = bad_sample_mask(raw)
    good = np.flatnonzero(~bad)
    out = raw.copy()
    for col in range(raw.shape[1]):
        out[bad, col] = np.interp(np.flatnonzero(bad), good, raw[good, col])
    return out


def derive_columns(
    cfg: TrainConfig, full: npt.NDArray, train_raw: npt.NDArray
) -> npt.NDArray:
    """Append `cfg`'s derived columns to chronological raw rows.

    Every column is causal (row `t` only uses rows `<= t`), so this is
    also the production step: run on the rows received so far (unknown
    horizon cells NaN), it yields the same values as on the full
    series. Fitted pieces (EWT band boundaries) come from `train_raw`.
    """
    if cfg.mass_balance_feature:
        full = add_mass_balance_feature(
            full,
            INPUT_COLS.index("vazao_entrada"),
            TARGET_IDX[0],
            window=cfg.mass_balance_window,
        )
    if cfg.mass_balance_ewma_span:
        full = add_ewma_feature(
            full, full.shape[1] - 1, span=cfg.mass_balance_ewma_span
        )
    if cfg.setpoint_feature:
        pressure = full[:, TARGET_IDX[2]]
        full = np.concatenate([full, seasonal_median(pressure)[:, None]], 1)
    if cfg.target_decomposition is not None:
        # Every component column is causal (row t only uses rows <= t),
        # so computing it on the concatenated series leaks nothing and
        # is exactly what production computes row by row.
        comps = [
            _components(cfg, full[:, col], train_raw[:, col])
            for col in TARGET_IDX
        ]
        full = np.concatenate([full, *comps], axis=1)
    return full


def _components(
    cfg: TrainConfig, series: npt.NDArray, train_series: npt.NDArray
) -> npt.NDArray:
    """`(n, n_components)` causal components of one output column."""
    if cfg.target_decomposition == "atrous":
        return atrous_components(series, cfg.decomposition_levels)
    if cfg.target_decomposition == "ewt":
        # Band boundaries come from the training rows only.
        _, _, bounds = EWT1D(train_series, N=cfg.decomposition_levels)
        return ewt_online_components(series, bounds, window=cfg.ewt_window)
    return seasonal_components(series, SEASON)


def _n_components(cfg: TrainConfig) -> int:
    if cfg.target_decomposition == "atrous":
        return cfg.decomposition_levels + 1
    if cfg.target_decomposition == "ewt":
        return cfg.decomposition_levels
    return 2


def _n_base_nodes(cfg: TrainConfig) -> int:
    """Nodes before the component columns (raw + mass-balance ones)."""
    return (
        N_NODES
        + bool(cfg.mass_balance_feature)
        + bool(cfg.mass_balance_ewma_span)
        + bool(cfg.setpoint_feature)
    )


def setpoint_column(cfg: TrainConfig) -> int:
    """Column of the pressure-setpoint estimate (`setpoint_feature`)."""
    return _n_base_nodes(cfg) - 1


def component_parents(cfg: TrainConfig) -> dict[int, int]:
    """Component column -> its output's raw column."""
    if cfg.target_decomposition is None:
        return {}
    k, base = _n_components(cfg), _n_base_nodes(cfg)
    return {
        base + i * k + j: TARGET_IDX[i]
        for i in range(N_TARGETS)
        for j in range(k)
    }


def target_layout(cfg: TrainConfig) -> TargetLayout:
    """Which columns the model predicts and how they add up."""
    aux = (INPUT_COLS.index("vazao_entrada"),) if cfg.predict_inflow else ()
    if cfg.target_decomposition is None or not cfg.components_as_targets:
        return TargetLayout(aux_cols=aux)
    if aux:
        raise ValueError("predict_inflow is not combined with decomposition")
    k, base = _n_components(cfg), _n_base_nodes(cfg)
    cols = [[base + i * k + j for j in range(k)] for i in range(N_TARGETS)]
    if cfg.target_decomposition == "seasonal24":
        # [difference (predicted), y(t - 24) (known at the origin)].
        return TargetLayout(
            pred_cols=tuple(c[0] for c in cols),
            pred_output=tuple(range(N_TARGETS)),
            known_cols=tuple(c[1] for c in cols),
            known_output=tuple(range(N_TARGETS)),
        )
    return TargetLayout(
        pred_cols=tuple(c for group in cols for c in group),
        pred_output=tuple(i for i in range(N_TARGETS) for _ in range(k)),
    )


def known_columns(cfg: TrainConfig) -> list[int]:
    """Columns known in the horizon: `cfg.future_known` inputs, plus
    the known components of the layout (the seasonal base `y(t-24)`,
    observed at the origin while the horizon is <= 24 rows)."""
    setpoint = [setpoint_column(cfg)] if cfg.setpoint_feature else []
    return (
        [INPUT_COLS.index(c) for c in cfg.future_known]
        + setpoint
        + list(target_layout(cfg).known_cols)
    )


def _reconstruction(
    pp: TimeSeriesPreprocessor, layout: TargetLayout, cfg: TrainConfig
) -> dict[str, torch.Tensor]:
    """Affine maps (`scaled = a * raw + b`) to rebuild the outputs.

    `Forecaster` turns the predicted components back to raw units,
    adds the known ones read from the input horizon, and re-scales the
    sum with the output's own map — all exact and differentiable.
    """
    width = len(pp.scaler.scale_)
    zero = pp.scaler.transform(np.zeros((1, width)))[0]
    a = pp.scaler.transform(np.ones((1, width)))[0] - zero

    def ab(cols: tuple[int, ...] | list[int]) -> tuple[torch.Tensor, ...]:
        idx = list(cols)
        return (
            torch.as_tensor(a[idx], dtype=torch.float32),
            torch.as_tensor(zero[idx], dtype=torch.float32),
        )

    pred_a, pred_b = ab(layout.pred_cols)
    known_a, known_b = ab(layout.known_cols)
    out_a, out_b = ab(TARGET_IDX)
    return {
        "pred_a": pred_a,
        "pred_b": pred_b,
        "pred_output": torch.as_tensor(layout.pred_output, dtype=torch.long),
        "known_cols": torch.as_tensor(layout.known_cols, dtype=torch.long),
        "known_a": known_a,
        "known_b": known_b,
        "known_output": torch.as_tensor(layout.known_output, dtype=torch.long),
        "out_a": out_a,
        "out_b": out_b,
    }


def _windows(  # noqa: PLR0914 - linear on purpose
    cfg: TrainConfig, splits: dict[str, npt.NDArray]
) -> tuple[TimeSeriesPreprocessor, dict, npt.NDArray, npt.NDArray]:
    """Fit the preprocessor on train; window val/test with train context.

    Val/test segments start `cfg.input_window` rows early so their
    first target is the first row of the split.
    """
    train_raw = splits["train"]
    splits = add_derived_features(cfg, splits)

    # `y` = the 3 outputs, then (decomposed layout) the predicted
    # components, then with `cascade_weight > 0` the known future
    # `vazao_entrada` for `Forecaster`'s cascade loss — see
    # `mass_balance_cascade.mass_balance_cascade_forecast_torch`.
    feats = list(TARGET_IDX)
    layout = target_layout(cfg)
    if layout.decomposed:
        feats += list(layout.pred_cols)
    feats += list(layout.aux_cols)
    if cfg.cascade_weight > 0:
        feats += [INPUT_COLS.index("vazao_entrada")]

    pp = TimeSeriesPreprocessor(
        window_size=cfg.input_window,
        horizon=cfg.output_window,
        plot=False,
        **cfg.preprocessor,
    )
    X_tr, y_tr, processed = pp.fit_transform(splits["train"])
    if cfg.target_decomposition and cfg.component_scale == "parent":
        # N2: every component on its output's scale (own offset), so
        # the components keep their relative size in the loss.
        if not isinstance(pp.scaler, MinMaxScaler):
            raise ValueError("component_scale='parent' needs scaler=minmax")
        sc = pp.scaler
        for col, parent in component_parents(cfg).items():
            sc.scale_[col] = sc.scale_[parent]
            sc.min_[col] = -sc.data_min_[col] * sc.scale_[parent]
        X_tr, y_tr = pp.transform(splits["train"])
        processed = pp.transform_series(splits["train"])
    raw = np.concatenate([splits["train"], splits["val"], splits["test"]])
    train_end = len(splits["train"])
    val_end = train_end + len(splits["val"])
    val_seg = raw[train_end - cfg.input_window : val_end]
    test_seg = raw[val_end - cfg.input_window :]
    windows = {"train": (X_tr, y_tr)}
    for name, seg in (("val", val_seg), ("test", test_seg)):
        windows[name] = pp.transform(seg)
    if cfg.clean_train:
        # Window i's target is rows i + input_window .. + output_window:
        # a failure there would make an invented label. Failures in the
        # input part stay (interpolated), as production histories have
        # them too; dropping every window that touches one cost ~half of
        # the training windows (blocks of failures x 180-row windows).
        bad = np.concatenate([[0], np.cumsum(bad_sample_mask(train_raw))])
        starts = np.arange(len(windows["train"][0])) * pp.step_size
        first = starts + cfg.input_window
        keep = bad[first + cfg.output_window] - bad[first] == 0
        windows["train"] = tuple(a[keep] for a in windows["train"])
    for name, (X, y) in windows.items():
        X_seen = (
            with_future_exog(X, y, known_columns(cfg))
            if cfg.future_exog
            else X
        )
        windows[name] = (X_seen, y[:, :, feats])
    return pp, windows, test_seg, processed


def train(  # noqa: PLR0912, PLR0914, PLR0915 - linear on purpose
    cfg: TrainConfig, extra_callbacks: list[Callback] | None = None
) -> TrainResult:
    """Train, evaluate and log one model. See module docstring."""
    L.seed_everything(cfg.seed, workers=True)
    splits = load_splits(cfg.data_prefix)
    pp, windows, test_seg, processed = _windows(cfg, splits)
    train_targets = processed[:, TARGET_IDX]
    loaders = make_loaders(windows, cfg.batch_size)

    n_nodes = windows["train"][0].shape[2]
    layout = target_layout(cfg)
    unsupported = (
        cfg.augment,
        cfg.mixup_alpha,
        cfg.revin,
        cfg.leaky_integrator_idx,
        cfg.per_output_heads,
    )
    if layout.decomposed and any(unsupported):
        raise ValueError(
            "output decomposition supports none of augment/mixup/revin/"
            "leaky integrator/per-output heads"
        )
    if cfg.predict_inflow and (
        not cfg.future_exog or "vazao_entrada" in cfg.future_known
    ):
        raise ValueError(
            "predict_inflow only applies when the future inflow is unknown"
        )
    if layout.known_cols and (
        not cfg.future_exog or cfg.output_window > SEASON
    ):
        raise ValueError("seasonal24 needs future_exog and horizon <= 24")
    model_kwargs = dict(cfg.model_kwargs)
    if cfg.model == "spectral_tf_net" and cfg.future_exog:
        # The model's horizon mask must match the columns actually
        # known: a default of all 4 inputs would read the zeroed
        # `vazao_entrada` of the `pumps` scenario as if it were real.
        known = tuple(known_columns(cfg))
        given = model_kwargs.setdefault("known_future_idx", known)
        if set(given) != set(known):
            raise ValueError(
                f"known_future_idx={tuple(given)} but the known "
                f"columns are {known}"
            )
    # With `future_exog`, every window is `input_window + output_window`
    # long (past + known future inputs, see `data.with_future_exog`).
    seen_window = cfg.input_window + cfg.output_window * cfg.future_exog
    model = build_model(
        cfg.model,
        seen_window,
        cfg.output_window,
        n_nodes,
        target_idx=list(layout.model_cols),
        **model_kwargs,
    )
    if cfg.revin:
        model = RevINWrapper(model, n_nodes, TARGET_IDX, mode=cfg.revin)
    if cfg.leaky_integrator_idx:
        model = LeakyIntegratorWrapper(
            model,
            TARGET_IDX,
            cfg.leaky_integrator_idx,
            init_leak=cfg.leaky_integrator_init,
        )
    if cfg.per_output_heads:
        model = PerOutputHeadWrapper(
            model, len(TARGET_IDX), hidden=cfg.per_output_heads_hidden
        )

    cascade_a, cascade_b = (
        fit_mass_balance(
            train_targets[:, 1],
            processed[:, INPUT_COLS.index("vazao_entrada")],
            train_targets[:, 0],
            exclude=bad_sample_mask(splits["train"])
            if cfg.clean_train
            else None,
        )
        if cfg.cascade_weight > 0
        else (0.0, 0.0)
    )
    # (train_targets/processed are both train-split arrays, same rows)
    loss_kwargs = dict(cfg.loss_kwargs)
    if cfg.loss == "mase":
        loss_kwargs["scale"] = seasonal_naive_scale(train_targets)
    augment = _augmentations(cfg)
    run_name = cfg.run_name or cfg.model

    mlflow.set_experiment(cfg.experiment)
    nested = mlflow.active_run() is not None
    with (
        mlflow.start_run(run_name=run_name, nested=nested) as run,
        tempfile.TemporaryDirectory() as ckpt_dir,
        tempfile.TemporaryDirectory() as r2_ckpt_dir,
    ):
        mlflow.log_params(cfg.flat_params())

        if cfg.pretrain_epochs > 0:
            pretrainer = ContrastivePretrainer(
                model,
                augment or _augmentations(TrainConfig(augment=True)),
                TARGET_IDX,
                sample=next(iter(loaders["train"]))[0],
                temperature=cfg.pretrain_temperature,
                lr=cfg.lr,
            )
            L.Trainer(
                max_epochs=cfg.pretrain_epochs,
                accelerator=cfg.accelerator,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=cfg.progress,
                enable_model_summary=False,
            ).fit(pretrainer, loaders["train"], loaders["val"])
            mlflow.log_metric(
                "pretrain_val_loss",
                float(
                    pretrainer.trainer.callback_metrics["pretrain_val_loss"]
                ),
            )

        module = Forecaster(
            model,
            loss=build_loss(cfg.loss, **loss_kwargs),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer,
            momentum=cfg.momentum,
            scheduler=cfg.scheduler,
            warmup_epochs=cfg.warmup_epochs,
            max_epochs=cfg.max_epochs,
            augment=augment,
            spectral_weight=cfg.spectral_weight,
            spectral_freq_weights=(
                decaying_freq_weights(
                    cfg.output_window,
                    cfg.spectral_cutoff_period,
                    cfg.spectral_decay,
                )
                if cfg.spectral_cutoff_period
                else None
            ),
            message_l1_weight=cfg.message_l1_weight,
            backcast_weight=cfg.backcast_weight,
            teacher_forcing_ratio=cfg.teacher_forcing_ratio,
            uncertainty_weighted=cfg.uncertainty_weighted,
            n_targets=(
                len(TARGET_IDX)
                if cfg.uncertainty_weighted
                or cfg.cascade_weight > 0
                or layout.decomposed
                or cfg.predict_inflow
                else None
            ),
            reconstruction=(
                _reconstruction(pp, layout, cfg) if layout.decomposed else None
            ),
            component_loss_weight=cfg.component_loss_weight,
            n_aux=len(layout.aux_cols),
            output_weights=cfg.output_weights,
            flow_integral_weight=cfg.flow_integral_weight,
            flow_integral_idx=cfg.flow_integral_idx,
            cascade_weight=cfg.cascade_weight,
            cascade_a=cascade_a,
            cascade_b=cascade_b,
            cascade_vazao_idx=0,
            cascade_nivel_idx=1,
            cascade_level_col=TARGET_IDX[1],
            cascade_inflow_col=INPUT_COLS.index("vazao_entrada"),
            cascade_outflow_col=TARGET_IDX[0],
            cascade_anchor_step=cfg.input_window - 1,
            cascade_inflow_known=(
                cfg.future_exog and "vazao_entrada" in cfg.future_known
            ),
        )
        checkpoint = MLflowModelCheckpoint(
            dirpath=ckpt_dir, monitor="val_mse", mode="min", save_top_k=1
        )
        # Tracks the best `val_r2` alongside `val_mse`'s checkpoint
        # (which still drives early stopping/restoration); doesn't
        # save a file, just the running best score for `Forecaster`'s
        # hyperparameter search to optimise directly.
        r2_tracker = ModelCheckpoint(
            dirpath=r2_ckpt_dir, monitor="val_r2", mode="max", save_top_k=1
        )
        callbacks: list[Callback] = [
            EarlyStopping(monitor="val_mse", patience=cfg.patience),
            checkpoint,
            r2_tracker,
            *(extra_callbacks or []),
        ]
        if cfg.ema_decay:
            callbacks.append(EMA(cfg.ema_decay))
        if cfg.swa_lr:
            callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.swa_lr))
        if cfg.freeze_encoder_epochs > 0:
            callbacks.append(EncoderFreeze(cfg.freeze_encoder_epochs))

        trainer = L.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator=cfg.accelerator,
            precision=cfg.precision,
            gradient_clip_val=cfg.gradient_clip_val,
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=cfg.progress,
            enable_model_summary=False,
        )
        trainer.fit(module, loaders["train"], loaders["val"])
        mlflow.log_metric("best_val_mse", float(checkpoint.best_model_score))
        # `save_top_k=1` keeps no file written to disk when fit ends
        # before a single validation epoch (e.g. `max_epochs=1`);
        # fall back to the last logged value in that edge case.
        best_val_r2 = r2_tracker.best_model_score
        if best_val_r2 is None:
            best_val_r2 = trainer.callback_metrics.get("val_r2")
        mlflow.log_metric("best_val_r2", float(best_val_r2))

        # SWA's averaged weights only exist at the end of training.
        ckpt_path = None if cfg.swa_lr else "best"
        test_out = trainer.test(module, loaders["test"], ckpt_path=ckpt_path)
        mlflow.log_metrics({k: float(v) for k, v in test_out[0].items()})

        module.eval()
        y_pred = chained_forecast(
            module.model,
            test_seg,
            pp,
            start=cfg.input_window,
            stop=len(test_seg),
            device=module.device,
            input_window=cfg.input_window,
            output_window=cfg.output_window,
            future_exog=cfg.future_exog,
            future_known_idx=known_columns(cfg),
            layout=layout,
        )
        y_true = test_seg[cfg.input_window : cfg.input_window + len(y_pred)][
            :, TARGET_IDX
        ]
        metrics = metrics_table(y_true, y_pred)
        bands = band_table(y_true, y_pred)
        for out in OUTPUT_COLS:
            for name, value in metrics.loc[out].items():
                key = name.replace("%", "pct")
                mlflow.log_metric(f"chained_{key}_{out}", float(value))
        for (band, out), row in bands.iterrows():
            mlflow.log_metric(
                f"energia_rel_pct_{band}_{out}", float(row["energia_rel_%"])
            )

        if cfg.log_plots:
            idx = np.arange(len(y_pred))
            log_figure(
                plot_series(y_true, y_pred, run_name, idx, metrics),
                "serie",
            )
            log_figure(
                plot_spectrum(y_true, y_pred, run_name, bands), "espectro"
            )

        if cfg.save:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            weights = MODELS_DIR / f"{run_name}.pth"
            pp_path = MODELS_DIR / f"{run_name}_preprocessor.joblib"
            torch.save(module.model.state_dict(), weights)
            joblib.dump(pp, pp_path)
            mlflow.log_artifact(str(weights), "model")
            mlflow.log_artifact(str(pp_path), "model")

    return TrainResult(
        module=module,
        preprocessor=pp,
        metrics=metrics,
        bands=bands,
        y_true=y_true,
        y_pred=y_pred,
        best_val_mse=float(checkpoint.best_model_score),
        best_val_r2=float(best_val_r2),
        run_id=run.info.run_id,
    )
