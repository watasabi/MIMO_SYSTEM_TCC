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
from typing import Any

import joblib
import lightning as L
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    StochasticWeightAveraging,
)
from torch import nn

from mimo_sys.architectures import (
    FourierGNN,
    MPNNForecaster,
    Seq2SeqLatentGNN,
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
    OUTPUT_COLS,
    OUTPUT_WINDOW,
    SEED,
    TARGET_IDX,
    load_splits,
    make_loaders,
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
)
from mimo_sys.training.forecaster import Forecaster, Scheduler
from mimo_sys.training.integrator import LeakyIntegratorWrapper
from mimo_sys.training.losses import (
    build_loss,
    decaying_freq_weights,
    seasonal_naive_scale,
)
from mimo_sys.training.output_heads import PerOutputHeadWrapper
from mimo_sys.training.scalers import RevINMode, RevINWrapper

ARCHITECTURES: dict[str, type[nn.Module]] = {
    "mpnn": MPNNForecaster,
    "sgn": SymbolicGraphNetwork,
    "stemgnn": StemGNN,
    "seq2seq_latent_gnn": Seq2SeqLatentGNN,
    "fourier_gnn": FourierGNN,
}

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
    """
    geometry = {
        "n_nodes": n_nodes,
        "input_window": input_window,
        "output_window": output_window,
        "target_idx": TARGET_IDX,
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
    run_id: str


def _augmentations(cfg: TrainConfig) -> Compose | None:
    augs: list = []
    if cfg.augment:
        augs += [
            Jitter(0.02),
            Scaling(TARGET_IDX, 0.1),
            MagnitudeWarp(TARGET_IDX, 0.1),
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
    if not cfg.mass_balance_feature:
        return splits
    full = np.concatenate([splits["train"], splits["val"], splits["test"]])
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
    train_end = len(splits["train"])
    val_end = train_end + len(splits["val"])
    return {
        "train": full[:train_end],
        "val": full[train_end:val_end],
        "test": full[val_end:],
    }


def _windows(
    cfg: TrainConfig, splits: dict[str, npt.NDArray]
) -> tuple[TimeSeriesPreprocessor, dict, npt.NDArray, npt.NDArray]:
    """Fit the preprocessor on train; window val/test with train context.

    Val/test segments start `cfg.input_window` rows early so their
    first target is the first row of the split.
    """
    splits = add_derived_features(cfg, splits)

    pp = TimeSeriesPreprocessor(
        window_size=cfg.input_window,
        horizon=cfg.output_window,
        plot=False,
        **cfg.preprocessor,
    )
    X_tr, y_tr, processed = pp.fit_transform(splits["train"], feats=TARGET_IDX)
    raw = np.concatenate([splits["train"], splits["val"], splits["test"]])
    train_end = len(splits["train"])
    val_end = train_end + len(splits["val"])
    val_seg = raw[train_end - cfg.input_window : val_end]
    test_seg = raw[val_end - cfg.input_window :]
    windows = {"train": (X_tr, y_tr)}
    for name, seg in (("val", val_seg), ("test", test_seg)):
        X, y = pp.transform(seg)
        windows[name] = (X, y[:, :, TARGET_IDX])
    train_targets = processed[:, TARGET_IDX]
    return pp, windows, test_seg, train_targets


def train(  # noqa: PLR0912, PLR0914, PLR0915 - linear on purpose
    cfg: TrainConfig, extra_callbacks: list[Callback] | None = None
) -> TrainResult:
    """Train, evaluate and log one model. See module docstring."""
    L.seed_everything(cfg.seed, workers=True)
    splits = load_splits(cfg.data_prefix)
    pp, windows, test_seg, train_targets = _windows(cfg, splits)
    loaders = make_loaders(windows, cfg.batch_size)

    n_nodes = N_NODES
    if cfg.mass_balance_feature:
        n_nodes += 1 + bool(cfg.mass_balance_ewma_span)
    model = build_model(
        cfg.model,
        cfg.input_window,
        cfg.output_window,
        n_nodes,
        **cfg.model_kwargs,
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
            n_targets=len(TARGET_IDX) if cfg.uncertainty_weighted else None,
            output_weights=cfg.output_weights,
        )
        checkpoint = MLflowModelCheckpoint(
            dirpath=ckpt_dir, monitor="val_mse", mode="min", save_top_k=1
        )
        callbacks: list[Callback] = [
            EarlyStopping(monitor="val_mse", patience=cfg.patience),
            checkpoint,
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
            gradient_clip_val=cfg.gradient_clip_val,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=cfg.progress,
            enable_model_summary=False,
        )
        trainer.fit(module, loaders["train"], loaders["val"])
        mlflow.log_metric("best_val_mse", float(checkpoint.best_model_score))

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
        run_id=run.info.run_id,
    )
