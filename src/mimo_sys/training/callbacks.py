"""Lightning callbacks: EMA of weights, encoder freeze, MLflow checkpoint.

SWA needs no custom code: use `lightning.pytorch.callbacks.
StochasticWeightAveraging` (wired by `train.TrainConfig.swa_lr`).
"""

from __future__ import annotations

import copy
from typing import Any

import lightning as L
import mlflow
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class EMA(Callback):
    """Exponential moving average of `pl_module.model` weights.

    Validation runs on the EMA weights, checkpoints store them, and at
    the end of training they replace the raw weights. Testing a
    checkpoint therefore evaluates the EMA model with no extra swap.

    Args:
        decay: Weight of the running average per optimiser step.
    """

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema: AveragedModel | None = None
        self._raw_state: dict[str, Any] | None = None

    def on_fit_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.ema = AveragedModel(
            pl_module.model,
            multi_avg_fn=get_ema_multi_avg_fn(self.decay),
            use_buffers=True,
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *args: Any,
    ) -> None:
        self.ema.update_parameters(pl_module.model)

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self.ema is None or trainer.sanity_checking:
            return
        self._raw_state = copy.deepcopy(pl_module.model.state_dict())
        pl_module.model.load_state_dict(self.ema.module.state_dict())

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._raw_state is not None:
            pl_module.model.load_state_dict(self._raw_state)
            self._raw_state = None

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        if self.ema is None:
            return
        for key, value in self.ema.module.state_dict().items():
            checkpoint["state_dict"][f"model.{key}"] = value.detach().clone()

    def on_train_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        pl_module.model.load_state_dict(self.ema.module.state_dict())


class EncoderFreeze(Callback):
    """Linear-probe then fine-tune: freeze the encoder for some epochs.

    Uses `pl_module.encoder_modules()` (see `Forecaster`). Typical use
    is right after contrastive pretraining, so the randomly initialised
    decoder adapts before the pretrained encoder starts moving.

    Args:
        unfreeze_at_epoch: First epoch with a trainable encoder.
    """

    def __init__(self, unfreeze_at_epoch: int = 10) -> None:
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        trainable = trainer.current_epoch >= self.unfreeze_at_epoch
        for module in pl_module.encoder_modules():
            module.requires_grad_(trainable)


class MLflowModelCheckpoint(ModelCheckpoint):
    """`ModelCheckpoint` that logs the best file to the active MLflow run."""

    def __init__(self, artifact_path: str = "checkpoints", **kwargs: Any):
        super().__init__(**kwargs)
        self.artifact_path = artifact_path

    def on_train_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        super().on_train_end(trainer, pl_module)
        if mlflow.active_run() and self.best_model_path:
            mlflow.log_artifact(self.best_model_path, self.artifact_path)
