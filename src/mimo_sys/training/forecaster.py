"""LightningModule that trains any forecaster with the shared contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import lightning as L
import torch
import torchmetrics
from torch import nn

from mimo_sys.architectures import (
    MPNNForecaster,
    Seq2SeqLatentGNN,
    StemGNN,
    SymbolicGraphNetwork,
)
from mimo_sys.training.augmentations import Batch
from mimo_sys.training.losses import SpectralLoss
from mimo_sys.training.scalers import RevINWrapper

Scheduler = Literal["plateau", "cosine_warmup", "onecycle", "none"]


def _stage_metrics() -> torchmetrics.MetricCollection:
    return torchmetrics.MetricCollection(
        {
            "mse": torchmetrics.MeanSquaredError(),
            "mae": torchmetrics.MeanAbsoluteError(),
            "r2": torchmetrics.R2Score(),
        }
    )


class Forecaster(L.LightningModule):
    """Train/val/test loop with optional regularisers and augmentations.

    Args:
        model: Forecaster `(B, L, n_nodes) -> (B, H, n_targets)`,
            optionally wrapped in `RevINWrapper`.
        loss: Point loss on the forecast (see `losses.build_loss`).
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        scheduler: `plateau` (ReduceLROnPlateau on `val_mse`),
            `cosine_warmup` (linear warmup then cosine), `onecycle`, or
            `none`.
        warmup_epochs: Warmup length for `cosine_warmup`.
        max_epochs: Horizon of `cosine_warmup`/`onecycle`.
        augment: Training-only `(x, y) -> (x, y)` augmentation.
        spectral_weight: Weight of `SpectralLoss` (0 disables).
        message_l1_weight: Weight of the message L1 of MPNN/SGN
            (Cranmer et al. use 1e-2; 0 disables).
        backcast_weight: Weight of `StemGNN.backcast_loss`; StemGNN's
            backcast head gets no gradient without it.
        teacher_forcing_ratio: Seq2SeqLatentGNN teacher forcing in
            training.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: Scheduler = "plateau",
        warmup_epochs: int = 5,
        max_epochs: int = 200,
        augment: Callable[[torch.Tensor, torch.Tensor], Batch] | None = None,
        spectral_weight: float = 0.0,
        message_l1_weight: float = 0.0,
        backcast_weight: float = 1.0,
        teacher_forcing_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss", "augment"])
        self.model = model
        self.loss_fn = loss if loss is not None else nn.MSELoss()
        self.augment = augment
        self.spectral = SpectralLoss() if spectral_weight > 0 else None
        # One collection per stage: sharing state across stages mixes
        # their epoch aggregates.
        self.train_metrics = _stage_metrics()
        self.val_metrics = _stage_metrics()
        self.test_metrics = _stage_metrics()

    @property
    def core(self) -> nn.Module:
        """The architecture itself, without the RevIN wrapper."""
        if isinstance(self.model, RevINWrapper):
            return self.model.model
        return self.model

    def encoder_modules(self) -> list[nn.Module]:
        """Encoder modules of the architecture (for `EncoderFreeze`)."""
        return self.core.encoder_modules()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast in preprocessor space."""
        return self.model(x)

    def _aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        core = self.core
        seen = (
            self.model.last_normalized
            if isinstance(self.model, RevINWrapper)
            else x
        )
        aux = torch.zeros((), device=x.device)
        hp = self.hparams
        if hp.message_l1_weight > 0 and isinstance(
            core, (MPNNForecaster, SymbolicGraphNetwork)
        ):
            aux += hp.message_l1_weight * core.message_l1()
        if hp.backcast_weight > 0 and isinstance(core, StemGNN):
            aux += hp.backcast_weight * core.backcast_loss(seen)
        return aux

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        x, y = batch
        training = stage == "train"
        if training and self.augment is not None:
            x, y = self.augment(x, y)

        tf = self.hparams.teacher_forcing_ratio
        if training and tf > 0 and isinstance(self.core, Seq2SeqLatentGNN):
            pred = self.model(x, target=y, teacher_forcing_ratio=tf)
        else:
            pred = self.model(x)

        loss = self.loss_fn(pred, y)
        if self.spectral is not None:
            loss += self.hparams.spectral_weight * self.spectral(pred, y)
        if training:
            loss += self._aux_loss(x)

        metrics = getattr(self, f"{stage}_metrics")
        metrics.update(pred.reshape(-1), y.reshape(-1))
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log_dict(
            {f"{stage}_{k}": v for k, v in metrics.items()}, on_epoch=True
        )
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._step(batch, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> dict:
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
        if hp.scheduler == "none":
            return {"optimizer": optimizer}
        if hp.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mse",
                },
            }
        if hp.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=hp.lr,
                total_steps=int(self.trainer.estimated_stepping_batches),
            )
            interval = "step"
        else:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=hp.warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, hp.max_epochs - hp.warmup_epochs)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, cosine], milestones=[hp.warmup_epochs]
            )
            interval = "epoch"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": interval},
        }
