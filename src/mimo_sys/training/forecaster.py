"""LightningModule that trains any forecaster with the shared contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import lightning as L
import torch
import torchmetrics
from torch import nn

from mimo_sys.architectures import (
    LinearEdgeNet,
    MPNNForecaster,
    Seq2SeqLatentGNN,
    SpectralTFNet,
    StemGNN,
    SymbolicGraphNetwork,
)
from mimo_sys.training.augmentations import Batch
from mimo_sys.training.integrator import LeakyIntegratorWrapper
from mimo_sys.training.losses import SpectralLoss, UncertaintyWeightedLoss
from mimo_sys.training.mass_balance_cascade import (
    mass_balance_cascade_forecast_torch,
)
from mimo_sys.training.output_heads import PerOutputHeadWrapper
from mimo_sys.training.scalers import RevINWrapper

# Seasonal period (h) of the inflow guess when it is not a known input.
SEASON = 24

_WRAPPERS = (RevINWrapper, LeakyIntegratorWrapper, PerOutputHeadWrapper)


def build_optimizer(
    params: object,
    name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    """`adam`, `adamw`, `rmsprop`, or `sgd_nesterov`."""
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    if name == "sgd_nesterov":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        )
    raise ValueError(f"unknown optimizer: {name!r}")


def _unwrap(model: nn.Module) -> nn.Module:
    """Peel RevIN/leaky-integrator/per-output-head wrappers to the core."""
    while isinstance(model, _WRAPPERS):
        model = model.model
    return model


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
        weight_decay: Optimizer weight decay.
        optimizer: `adam`, `adamw`, `rmsprop`, or `sgd_nesterov`.
        momentum: Momentum for `sgd_nesterov`/`rmsprop`; ignored by
            `adam`/`adamw`.
        scheduler: `plateau` (ReduceLROnPlateau on `val_mse`),
            `cosine_warmup` (linear warmup then cosine), `onecycle`, or
            `none`.
        warmup_epochs: Warmup length for `cosine_warmup`.
        max_epochs: Horizon of `cosine_warmup`/`onecycle`.
        augment: Training-only `(x, y) -> (x, y)` augmentation.
        spectral_weight: Weight of `SpectralLoss` (0 disables).
        spectral_freq_weights: Optional per-frequency-bin weight for
            `SpectralLoss` (see `losses.decaying_freq_weights`), to
            tolerate error in the high frequencies instead of
            penalising the whole spectrum equally.
        message_l1_weight: Weight of the message L1 of MPNN/SGN
            (Cranmer et al. use 1e-2; 0 disables).
        backcast_weight: Weight of the model's `backcast_loss`
            (`StemGNN`'s backcast head gets no gradient without it;
            `SpectralTFNet`'s is an auxiliary auto-consistency term,
            see `SpectralTFNet.backcast_loss`).
        teacher_forcing_ratio: Seq2SeqLatentGNN teacher forcing in
            training.
        uncertainty_weighted: Combine per-output losses with
            `losses.UncertaintyWeightedLoss` instead of averaging them
            — lets a structurally harder output (e.g. `nivel_res`,
            which integrates the inflow/outflow imbalance instead of
            being near-algebraic in the inputs) keep pulling the
            optimiser instead of being drowned out by easier outputs.
            `n_targets` must be passed when this is on.
        n_targets: Number of outputs; required with
            `uncertainty_weighted=True`.
        output_weights: Fixed per-output loss multiplier, `(n_targets,)`
            — a manual alternative to `uncertainty_weighted` (ignored
            when that is on). Unlike the learned weighting, this lets
            you force the optimiser to trade the easy outputs' error
            for the hard one's, instead of letting it find its own
            balance.
        flow_integral_weight: Weight of `MSE(cumsum(pred - target))`
            on output `flow_integral_idx` over the horizon — a cheap
            proxy that penalises a per-window *drift* in that output's
            error, not just its instantaneous size (0 disables).
        flow_integral_idx: Which output channel (index into `pred`'s
            last dim) `flow_integral_weight` applies to.
        cascade_weight: Weight of the physics-cascade loss
            (`mass_balance_cascade.mass_balance_cascade_forecast_torch`)
            — trains the `cascade_vazao_idx` output *through* the mass
            -balance cascade against the true `nivel_res`, instead of
            combining the two post-hoc (see
            `docs/7_tecnica-cascata-fisica.md`). Needs `cascade_a`/
            `cascade_b` (from `mass_balance_cascade.fit_mass_balance`,
            fit on the preprocessor's output space) and the dataset's
            `y` to carry one extra trailing column with the known
            future inflow (0 disables; the extra `y` column is then
            not required).
        cascade_a: Mass-balance gray-box slope, fit on the scaled
            (preprocessor output) space.
        cascade_b: Mass-balance gray-box intercept, same space.
        cascade_vazao_idx: Index into `pred`'s last dim of the output
            fed into the cascade as the predicted outflow.
        cascade_nivel_idx: Index into `pred`'s last dim (and into
            `y`'s target columns) of `nivel_res`, the cascade's target.
        cascade_level_col: Index into `x`'s node dim of `nivel_res`,
            to read the window's level anchor
            (`x[:, cascade_anchor_step, col]`).
        cascade_inflow_col: Index into `x`'s node dim of
            `vazao_entrada`, for the anchor imbalance.
        cascade_outflow_col: Index into `x`'s node dim of
            `vazao_distribuicao`, for the anchor imbalance.
        cascade_anchor_step: Time index into `x` of the last observed
            step (`-1`, or `input_window - 1` when `x` also carries
            the known future inputs, see `data.with_future_exog`).
        cascade_inflow_known: The future `vazao_entrada` is a known
            input. When False the cascade uses the inflow of the same
            hours one day earlier (seasonal naive, past data only)
            instead of reading the extra `y` column.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - every knob is a config field
        self,
        model: nn.Module,
        loss: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        momentum: float = 0.9,
        scheduler: Scheduler = "plateau",
        warmup_epochs: int = 5,
        max_epochs: int = 200,
        augment: Callable[[torch.Tensor, torch.Tensor], Batch] | None = None,
        spectral_weight: float = 0.0,
        spectral_freq_weights: torch.Tensor | None = None,
        message_l1_weight: float = 0.0,
        backcast_weight: float = 1.0,
        teacher_forcing_ratio: float = 0.0,
        uncertainty_weighted: bool = False,
        n_targets: int | None = None,
        output_weights: torch.Tensor | None = None,
        flow_integral_weight: float = 0.0,
        flow_integral_idx: int = 0,
        cascade_weight: float = 0.0,
        cascade_a: float = 0.0,
        cascade_b: float = 0.0,
        cascade_vazao_idx: int = 0,
        cascade_nivel_idx: int = 1,
        cascade_level_col: int = 0,
        cascade_inflow_col: int = 0,
        cascade_outflow_col: int = 0,
        cascade_anchor_step: int = -1,
        cascade_inflow_known: bool = True,
        reconstruction: dict[str, torch.Tensor] | None = None,
        component_loss_weight: float = 1.0,
        n_aux: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "loss",
                "augment",
                "spectral_freq_weights",
                "output_weights",
                "reconstruction",
            ]
        )
        self.decomposed = reconstruction is not None
        for name, value in (reconstruction or {}).items():
            self.register_buffer(f"rec_{name}", value)
        self.model = model
        self.loss_fn = loss if loss is not None else nn.MSELoss()
        self.augment = augment
        self.spectral = (
            SpectralLoss(freq_weights=spectral_freq_weights)
            if spectral_weight > 0
            else None
        )
        self.uncertainty = (
            UncertaintyWeightedLoss(n_targets)
            if uncertainty_weighted
            else None
        )
        if output_weights is not None:
            self.register_buffer(
                "output_weights", torch.as_tensor(output_weights).float()
            )
        else:
            self.output_weights = None
        # One collection per stage: sharing state across stages mixes
        # their epoch aggregates.
        self.train_metrics = _stage_metrics()
        self.val_metrics = _stage_metrics()
        self.test_metrics = _stage_metrics()

    @property
    def core(self) -> nn.Module:
        """The architecture itself, under any RevIN/integrator wrappers."""
        return _unwrap(self.model)

    def encoder_modules(self) -> list[nn.Module]:
        """Encoder modules of the architecture (for `EncoderFreeze`)."""
        return self.core.encoder_modules()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast in preprocessor space."""
        return self.model(x)

    def _aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        core = self.core
        seen = x
        probe = self.model
        while isinstance(probe, _WRAPPERS):
            if isinstance(probe, RevINWrapper):
                seen = probe.last_normalized
                break
            probe = probe.model
        aux = torch.zeros((), device=x.device)
        hp = self.hparams
        if hp.message_l1_weight > 0 and isinstance(
            core, (MPNNForecaster, SymbolicGraphNetwork)
        ):
            aux += hp.message_l1_weight * core.message_l1()
        if hp.message_l1_weight > 0 and isinstance(
            core, (LinearEdgeNet, SpectralTFNet)
        ):
            aux += hp.message_l1_weight * core.group_l1()
        if hp.backcast_weight > 0 and isinstance(
            core, (StemGNN, SpectralTFNet)
        ):
            aux += hp.backcast_weight * core.backcast_loss(seen)
        return aux

    def reconstruct(
        self, comp_pred: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Outputs (scaled like the raw outputs) from their components.

        Each predicted component goes back to raw units through its own
        affine scaler map, the known components (seasonal base) are
        read from the input horizon, everything is summed per output in
        raw units — offsets do not cancel in the scaled space — and the
        sum is scaled with the output's own map.
        """
        raw = (comp_pred - self.rec_pred_b) / self.rec_pred_a
        out = comp_pred.new_zeros(*comp_pred.shape[:2], len(self.rec_out_a))
        out = out.index_add(2, self.rec_pred_output, raw)
        if len(self.rec_known_cols):
            horizon = comp_pred.shape[1]
            known = x[:, -horizon:, self.rec_known_cols]
            known = (known - self.rec_known_b) / self.rec_known_a
            out = out.index_add(2, self.rec_known_output, known)
        return out * self.rec_out_a + self.rec_out_b

    def _step(  # noqa: PLR0912, PLR0914, PLR0915 - linear on purpose
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        x, y = batch
        training = stage == "train"
        # `y` carries one extra trailing column (the known future
        # `vazao_entrada`) when `cascade_weight > 0` — never fed to
        # the model, nor to augmentations (which index it by target
        # position), only used below for the cascade loss. With an
        # output decomposition the predicted components sit between
        # the outputs and that column.
        n_targets = self.hparams.n_targets or y.shape[-1]
        n_comp = len(self.rec_pred_a) if self.decomposed else 0
        n_aux = self.hparams.n_aux
        y, y_comp, y_aux, exog = (
            y[..., :n_targets],
            y[..., n_targets : n_targets + n_comp],
            y[..., n_targets + n_comp : n_targets + n_comp + n_aux],
            y[..., n_targets + n_comp + n_aux :],
        )
        if training and self.augment is not None:
            x, y = self.augment(x, y)

        tf = self.hparams.teacher_forcing_ratio
        comp_pred = aux_pred = None
        if training and tf > 0 and isinstance(self.core, Seq2SeqLatentGNN):
            pred = self.model(x, target=y, teacher_forcing_ratio=tf)
        else:
            pred = self.model(x)
            if n_aux:
                pred, aux_pred = pred[..., :-n_aux], pred[..., -n_aux:]
            if self.decomposed:
                comp_pred = pred
                pred = self.reconstruct(comp_pred, x)

        if self.uncertainty is not None:
            per_output = torch.stack(
                [
                    self.loss_fn(pred[..., i], y[..., i])
                    for i in range(pred.shape[-1])
                ]
            )
            loss = self.uncertainty(per_output)
            self.log_dict(
                {
                    f"{stage}_uncertainty_weight_{i}": w
                    for i, w in enumerate(self.uncertainty.weights())
                },
                on_epoch=True,
            )
        elif self.output_weights is not None:
            per_output = torch.stack(
                [
                    self.loss_fn(pred[..., i], y[..., i])
                    for i in range(pred.shape[-1])
                ]
            )
            loss = (per_output * self.output_weights).sum()
        else:
            loss = self.loss_fn(pred, y)
        if comp_pred is not None:
            # lambda * loss on the rebuilt outputs (what the metric
            # sees) + (1 - lambda) * loss on the scaled components
            # (each band weighted by its own scale, not its variance
            # share of the sum).
            lam = self.hparams.component_loss_weight
            comp_loss = self.loss_fn(comp_pred, y_comp)
            loss = lam * loss + (1 - lam) * comp_loss
            self.log(f"{stage}_component_loss", comp_loss, on_epoch=True)
        if aux_pred is not None:
            aux_loss = self.loss_fn(aux_pred, y_aux)
            loss += aux_loss
            self.log(f"{stage}_aux_loss", aux_loss, on_epoch=True)
        if self.spectral is not None:
            loss += self.hparams.spectral_weight * self.spectral(pred, y)
        if self.hparams.flow_integral_weight > 0:
            idx = self.hparams.flow_integral_idx
            err = pred[..., idx] - y[..., idx]
            loss += (
                self.hparams.flow_integral_weight
                * torch.cumsum(err, dim=1).pow(2).mean()
            )
        if self.hparams.cascade_weight > 0:
            hp = self.hparams
            inflow = exog[..., 0]
            if aux_pred is not None:
                # The graph's own forecast of the unknown inflow.
                inflow = aux_pred[..., 0]
            elif not hp.cascade_inflow_known:
                # Same hours of the previous day (seasonal naive): past
                # data only, ~3x less integrated error than holding the
                # last value. Needs input_window >= 24 >= horizon.
                start = hp.cascade_anchor_step + 1 - SEASON
                horizon = inflow.shape[1]
                inflow = x[:, start : start + horizon, hp.cascade_inflow_col]
            cascade_nivel = mass_balance_cascade_forecast_torch(
                level_anchor=x[
                    :, hp.cascade_anchor_step, hp.cascade_level_col
                ],
                inflow=inflow,
                predicted_outflow=pred[..., hp.cascade_vazao_idx],
                a=hp.cascade_a,
                b=hp.cascade_b,
                last_imbalance=(
                    x[:, hp.cascade_anchor_step, hp.cascade_inflow_col]
                    - x[:, hp.cascade_anchor_step, hp.cascade_outflow_col]
                ),
            )
            cascade_loss = self.loss_fn(
                cascade_nivel, y[..., hp.cascade_nivel_idx]
            )
            loss += hp.cascade_weight * cascade_loss
            self.log(f"{stage}_cascade_loss", cascade_loss, on_epoch=True)
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
        optimizer = build_optimizer(
            self.parameters(),
            name=hp.optimizer,
            lr=hp.lr,
            weight_decay=hp.weight_decay,
            momentum=hp.momentum,
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
