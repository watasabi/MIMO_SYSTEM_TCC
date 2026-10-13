"""Contrastive (SimCLR-style) pretraining of a forecaster's encoder.

Two augmented views of each training window go through
`model.encode`, are mean-pooled, projected, and pulled together with
NT-Xent against the other windows of the batch. Only the encoder is
trained (in place, on the same model object); the projection head is
discarded afterwards. Supported: `Seq2SeqLatentGNN`, and
`MPNNForecaster`/`SymbolicGraphNetwork` with `n_layers >= 2`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from mimo_sys.training.augmentations import Batch


def nt_xent(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
    """Normalised-temperature cross-entropy over a batch of pairs.

    Row `i` of `z1` and row `i` of `z2` are positives; every other row
    of both is a negative.
    """
    batch = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    labels = torch.arange(2 * batch, device=z.device)
    labels = (labels + batch) % (2 * batch)
    return F.cross_entropy(sim, labels)


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head."""

    def __init__(self, in_dim: int, hidden: int = 64, out: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project pooled embeddings."""
        return self.net(h)


class ContrastivePretrainer(L.LightningModule):
    """Pretrain `model.encode` with NT-Xent on augmented view pairs.

    Args:
        model: Forecaster exposing `encode` and `encoder_modules`.
        augment: `(x, y) -> (x, y)` augmentation; called twice per batch
            for the two views.
        target_idx: Node indices of the targets (the joint
            augmentations need a matching `y`; the last input step is
            used as a stand-in).
        sample: One input batch, used to infer the encoder width.
        temperature: NT-Xent temperature.
        lr: AdamW learning rate.
        proj_dim: Projection output width.
    """

    def __init__(
        self,
        model: nn.Module,
        augment: Callable[[torch.Tensor, torch.Tensor], Batch],
        target_idx: Sequence[int],
        sample: torch.Tensor,
        temperature: float = 0.2,
        lr: float = 1e-3,
        proj_dim: int = 32,
    ) -> None:
        super().__init__()
        self.model = model
        self.augment = augment
        self.target_idx = list(target_idx)
        self.temperature = temperature
        self.lr = lr
        with torch.no_grad():
            width = model.encode(sample[:2]).shape[-1]
        self.head = ProjectionHead(width, out=proj_dim)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.model.encode(x).mean(dim=1))

    def _loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, _ = batch
        stand_in = x[:, -1:, self.target_idx]
        view1, _ = self.augment(x, stand_in)
        view2, _ = self.augment(x, stand_in)
        return nt_xent(
            self._project(view1), self._project(view2), self.temperature
        )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("pretrain_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self.log("pretrain_val_loss", self._loss(batch), prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [
            p for m in self.model.encoder_modules() for p in m.parameters()
        ]
        params += list(self.head.parameters())
        return torch.optim.AdamW(params, lr=self.lr)
