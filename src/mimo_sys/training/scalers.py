"""Per-window normalisation inside the model (RevIN / Nixtla TemporalNorm).

Unlike the preprocessor's scaler (fitted once on the training rows and
inverted by `inverse_transform`), RevIN normalises every input window by
its own statistics and undoes it on the output, inside `forward`. The
model therefore still maps preprocessor space to preprocessor space and
reconstruction is untouched. It helps when the level drifts between
train and test, which a global scaler cannot follow.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

RevINMode = Literal["standard", "robust", "minmax"]


class RevINWrapper(nn.Module):
    """Wrap a forecaster with reversible instance normalisation.

    Args:
        model: Forecaster with the shared contract
            `(batch, input_window, n_nodes) -> (batch, horizon, n_targets)`.
        n_nodes: Number of input series.
        target_idx: Node indices of the outputs, in output order.
        mode: Window statistic: mean/std, median/IQR or min/max.
        affine: Learn a per-node scale and shift after normalising.

    Attributes:
        last_normalized: Normalised input of the last call, the tensor
            the wrapped model actually saw (needed by auxiliary losses
            such as `StemGNN.backcast_loss`).
    """

    def __init__(
        self,
        model: nn.Module,
        n_nodes: int,
        target_idx: list[int],
        mode: RevINMode = "standard",
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.mode = mode
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_nodes))
            self.bias = nn.Parameter(torch.zeros(n_nodes))
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        self.last_normalized: torch.Tensor | None = None

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "standard":
            loc = x.mean(dim=1, keepdim=True)
            scale = x.std(dim=1, keepdim=True, unbiased=False)
        elif self.mode == "robust":
            loc = x.median(dim=1, keepdim=True).values
            q = torch.quantile(
                x, torch.tensor([0.25, 0.75], device=x.device), dim=1
            )
            scale = (q[1] - q[0]).unsqueeze(1)
        else:
            lo = x.min(dim=1, keepdim=True).values
            loc, scale = lo, x.max(dim=1, keepdim=True).values - lo
        return loc.detach(), scale.detach() + 1e-5

    def normalize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalised window plus the `(loc, scale)` used."""
        loc, scale = self._stats(x)
        z = (x - loc) / scale
        if self.affine:
            z = z * self.weight + self.bias
        return z, loc, scale

    def denormalize(
        self, z: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Undo `normalize` on target channels `(batch, h, n_targets)`."""
        idx = self.target_idx
        if self.affine:
            z = (z - self.bias[idx]) / (self.weight[idx] + 1e-8)
        return z * scale[..., idx] + loc[..., idx]

    def _normalize_targets(
        self, y: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        idx = self.target_idx
        z = (y - loc[..., idx]) / scale[..., idx]
        if self.affine:
            z = z * self.weight[idx] + self.bias[idx]
        return z

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Normalise, forecast, denormalise.

        A `target=` keyword (teacher forcing) is normalised with the
        window's statistics before reaching the wrapped model.
        """
        z, loc, scale = self.normalize(x)
        self.last_normalized = z
        target = kwargs.get("target")
        if isinstance(target, torch.Tensor):
            kwargs["target"] = self._normalize_targets(target, loc, scale)
        return self.denormalize(self.model(z, **kwargs), loc, scale)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapped model's `encode` on the normalised window."""
        return self.model.encode(self.normalize(x)[0])

    def encoder_modules(self) -> list[nn.Module]:
        """Wrapped model's encoder modules."""
        return self.model.encoder_modules()
