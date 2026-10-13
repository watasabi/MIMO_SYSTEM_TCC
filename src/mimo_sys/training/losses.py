"""Forecast losses: point losses, MASE and a spectral amplitude loss.

All take `(batch, horizon, n_targets)` predictions and targets.
"""

from __future__ import annotations

import torch
from torch import nn


class SMAPELoss(nn.Module):
    """Symmetric MAPE (fraction, not %)."""

    def forward(  # noqa: PLR6301 - nn.Module API
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Mean of `2|e| / (|y| + |ŷ|)`."""
        denom = pred.abs() + target.abs() + 1e-8
        return (2 * (pred - target).abs() / denom).mean()


class MASELoss(nn.Module):
    """MAE scaled by the in-sample seasonal-naive MAE (Nixtla `MASE`).

    Args:
        scale: `(n_targets,)` seasonal-naive MAE per output, computed
            once on the training series with `seasonal_naive_scale`.
    """

    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale).float())

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-output MAE divided by `scale`, averaged."""
        mae = (pred - target).abs().mean(dim=(0, 1))
        return (mae / (self.scale + 1e-8)).mean()


def seasonal_naive_scale(
    series: torch.Tensor, season: int = 24
) -> torch.Tensor:
    """MAE of `y_t ≈ y_{t-season}` per column, `series` `(n, cols)`."""
    series = torch.as_tensor(series, dtype=torch.float32)
    return (series[season:] - series[:-season]).abs().mean(dim=0)


class SpectralLoss(nn.Module):
    """L1 between FFT amplitudes of prediction and target along time.

    Point losses reward smooth forecasts; the collection analysis showed
    the networks lose 30–94% of the energy above the daily band. This
    term penalises missing (or injected) energy per frequency directly.

    Args:
        log: Compare `log1p` amplitudes, so small high-frequency
            components are not drowned by the low-frequency ones.
    """

    def __init__(self, log: bool = True) -> None:
        super().__init__()
        self.log = log

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Mean absolute amplitude difference over frequencies."""
        amp_pred = torch.fft.rfft(pred, dim=1).abs()
        amp_true = torch.fft.rfft(target, dim=1).abs()
        if self.log:
            amp_pred, amp_true = amp_pred.log1p(), amp_true.log1p()
        return (amp_pred - amp_true).abs().mean()


def build_loss(name: str, **kwargs: object) -> nn.Module:
    """Point loss by name: `mse`, `mae`, `huber`, `smape` or `mase`.

    `mase` needs `scale=...`; `huber` accepts `delta=...`.
    """
    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": nn.HuberLoss,
        "smape": SMAPELoss,
        "mase": MASELoss,
    }
    if name not in losses:
        raise ValueError(f"unknown loss {name!r}; options: {list(losses)}")
    return losses[name](**kwargs)
