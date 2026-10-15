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


def decaying_freq_weights(
    n_steps: int, cutoff_period: float = 12.0, decay: float = 2.0
) -> torch.Tensor:
    """Per-frequency-bin weight that fades out above `cutoff_period`.

    `vazao_distribuicao`/`nivel_res`/`pressao` have their energy
    concentrated at the daily cycle and its ~12h harmonic (EDA,
    `docs/5_eda-series-temporais.md` §7); content shorter than
    `cutoff_period` is more often sensor noise than signal, so it is
    down-weighted rather than dropped (weight never reaches 0).

    Args:
        n_steps: Horizon the FFT runs over (`rfft` length is
            `n_steps // 2 + 1`).
        cutoff_period: Period (in steps) below which weight starts
            decaying; bins at or below the Nyquist get weight 1.
        decay: How fast weight falls past the cutoff — `1 /
            (1 + decay * (f / f_cutoff - 1))` for `f > f_cutoff`.

    Returns:
        `(n_steps // 2 + 1,)` weights in `(0, 1]`, DC always 1.
    """
    freqs = torch.fft.rfftfreq(n_steps)
    cutoff_freq = 1.0 / cutoff_period
    ratio = torch.clamp(freqs / cutoff_freq - 1.0, min=0.0)
    return 1.0 / (1.0 + decay * ratio)


class SpectralLoss(nn.Module):
    """L1 between FFT amplitudes of prediction and target along time.

    Point losses reward smooth forecasts; the collection analysis showed
    the networks lose 30–94% of the energy above the daily band. This
    term penalises missing (or injected) energy per frequency directly
    — optionally down-weighting the high-frequency bins that are more
    often noise than signal (`freq_weights`/`decaying_freq_weights`).

    Args:
        log: Compare `log1p` amplitudes, so small high-frequency
            components are not drowned by the low-frequency ones.
        freq_weights: Optional `(horizon // 2 + 1,)` per-bin weight,
            e.g. from `decaying_freq_weights`. Defaults to uniform.
    """

    def __init__(
        self, log: bool = True, freq_weights: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.log = log
        if freq_weights is not None:
            self.register_buffer(
                "freq_weights", torch.as_tensor(freq_weights).float()
            )
        else:
            self.freq_weights = None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Mean absolute amplitude difference over frequencies."""
        amp_pred = torch.fft.rfft(pred, dim=1).abs()
        amp_true = torch.fft.rfft(target, dim=1).abs()
        if self.log:
            amp_pred, amp_true = amp_pred.log1p(), amp_true.log1p()
        diff = (amp_pred - amp_true).abs()
        if self.freq_weights is not None:
            diff *= self.freq_weights.view(-1, 1)
        return diff.mean()


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
