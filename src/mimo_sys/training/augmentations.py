"""Batch augmentations for forecasting windows, applied only in training.

Every augmentation is a callable `(x, y) -> (x, y)` on
`x: (batch, input_window, n_nodes)` and
`y: (batch, horizon, n_targets)`, in preprocessor space.

Two families:

- **joint** (`Scaling`, `MagnitudeWarp`, `Mixup`): transform input and
  target consistently, so the input→output relation stays physically
  plausible;
- **input-only** (`Jitter`, `TimeWarp`, `SpanMask`, `ChannelMask`,
  `FreqMask`): perturb what the model sees, the target is untouched —
  robustness to noise, gaps and missing sensors.

They never touch reconstruction: evaluation runs without them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

Batch = tuple[torch.Tensor, torch.Tensor]


class Jitter:
    """Additive Gaussian noise on the input."""

    def __init__(self, sigma: float = 0.03) -> None:
        self.sigma = sigma

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        return x + torch.randn_like(x) * self.sigma, y


class Scaling:
    """Scale each node's deviation from its window mean, input and target.

    Args:
        sigma: Std of the log-normal-ish factor around 1.
        target_idx: Node index of each target column.
    """

    def __init__(self, target_idx: Sequence[int], sigma: float = 0.1) -> None:
        self.sigma = sigma
        self.target_idx = list(target_idx)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        factor = 1 + torch.randn(x.shape[0], 1, x.shape[2]) * self.sigma
        factor = factor.to(x)
        mean = x.mean(dim=1, keepdim=True)
        x = mean + (x - mean) * factor
        t = self.target_idx
        y = mean[..., t] + (y - mean[..., t]) * factor[..., t]
        return x, y


class MagnitudeWarp:
    """Smooth multiplicative curve over input+horizon, per node.

    The curve is a cubic interpolation of `knots` random points around
    1, sampled over the joint `input_window + horizon` axis, so input
    and target are warped by the same continuous curve.
    """

    def __init__(
        self, target_idx: Sequence[int], sigma: float = 0.1, knots: int = 4
    ) -> None:
        self.target_idx = list(target_idx)
        self.sigma = sigma
        self.knots = knots

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        b, length, n = x.shape
        total = length + y.shape[1]
        points = 1 + torch.randn(b, n, self.knots + 2) * self.sigma
        curve = F.interpolate(
            points, size=total, mode="linear", align_corners=True
        )
        curve = curve.transpose(1, 2).to(x)  # [b, total, n]
        mean = x.mean(dim=1, keepdim=True)
        t = self.target_idx
        x = mean + (x - mean) * curve[:, :length]
        y = mean[..., t] + (y - mean[..., t]) * curve[:, length:, t]
        return x, y


class Mixup:
    """Mixup for regression: convex combination of pairs, x and y.

    Args:
        alpha: Beta(alpha, alpha) concentration; small values keep
            mixes close to one of the two samples.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.beta = torch.distributions.Beta(alpha, alpha)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        lam = self.beta.sample((x.shape[0], 1, 1)).to(x)
        perm = torch.randperm(x.shape[0], device=x.device)
        return (
            lam * x + (1 - lam) * x[perm],
            lam * y + (1 - lam) * y[perm],
        )


class TimeWarp:
    """Smoothly resample the input time axis, endpoints fixed.

    The last input step is kept in place so the window still connects
    to the (unchanged) target.
    """

    def __init__(self, sigma: float = 0.2, knots: int = 4) -> None:
        self.sigma = sigma
        self.knots = knots

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        b, length, _ = x.shape
        speed = (1 + torch.randn(b, 1, self.knots + 2) * self.sigma).clamp(
            min=0.1
        )
        speed = F.interpolate(
            speed, size=length, mode="linear", align_corners=True
        )[:, 0]
        warp = torch.cumsum(speed, dim=1)
        warp = (warp - warp[:, :1]) / (warp[:, -1:] - warp[:, :1])
        grid_x = (warp * 2 - 1).to(x)  # [-1, 1] over the window
        grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1)
        image = x.transpose(1, 2).unsqueeze(2)  # [b, n, 1, length]
        warped = F.grid_sample(
            image, grid.unsqueeze(1), align_corners=True
        )  # [b, n, 1, length]
        return warped[:, :, 0].transpose(1, 2), y


class SpanMask:
    """Replace a random contiguous span of the input by its window mean."""

    def __init__(self, max_frac: float = 0.25) -> None:
        self.max_frac = max_frac

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        b, length, _ = x.shape
        span = max(1, int(length * self.max_frac))
        widths = torch.randint(1, span + 1, (b,))
        # Never mask the last step: it anchors the forecast.
        starts = (torch.rand(b) * (length - widths)).long()
        steps = torch.arange(length)
        mask = (steps >= starts[:, None]) & (
            steps < (starts + widths)[:, None]
        )
        mask = mask.to(x.device)[..., None]
        return torch.where(mask, x.mean(dim=1, keepdim=True), x), y


class ChannelMask:
    """Blank whole input channels (sensor dropout), never the targets.

    Args:
        channels: Node indices that may be dropped (e.g. `INPUT_IDX`).
        p: Drop probability per channel and sample.
    """

    def __init__(self, channels: Sequence[int], p: float = 0.15) -> None:
        self.channels = list(channels)
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        drop = torch.zeros(x.shape[0], 1, x.shape[2], dtype=torch.bool)
        drop[..., self.channels] = (
            torch.rand(x.shape[0], 1, len(self.channels)) < self.p
        )
        drop = drop.to(x.device)
        return torch.where(drop, x.mean(dim=1, keepdim=True), x), y


class FreqMask:
    """Zero a random contiguous band of the input spectrum (not DC)."""

    def __init__(self, max_frac: float = 0.2) -> None:
        self.max_frac = max_frac

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        length = x.shape[1]
        spec = torch.fft.rfft(x, dim=1)
        n_freq = spec.shape[1]
        width = max(1, int((n_freq - 1) * self.max_frac))
        start = int(torch.randint(1, max(2, n_freq - width + 1), (1,)))
        spec[:, start : start + width] = 0
        return torch.fft.irfft(spec, n=length, dim=1), y


class Compose:
    """Apply each augmentation independently with probability `p`."""

    def __init__(
        self,
        augmentations: Sequence[Callable[[torch.Tensor, torch.Tensor], Batch]],
        p: float = 0.5,
    ) -> None:
        self.augmentations = list(augmentations)
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        for augment in self.augmentations:
            if torch.rand(1).item() < self.p:
                x, y = augment(x, y)
        return x, y
