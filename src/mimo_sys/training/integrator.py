"""Leaky-integrator head for state-like outputs (e.g. reservoir level).

`docs/1_analise-dados-reservatorio.md` §4-5 identifies `nivel_res` as
the system's one genuine state: it integrates the inflow/outflow
imbalance over time (`A * dy/dt = u4 - y1`), while the other two
outputs are near-algebraic in the state plus the current inputs. A
pure integrator diverges in free-run (R² -19 to -35 in that doc's
gray-box experiment); adding a learned "leak" term (`-lambda * y`)
stabilises it (R² +0.36) at the cost of explaining less variance — the
same bias/stability trade-off this wrapper lets a network learn per
output, instead of hard-coding either choice.

`vazao_distribuicao`/`pressao` stay direct (point) forecasts; only the
state-like output(s) get wrapped, selected by `state_idx`.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["LeakyIntegratorWrapper"]


class LeakyIntegratorWrapper(nn.Module):
    """Wrap a forecaster so selected outputs are deltas, not levels.

    The wrapped model's raw output is reinterpreted, on `state_idx`
    channels only, as a per-step increment `delta_h`. The forecast is
    then the leaky-integrated sum

        y_h = leak^h * y_last + sum_{k=1}^{h} leak^{h-k} * delta_k

    with `leak = sigmoid(raw_leak)` a learned, per-state-output scalar
    in `(0, 1)` (1 = pure integrator, pushed towards instability; less
    than 1 = integrator with leak, trading bias for stability — see
    module docstring). `y_last` is the last observed value of that
    output in the input window, i.e. `x[:, -1, node]`.

    Args:
        model: Forecaster with the shared contract
            `(batch, input_window, n_nodes) -> (batch, horizon,
            n_targets)`.
        target_idx: Node indices of every output, in output order
            (same as the wrapped model's `target_idx`).
        state_idx: Node indices (a subset of `target_idx`) whose
            output channel is integrated. Indices are node indices,
            not positions into `target_idx` — resolved internally.
        init_leak: Initial `leak` value (close to 1: mild leak, stable
            but close to a pure integrator at the start of training).
    """

    def __init__(
        self,
        model: nn.Module,
        target_idx: list[int],
        state_idx: list[int],
        init_leak: float = 0.98,
    ) -> None:
        super().__init__()
        self.model = model
        state_positions = [target_idx.index(i) for i in state_idx]
        self.register_buffer(
            "state_positions", torch.tensor(state_positions, dtype=torch.long)
        )
        self.register_buffer(
            "state_node_idx", torch.tensor(state_idx, dtype=torch.long)
        )
        init_logit = torch.logit(torch.tensor(init_leak))
        self.raw_leak = nn.Parameter(init_logit * torch.ones(len(state_idx)))

    def _integrate(
        self, delta: torch.Tensor, y_last: torch.Tensor
    ) -> torch.Tensor:
        """`(batch, horizon, n_state)` deltas -> leaky-integrated levels."""
        horizon = delta.shape[1]
        leak = torch.sigmoid(self.raw_leak)  # (n_state,)
        powers = leak.pow(
            torch.arange(1, horizon + 1, device=delta.device)[:, None]
        )  # (horizon, n_state)
        # Discount each step's delta by how many leaky steps remain
        # until the horizon it lands on, then cumulative-sum: a causal
        # convolution with a geometric kernel, done via cumsum of
        # rescaled terms (no explicit O(horizon^2) loop).
        discounted = delta / (powers + 1e-12)
        accumulated = torch.cumsum(discounted, dim=1) * powers
        return powers * y_last.unsqueeze(1) + accumulated

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Forecast, then leaky-integrate the state-output channels."""
        out = self.model(x, **kwargs)
        y_last = x[:, -1, self.state_node_idx]
        delta = out[..., self.state_positions]
        levels = self._integrate(delta, y_last)
        out = out.clone()
        out[..., self.state_positions] = levels
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapped model's `encode`, unchanged by the integrator."""
        return self.model.encode(x)

    def encoder_modules(self) -> list[nn.Module]:
        """Wrapped model's encoder modules."""
        return self.model.encoder_modules()
