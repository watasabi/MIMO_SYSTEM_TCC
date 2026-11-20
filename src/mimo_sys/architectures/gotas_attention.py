"""GOTAS-A: graph of attention filters, direct multi-step forecast.

Plan: `docs/18_plano-gotas-w.md` (exploração). Idea from the
encoder-decoder with dot-product attention of
`data/external/seq2seq attention.ipynb`: at every horizon step the
model spreads one weight per past hour, and the attention map tells
which hours it reads. Here the attention lives on each edge of the
graph, and the forecast is **direct** (every step from the observed
past, no recursion) and **anchored on the last observed value**, the two
choices that make the tree ensembles robust in the cross-validation:

    y_i(t+k) = y_i(t-1) + b_ik
               + sum_j g_ijk * sum_tau a_ijk(tau) * [s_j(t-tau) - s_j(t-1)]
               + sum_{j known} sum_{k' <= k} c_ijkk' * [u_j(t+k') - u_j(t-1)]

`a_ijk` is a softmax over the lags `tau = 1..W` (sums to 1), from a
query of target `i` at step `k` against a key of sender `j` at lag
`tau`. With `operating_query=False` the attention is fixed after
training and every edge is the explicit kernel `g_ijk * a_ijk(tau)`
(`kernel`). With `operating_query=True` the query also reads the known
inputs at step `k` (the pump schedule), so the lags each step reads
change with the operating point; `attention` returns the map.

Shape:
    - Input: `(batch, past + horizon, n_nodes)` (known future in input)
    - Output: `(batch, horizon, len(target_idx))`
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["GOTASAttention"]

DEFAULT_KNOWN_FUTURE = (0, 1, 2, 3)


class GOTASAttention(nn.Module):
    """Direct graph forecaster with one attention filter per edge and step.

    Args:
        n_nodes: Number of series.
        input_window: Length of `x`: past window plus the horizon.
        output_window: Forecast horizon `K`.
        target_idx: Output nodes, in return order.
        attention_dim: Size of queries and keys.
        operating_query: Queries also read the known inputs at the step.
        residual_hidden: Width of a nonlinear residual MLP on each
            output's own past (0 disables), zero-initialised.
        known_future_idx: Nodes whose future values are real inputs.
    """

    def __init__(  # noqa: PLR0913 - every knob is a config field
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
        *,
        attention_dim: int = 8,
        operating_query: bool = False,
        residual_hidden: int = 0,
        known_future_idx: tuple[int, ...] = DEFAULT_KNOWN_FUTURE,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.output_window = output_window
        self.past_len = input_window - output_window
        if self.past_len < 2:  # noqa: PLR2004
            raise ValueError("input_window must leave at least 2 past rows")
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        self.register_buffer(
            "known_idx", torch.tensor(known_future_idx, dtype=torch.long)
        )
        n_t, n_k, h = len(target_idx), len(known_future_idx), output_window
        lags = self.past_len - 1  # tau = 1..W-1 (tau = 0 is the anchor)
        self.scale = 1 / math.sqrt(attention_dim)
        self.query = nn.Parameter(0.1 * torch.randn(n_t, h, attention_dim))
        self.key = nn.Parameter(
            0.1 * torch.randn(n_nodes, lags, attention_dim)
        )
        self.operating = (
            nn.Linear(n_k, n_t * attention_dim, bias=False)
            if operating_query
            else None
        )
        self.gain = nn.Parameter(0.01 * torch.randn(n_t, h, n_nodes))
        # Known future inputs, causal over the horizon: (i, k, j, k').
        self.future = nn.Parameter(torch.zeros(n_t, h, n_k, h))
        self.register_buffer("future_causal", torch.tril(torch.ones(h, h)))
        self.bias = nn.Parameter(torch.zeros(n_t, h))
        self.residual = None
        if residual_hidden > 0:
            self.residual = nn.Sequential(
                nn.Linear(self.past_len, residual_hidden),
                nn.Tanh(),
                nn.Linear(residual_hidden, output_window),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        self._last_attention: torch.Tensor | None = None

    def attention(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """`(batch, target, step, sender, lag)` softmax over the lags.

        Without `x` (or without `operating_query`) the batch axis is 1:
        the fixed attention of the trained model.
        """
        query = self.query[None]  # (1, i, k, d)
        if self.operating is not None and x is not None:
            known = x[:, self.past_len :, self.known_idx]  # (b, k, n_k)
            shift = self.operating(known)  # (b, k, i * d)
            shift = shift.view(
                x.shape[0], self.output_window, -1, query.shape[-1]
            )
            # Não `+=`: `query` é uma visão do parâmetro.
            query = query + shift.transpose(1, 2)  # noqa: PLR6104
        scores = torch.einsum("bikd,jld->bikjl", query, self.key) * self.scale
        return torch.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        past = x[:, : self.past_len]  # (b, W, n)
        last = past[:, -1]  # (b, n)
        # Lag tau = 1..W-1 relative to the last row, newest first.
        history = past[:, :-1].flip(1) - last[:, None]  # (b, W-1, n)
        attn = self.attention(x)  # (b|1, i, k, j, l)
        self._last_attention = attn.detach()
        summary = torch.einsum("bikjl,blj->bikj", attn, history)
        message = (summary * self.gain).sum(-1)  # (b, i, k)
        known = (
            x[:, self.past_len :, self.known_idx]
            - last[:, None, self.known_idx]
        )
        future = torch.einsum(
            "ikjm,km,bmj->bik", self.future, self.future_causal, known
        )
        out = message + future + self.bias  # (b, i, k)
        if self.residual is not None:
            target_past = (
                past[..., self.target_idx] - last[:, None, self.target_idx]
            )
            out += self.residual(target_past.transpose(1, 2))
        return (out + last[:, self.target_idx, None]).transpose(1, 2)

    @torch.no_grad()
    def kernel(self) -> torch.Tensor:
        """`(target, step, sender, lag)` explicit filter `g * a` per edge.

        Only an equation when `operating_query=False`; otherwise it is the
        kernel at the mean operating point (zero query shift).
        """
        return self.attention()[0] * self.gain[..., None]

    def group_l1(self, alpha: float = 0.5) -> torch.Tensor:
        """Sparse gains: single (edge, step) terms and whole edges."""
        per_term = self.gain.abs().sum() + self.future.abs().sum()
        per_edge = self.gain.pow(2).sum(dim=1).sqrt().sum()
        return alpha * per_term + (1 - alpha) * per_edge
