"""Learned MIMO transfer-function matrix: a linear ARX, fit by gradient.

Motivated by `docs/changes/2026-10-16-rota-a-b-extracao.md`: `MPNNForecaster`'s
shared pairwise `message_fn` cannot express a difference of two senders
(`u4 - y1`) — the sum aggregation only ever adds contributions, so the
mass balance comes out rotated (`u4 + y1 - 2*y2`) instead of literal.

Here, edge `i <- j` is a **linear FIR filter**
(`nn.Linear(input_window, output_window, bias=False)`), and aggregation
is still a sum — but because a linear filter's weights can be negative,
the subtraction the mass balance needs falls out on its own; there is
no shared nonlinearity to rotate it.

This is exactly a MIMO ARX model (`docs/0_referencias-tcc.md`'s
reference class), estimated by gradient descent instead of
least-squares/a metaheuristic. Two routes come from the same object,
with no distillation step:

- **Route A (equation)**: the filter coefficients *are* the equation —
  `y_i[t:t+H] = sum_j G_ij @ x_j[t-W:t] + b_i`. Read `edge_filter(i,
  j)` directly.
- **Route B (topology)**: `edge_strength()` (`||G_ij||_F` per edge) is
  the coupling matrix; `group_l1()` penalises a whole edge's filter at
  once (not coefficient-by-coefficient), so edges die as a unit during
  training — "edge X->Y is zero at lambda=L" instead of a heatmap.

A small, capacity-limited per-node MLP residual is added on top
(`residual_hidden`), so a system that is not quite linear is not
forced through a linear-only fit; `nonlinearity_share()` gives the
fraction of output energy coming from that residual, a single number
for "how non-linear is this system", instead of an argument.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["LinearEdgeNet"]


class LinearEdgeNet(nn.Module):
    """MIMO ARX with one learned linear filter per edge.

    Args:
        n_nodes: Number of series (nodes), self-edges included — a
            node's own lag window is itself an ARX regressor, not just
            its neighbours'.
        input_window: Lag window length (filter order).
        output_window: Forecast horizon.
        target_idx: Indices of the output nodes, in return order.
        residual_hidden: Width of the small per-node nonlinear
            residual MLP (own window -> own horizon). `0` disables it
            (pure linear ARX).

    Shape:
        - Input: `(batch, input_window, n_nodes)`
        - Output: `(batch, output_window, len(target_idx))`
    """

    def __init__(
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
        *,
        residual_hidden: int = 0,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.input_window = input_window
        self.output_window = output_window
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        n_targets = len(target_idx)

        # edges[i][j]: sender j -> receiver target i (dense, self-edges
        # included). One Linear per edge keeps each filter's weight
        # matrix separately addressable for `edge_filter`/`group_l1`.
        self.edges = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(input_window, output_window, bias=False)
                        for _ in range(n_nodes)
                    ]
                )
                for _ in range(n_targets)
            ]
        )
        self.bias = nn.Parameter(torch.zeros(n_targets, output_window))

        self.residual = None
        if residual_hidden > 0:
            self.residual = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(input_window, residual_hidden),
                        nn.Tanh(),
                        nn.Linear(residual_hidden, output_window),
                    )
                    for _ in range(n_targets)
                ]
            )

        # Filled by the last forward pass, for `edge_strength`/
        # `nonlinearity_share`: (batch, n_targets, output_window).
        self._last_linear: torch.Tensor | None = None
        self._last_residual: torch.Tensor | None = None

    def _linear_part(self, x: torch.Tensor) -> torch.Tensor:
        """`(batch, output_window, n_targets)`, the ARX sum alone."""
        out = []
        for i, row in enumerate(self.edges):
            contrib = sum(row[j](x[:, :, j]) for j in range(self.n_nodes))
            out.append(contrib + self.bias[i])
        return torch.stack(out, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ARX sum plus the optional small nonlinear residual."""
        linear = self._linear_part(x)
        self._last_linear = linear.transpose(1, 2)
        if self.residual is None:
            self._last_residual = None
            return linear
        own = x[:, :, self.target_idx]
        residual = torch.stack(
            [
                self.residual[i](own[:, :, i])
                for i in range(len(self.residual))
            ],
            dim=-1,
        )
        self._last_residual = residual.transpose(1, 2)
        return linear + residual

    def edge_filter(self, i: int, j: int) -> torch.Tensor:
        """Learned FIR filter of edge `j -> target_idx[i]`.

        Returns:
            `(output_window, input_window)` weight matrix — row `h`,
            column `tau` is the coefficient of `x_j[t - tau]` in
            `y_{target_idx[i]}[t + h]`, the literal ARX equation.
        """
        return self.edges[i][j].weight.detach()

    @torch.no_grad()
    def edge_strength(self) -> torch.Tensor:
        """`(n_targets, n_nodes)` coupling matrix, `||G_ij||_F` per edge."""
        return torch.stack(
            [
                torch.stack(
                    [row[j].weight.norm() for j in range(self.n_nodes)]
                )
                for row in self.edges
            ]
        )

    def group_l1(self) -> torch.Tensor:
        """Sum of `||G_ij||_F` over all edges.

        Add to the training loss: penalising a whole filter's norm
        (not its individual taps) drives an edge's *entire* filter
        toward zero as one group, the sparsity path route B needs
        (Yuan & Lin, 2006 group lasso) — unlike plain L1 on the raw
        weights, which can zero out individual taps of a filter that
        is still, on net, carrying signal.
        """
        return sum(
            row[j].weight.norm()
            for row in self.edges
            for j in range(self.n_nodes)
        )

    def nonlinearity_share(self) -> float:
        """Fraction of the last forward pass's output energy from the
        residual MLP, `||residual||^2 / (||linear||^2 + ||residual||^2)`.

        0 with `residual_hidden=0` (no residual to measure). Close to 0
        on a trained model says the system is well explained by the
        linear ARX alone; close to 1 says the nonlinear term is doing
        the real work and the linear part should not be trusted as
        "the equation".
        """
        if self._last_residual is None:
            return 0.0
        lin_energy = self._last_linear.pow(2).sum()
        res_energy = self._last_residual.pow(2).sum()
        return (res_energy / (lin_energy + res_energy + 1e-12)).item()
