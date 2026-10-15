"""FourierGNN (Yi et al., NeurIPS 2023): pure-graph forecasting via a
Fourier Graph Operator over a hypervariate graph.

Reference implementation of `docs/pappers/FourierGNN_2023.pdf`. The
triage in `docs/3_investigacao-arquiteturas.md` picked this to replace
`StemGNN` as the accuracy/topology baseline ("Modelo A"): both reduce
to a fully-connected graph over the series, but FourierGNN's graph
Fourier transform is the **fixed** DFT, not an eigenbasis learned from
the data (what makes `StemGNN`'s node attention collapse and its edge
masking ill-defined at `n_nodes=7` — see `stemgnn.py`). That fixed
basis is also why this *does* serve the subgraph/attribution route
(B1, `docs/4_rotas-de-extracao.md`) even though it does not serve
symbolic distillation (route A): the Fourier Graph Operator (FGO) is a
single n-invariant complex matrix shared by every node, not a per-edge
MLP, so there is nothing with named physical inputs to distill.

Node = `(variable, lag)` pair ("hypervariate graph", paper §4.1): a
window `(input_window, n_nodes)` is flattened to `input_window *
n_nodes` nodes, embedded to `d` dimensions, and the whole forecast
runs as `K` stacked FGOs applied in Fourier space over that flat node
axis, then an inverse DFT and a 3-layer FFN head back to `(output_window,
n_targets)`.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FourierGNN"]


class _FourierGraphOperator(nn.Module):
    """One FGO: a learned n-invariant complex matrix `S in C^{d x d}`.

    Paper §4.2/Appendix C.2: parameterising the *full* `F(kappa) in
    C^{n x d x d}` kernel is more expressive but scales with the
    number of nodes `n = input_window * n_nodes` and overfits on small
    data; the n-invariant special case (one shared `(d, d)` matrix,
    applied identically to every node's spectrum) is what the paper
    recommends and what its ablation (Table 6) shows generalises
    better. Complex weights are stored as separate real/imaginary
    `nn.Linear`s, since PyTorch's autograd on complex parameters is
    still second-class.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.real = nn.Linear(dim, dim, bias=False)
        self.imag = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Complex `x @ S + b`, `x` real/imag stacked on the last dim.

        Args:
            x: `(..., 2, dim)`, `x[..., 0, :]` real and `x[..., 1, :]`
                imaginary parts.

        Returns:
            Same shape, `(x_re + i x_im) @ (S_re + i S_im) + b`.
        """
        x_re, x_im = x[..., 0, :], x[..., 1, :]
        out_re = self.real(x_re) - self.imag(x_im) + self.bias[0]
        out_im = self.real(x_im) + self.imag(x_re) + self.bias[1]
        return torch.stack([out_re, out_im], dim=-2)


class FourierGNN(nn.Module):
    """Pure-graph forecaster: hypervariate graph + stacked FGOs.

    Args:
        n_nodes: Number of series.
        input_window: Lag window length.
        output_window: Forecast horizon.
        target_idx: Indices of the output nodes, in return order.
        embed_dim: Width `d` of the per-(variable, lag) node embedding
            and of every FGO (paper's `d`, default matches the
            paper's small-dataset settings).
        n_layers: Stacked FGOs `K` (paper default 3).
        ffn_hidden: Width of the two hidden FFN layers that map the
            `(input_window * n_nodes, embed_dim)` output back to
            `(output_window, n_targets)`.

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
        embed_dim: int = 32,
        n_layers: int = 3,
        ffn_hidden: int = 64,
    ) -> None:
        super().__init__()
        n_targets = len(target_idx)
        n_graph_nodes = input_window * n_nodes

        self.embed = nn.Linear(1, embed_dim)
        self.fgo_layers = nn.ModuleList(
            [_FourierGraphOperator(embed_dim) for _ in range(n_layers)]
        )
        self.ffn = nn.Sequential(
            nn.Linear(n_graph_nodes * embed_dim, ffn_hidden),
            nn.LeakyReLU(),
            nn.Linear(ffn_hidden, ffn_hidden),
            nn.LeakyReLU(),
            nn.Linear(ffn_hidden, output_window * n_targets),
        )
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        self.n_graph_nodes = n_graph_nodes
        self.output_window = output_window
        self.n_targets = n_targets
        # Filled by the last forward pass: (batch, n_graph_nodes,
        # embed_dim) time-domain node representation — see
        # `edge_strength`.
        self.last_node_repr: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stacked FGOs and return the forecast."""
        batch = x.shape[0]
        # (batch, input_window, n_nodes) -> (batch, n_graph_nodes, 1):
        # each (lag, series) pair becomes its own hypervariate node.
        flat = x.reshape(batch, self.n_graph_nodes, 1)
        embedded = self.embed(flat)  # (batch, n_graph_nodes, embed_dim)

        spectrum = torch.fft.fft(embedded, dim=1)
        freq = torch.stack([spectrum.real, spectrum.imag], dim=-2)

        total = freq  # k=0 term: identity FGO, the residual (Eq. 6)
        acc = freq
        for fgo in self.fgo_layers:
            acc = fgo(acc)
            total = total + acc  # noqa: PLR6104 - `total` aliases `freq`

        spectrum_out = torch.complex(total[..., 0, :], total[..., 1, :])
        node_repr = torch.fft.ifft(spectrum_out, dim=1).real
        self.last_node_repr = node_repr.detach()

        out = self.ffn(node_repr.reshape(batch, -1))
        out = out.reshape(batch, self.output_window, self.n_targets)
        return out

    @torch.no_grad()
    def edge_strength(self, x: torch.Tensor) -> torch.Tensor:
        """Node-node correlation of the learned representation.

        Paper §5.4/Appendix E.5: visualise spatial correlations as
        `A = Y Y^T` on the FourierGNN output `Y` (before IDFT's
        residual is folded into the FFN). Here, summed back from
        `(input_window * n_nodes)` hypervariate nodes to the original
        `n_nodes` by max-pooling over lags, so it is directly
        comparable to `MPNNForecaster.edge_strength` — route B1,
        `docs/4_rotas-de-extracao.md`.

        Args:
            x: Batch of input windows, `(batch, input_window, n_nodes)`.

        Returns:
            `(n_nodes, n_nodes)` tensor, entry `[i, j]` the mean
            correlation between any lag of series `i` and any lag of
            series `j`.
        """
        self.forward(x)
        assert self.last_node_repr is not None
        repr_ = self.last_node_repr.mean(dim=0)  # (n_graph_nodes, embed_dim)
        corr = (repr_ @ repr_.transpose(-2, -1)).abs()
        n_nodes = x.shape[-1]
        input_window = self.n_graph_nodes // n_nodes
        corr = corr.view(input_window, n_nodes, input_window, n_nodes)
        return corr.amax(dim=(0, 2))
