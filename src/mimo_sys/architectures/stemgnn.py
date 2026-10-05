"""StemGNN — Spectral Temporal GNN (Cao et al., NeurIPS 2020).

Reference implementation of `docs/pappers/StemGNN_2020.pdf`, kept as the
spectral baseline. It is **not** an extraction vehicle: after the graph
transform the computation lives in a spectral basis of the learned
Laplacian, so a distilled expression would be written over spectral
modes rather than over `u1..u4` / `y1..y3`. See §2 of
`docs/3_investigacao-arquiteturas.md`.

Note on the graph transform. The paper describes a GFT via
eigendecomposition; `microsoft/StemGNN` instead uses Chebyshev
polynomials of the Laplacian, and that choice turns out to be load
bearing at this problem's scale, not just a speed optimisation. With
`n_nodes = 7` the learned attention row is almost exactly uniform
(1/7 each), so the symmetric normalised Laplacian has eigenvalue 0 once
and eigenvalue 1 with multiplicity 6 — consecutive gaps of order 1e-4.
Eigenvectors inside that near-degenerate subspace are numerically
arbitrary, and because the Spe-Seq cell sits between the forward and
inverse transform the arbitrary signs do not cancel: an exact-GFT
implementation is not reproducible run to run (measured: up to 0.21
absolute difference between two calls of the same weights). Chebyshev
polynomials are functions of the Laplacian itself and never reference an
eigenbasis, so they are invariant to that ambiguity. This is covered by
`test_stemgnn_is_reproducible`.
"""

import torch
from torch import nn

MIN_CHEB_ORDER = 2

__all__ = [
    "LatentCorrelationLayer",
    "SpeSeqCell",
    "StemGNN",
    "chebyshev_basis",
]


class LatentCorrelationLayer(nn.Module):
    """Learns the adjacency from the data, with no prior topology.

    A GRU runs over the *node* axis (as in the authors' code: sequence
    length `n_nodes`, features `input_window`), then single-head
    self-attention over the node embeddings gives
    `W = softmax(Q K^T / sqrt(d))`.

    Args:
        input_window: Lag window length, the GRU input size.
        hidden_size: GRU hidden size and attention dimension.
    """

    def __init__(self, input_window: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_window, hidden_size, batch_first=False)
        self.to_query = nn.Linear(hidden_size, hidden_size)
        self.to_key = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attention adjacency, `(n_nodes, n_nodes)`.

        Args:
            x: `(batch, input_window, n_nodes)`.
        """
        # [batch, window, nodes] -> [nodes, batch, window]
        seq = x.permute(2, 0, 1)
        embedded, _ = self.gru(seq)
        embedded = embedded.permute(1, 0, 2)  # [batch, nodes, hidden]
        query = self.to_query(embedded)
        key = self.to_key(embedded)
        scale = self.hidden_size**0.5
        scores = torch.matmul(query, key.transpose(1, 2)) / scale
        attention = torch.softmax(scores, dim=-1)
        return attention.mean(dim=0)


def normalised_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetric normalised Laplacian of a weighted adjacency.

    The attention matrix is symmetrised first, so the result is
    symmetric and its Chebyshev polynomials stay symmetric too.

    Args:
        adjacency: `(n_nodes, n_nodes)` non-negative weights.

    Returns:
        `(n_nodes, n_nodes)` Laplacian.
    """
    sym = 0.5 * (adjacency + adjacency.transpose(0, 1))
    degree = sym.sum(dim=1)
    inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-7))
    identity = torch.eye(sym.shape[0], device=sym.device)
    return identity - inv_sqrt @ sym @ inv_sqrt


def chebyshev_basis(adjacency: torch.Tensor, order: int = 4) -> torch.Tensor:
    """Chebyshev polynomials of the normalised Laplacian.

    `T_0 = I`, `T_1 = L`, `T_k = 2 L T_{k-1} - T_{k-2}`. Used instead of
    the eigenbasis for the reason given in the module docstring.

    Args:
        adjacency: `(n_nodes, n_nodes)` non-negative weights.
        order: Number of polynomials to return (at least 2).

    Returns:
        `(order, n_nodes, n_nodes)` stack of polynomials.
    """
    if order < MIN_CHEB_ORDER:
        raise ValueError(f"order must be >= {MIN_CHEB_ORDER}")
    laplacian = normalised_laplacian(adjacency)
    n = laplacian.shape[0]
    polynomials = [
        torch.eye(n, device=laplacian.device),
        laplacian,
    ]
    for _ in range(order - 2):
        polynomials.append(2 * laplacian @ polynomials[-1] - polynomials[-2])
    return torch.stack(polynomials, dim=0)


class SpeSeqCell(nn.Module):
    """DFT -> Conv1D -> GLU -> inverse DFT over the time axis.

    Real and imaginary parts are stacked as channels so a real-valued
    convolution can mix them, then split back before the inverse
    transform.

    Args:
        channels: Number of channels per node.
        kernel_size: Convolution width over the frequency axis (odd).
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.channels = channels
        self.conv = nn.Conv1d(
            2 * channels,
            4 * channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.glu = nn.GLU(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of `(batch, nodes, channels, window)`."""
        batch, nodes, channels, window = x.shape
        freq = torch.fft.rfft(x, dim=-1)
        stacked = torch.cat([freq.real, freq.imag], dim=2)
        stacked = stacked.reshape(batch * nodes, 2 * channels, -1)
        filtered = self.glu(self.conv(stacked))
        filtered = filtered.reshape(batch, nodes, 2 * channels, -1)
        real, imag = filtered.split(channels, dim=2)
        return torch.fft.irfft(torch.complex(real, imag), n=window, dim=-1)


class _StemGNNBlock(nn.Module):
    """Multi-order graph conv -> Spe-Seq -> forecast/backcast."""

    def __init__(
        self,
        input_window: int,
        output_window: int,
        cheb_order: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.cell = SpeSeqCell(cheb_order, kernel_size)
        self.order_weight = nn.Parameter(
            torch.full((cheb_order,), 1.0 / cheb_order)
        )
        self.forecast = nn.Linear(input_window, output_window)
        self.backcast = nn.Linear(input_window, input_window)

    def forward(
        self, x: torch.Tensor, basis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: x `(batch, nodes, window)`; basis `(order, n, n)`.

        Returns:
            `(forecast, backcast)`, both `(batch, nodes, ...)`.
        """
        # One spectral channel per Chebyshev order.
        spectral = torch.einsum("knm,bmt->bnkt", basis, x)
        processed = self.cell(spectral)
        collapsed = torch.einsum("bnkt,k->bnt", processed, self.order_weight)
        return self.forecast(collapsed), self.backcast(collapsed)


class StemGNN(nn.Module):
    """Spectral Temporal GNN forecaster.

    Args:
        n_nodes: Number of series.
        input_window: Lag window length.
        output_window: Forecast horizon.
        target_idx: Indices of the output nodes, in return order.
        hidden_size: GRU/attention width of the correlation layer.
        cheb_order: Chebyshev polynomials of the Laplacian to use, which
            is also the number of spectral channels in a block.
        n_blocks: Stacked blocks, chained by the backcast residual.
        kernel_size: Spe-Seq convolution width (odd).

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
        hidden_size: int = 64,
        cheb_order: int = 4,
        n_blocks: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.correlation = LatentCorrelationLayer(input_window, hidden_size)
        self.blocks = nn.ModuleList(
            [
                _StemGNNBlock(
                    input_window,
                    output_window,
                    cheb_order,
                    kernel_size,
                )
                for _ in range(n_blocks)
            ]
        )
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        self.n_nodes = n_nodes
        self.cheb_order = cheb_order
        # Filled by the last forward pass: [nodes, nodes].
        self.last_adjacency: torch.Tensor | None = None
        # Filled by the last forward pass: [batch, nodes, window].
        self.last_backcast: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block stack and return the forecast."""
        adjacency = self.correlation(x)
        self.last_adjacency = adjacency
        basis = chebyshev_basis(adjacency, self.cheb_order)

        residual = x.transpose(1, 2)  # [batch, nodes, window]
        total = None
        reconstruction = None
        for block in self.blocks:
            forecast, backcast = block(residual, basis)
            total = forecast if total is None else total + forecast
            reconstruction = (
                backcast
                if reconstruction is None
                else reconstruction + backcast
            )
            # Not `-=`: the previous block's Linear saved this
            # tensor for backward, so an in-place write would
            # break autograd.
            residual = residual - backcast  # noqa: PLR6104

        assert total is not None
        self.last_backcast = reconstruction
        out = total.index_select(1, self.target_idx)
        return out.transpose(1, 2)

    def backcast_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss of the backcast (auto-encoder) branch.

        StemGNN trains forecasting and backcasting jointly; without this
        term the last block's backcast head gets no gradient at all.
        Add it to the forecast loss in the training step.

        Args:
            x: The same `(batch, input_window, n_nodes)` passed to
                `forward`.
        """
        if self.last_backcast is None:
            raise RuntimeError("call forward() before backcast_loss()")
        return nn.functional.mse_loss(self.last_backcast, x.transpose(1, 2))
