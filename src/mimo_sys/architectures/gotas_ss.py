"""GOTAS-SS: linear state-space autoencoder over the graph of series.

Plan: `docs/19_plano-melhoria-gotas.md`. A symmetric, fully linear
encoder-latent-decoder (a Koopman-style autoencoder whose maps are all
linear), so the whole forecaster is a state-space model and is read off
as one:

    x_0     = E vec(s(t-1), ..., s(t-L))          (encoder, centred lags)
    x_{k+1} = A x_k + B u(t+k)                    (latent dynamics)
    y(t+k)  = C x_k + D u(t+k) + mu_y              (decoder)

`A` is kept stable (spectral norm < 1). Poles are the eigenvalues of
`A` and the transfer function is `C (zI - A)^-1 B + D` (`transfer`). The
optional residual MLP on each output's own past is the same as in
`GOTAS`.

Shape:
    - Input: `(batch, past + horizon, n_nodes)` (known future in input)
    - Output: `(batch, horizon, len(target_idx))`
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["GOTASStateSpace"]

DEFAULT_KNOWN_FUTURE = (0, 1, 2, 3)


class GOTASStateSpace(nn.Module):
    """Linear state-space forecaster with a linear encoder.

    Args:
        n_nodes: Number of series.
        input_window: Length of `x`: past window plus the horizon.
        output_window: Forecast horizon `K`.
        target_idx: Output nodes, in return order.
        state_dim: Size of the latent state `d`.
        encoder_lags: Past rows of every node read by the encoder.
        residual_hidden: Width of the residual MLP (0 disables).
        known_future_idx: Nodes whose future values are real inputs.
    """

    def __init__(  # noqa: PLR0913 - every knob is a config field
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
        *,
        state_dim: int = 8,
        encoder_lags: int = 24,
        residual_hidden: int = 0,
        known_future_idx: tuple[int, ...] = DEFAULT_KNOWN_FUTURE,
    ) -> None:
        super().__init__()
        self.output_window = output_window
        self.past_len = input_window - output_window
        self.encoder_lags = min(encoder_lags, self.past_len)
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        self.register_buffer(
            "known_idx", torch.tensor(known_future_idx, dtype=torch.long)
        )
        n_t, n_k = len(target_idx), len(known_future_idx)
        self.encoder = nn.Linear(n_nodes * self.encoder_lags, state_dim)
        self.a_raw = nn.Parameter(0.5 * torch.eye(state_dim))
        self.b = nn.Parameter(0.01 * torch.randn(state_dim, n_k))
        self.c = nn.Parameter(0.1 * torch.randn(n_t, state_dim))
        self.d = nn.Parameter(torch.zeros(n_t, n_k))
        self.residual = None
        if residual_hidden > 0:
            self.residual = nn.Sequential(
                nn.Linear(self.past_len, residual_hidden),
                nn.Tanh(),
                nn.Linear(residual_hidden, output_window),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def state_matrix(self) -> torch.Tensor:
        """`A` scaled to spectral norm below 1 (a stable latent)."""
        norm = torch.linalg.matrix_norm(self.a_raw, ord=2)
        return self.a_raw * (0.99 / torch.clamp(norm, min=0.99))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        past = x[:, : self.past_len]
        mu = past.mean(dim=1, keepdim=True)  # (b, 1, n), as in GOTAS
        lags = (past[:, -self.encoder_lags :] - mu).flip(1)
        state = self.encoder(lags.reshape(x.shape[0], -1))  # (b, d)
        known = (
            x[:, self.past_len :, self.known_idx] - mu[:, :, self.known_idx]
        )
        a = self.state_matrix()
        outputs = []
        for k in range(self.output_window):
            u = known[:, k]
            outputs.append(state @ self.c.T + u @ self.d.T)
            state = state @ a.T + u @ self.b.T
        out = torch.stack(outputs, dim=1)  # (b, K, i)
        if self.residual is not None:
            target_past = past[..., self.target_idx] - mu[..., self.target_idx]
            residual = self.residual(target_past.transpose(1, 2))
            out += residual.transpose(1, 2)
        return out + mu[:, :, self.target_idx]

    def group_l1(self, alpha: float = 0.5) -> torch.Tensor:
        """Sparse input and output maps: which inputs drive which states."""
        per_term = self.b.abs().sum() + self.c.abs().sum() + self.d.abs().sum()
        per_state = (
            (self.b.pow(2).sum(1) + self.c.pow(2).sum(0) + 1e-12).sqrt().sum()
        )
        return alpha * per_term + (1 - alpha) * per_state

    @torch.no_grad()
    def poles(self) -> torch.Tensor:
        """Eigenvalues of `A`: the poles of the latent dynamics."""
        return torch.linalg.eigvals(self.state_matrix())

    @torch.no_grad()
    def transfer(self, z: torch.Tensor) -> torch.Tensor:
        """`(n_targets, n_known)` transfer `C (zI - A)^-1 B + D` at `z`."""
        a = self.state_matrix().to(torch.complex64)
        eye = torch.eye(a.shape[0], dtype=torch.complex64)
        resolvent = torch.linalg.inv(z * eye - a)
        c = self.c.to(torch.complex64)
        b = self.b.to(torch.complex64)
        return c @ resolvent @ b + self.d.to(torch.complex64)
