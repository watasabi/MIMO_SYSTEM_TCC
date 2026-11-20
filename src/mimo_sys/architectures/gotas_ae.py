"""GOTAS-AE: linear autoencoder per wavelet band, attention in the latent.

Plan: `docs/19_plano-melhoria-gotas.md`. The seq2seq-with-attention idea
of `data/external/seq2seq attention.ipynb`, kept linear in the inputs
so that it can still be read as a graph. Each node is split by the
causal Haar à trous of `GOTAS-W` into `levels` details and one
approximation (`levels=0` keeps the raw node). For every component `m`:

    z^m(t-τ)   = E_m s^m(t-τ)                      encoder: nodes -> latent
    α^m_{k,τ}  = softmax_τ( q^m_k · (K_m z^m(t-τ) + p^m_τ) )
    c^m_k      = sum_τ α^m_{k,τ} z^m(t-τ)            context of step k
    y(t+k)     = anchor + sum_m C_m c^m_k + D u(t+k)  decoder, bands added

The query `q^m_k` of horizon step `k` comes from the known inputs at
`t+k`, the last latent state and a learned step embedding, so it never
reads a later step. Given the attention, the forecast is linear in the
lags: the edge `j -> i` at lag `τ`, step `k` and band `m` has the gain
`α^m_{k,τ} (C_m E_m)_{ij}` (`edge_gain`, `kernel`), and the attention
maps say which hours of the past each step reads.

Why the attention: a softmax context is a convex mix of past latent
states, so a sensor failure in the window moves the forecast at most
by its own latent size instead of being summed along a long kernel (the
failure of `GOTAS` on the level in fold 3; `docs/19`), and the scores
can learn to skip the failed hours. `anchor="last"` forecasts each
output as a change from its last observed value, as the cascade does
for the level.

Shape:
    - Input: `(batch, past + horizon, n_nodes)` (known future in input)
    - Output: `(batch, horizon, len(target_idx))`
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mimo_sys.architectures.gotas_wavelet import atrous

__all__ = ["GOTASAutoencoder"]

DEFAULT_KNOWN_FUTURE = (0, 1, 2, 3)
ANCHORS = ("mean", "last")


class GOTASAutoencoder(nn.Module):
    """Per-band linear autoencoder with attention over the past latents.

    Args:
        n_nodes: Number of series.
        input_window: Length of `x`: past window plus the horizon.
        output_window: Forecast horizon `K`.
        target_idx: Output nodes, in return order.
        levels: Detail levels of the à trous (0: no decomposition).
        latent_dim: Size of each band's latent state.
        attention_dim: Size of the query and key vectors.
        anchor: Outputs relative to the past `mean` or to their `last`
            observed value (inputs are always centred on the mean).
        residual_hidden: Width of the residual MLP on each output's own
            past (0 disables), as in `GOTAS`.
        residual_kind: `mlp` (no closed form) or `chebyshev`: the
            explicit polynomial NARX correction of `GOTAS`,
            `r_i[k] = sum_{l,d} c[i, k, l, d] T_d(z_i(t - l))` over the
            last `residual_lags` centred values of output `i`.
        known_future_idx: Nodes whose future values are real inputs.
    """

    def __init__(  # noqa: PLR0913 - every knob is a config field
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
        *,
        levels: int = 3,
        latent_dim: int = 8,
        attention_dim: int = 8,
        anchor: str = "last",
        residual_hidden: int = 0,
        residual_kind: str = "mlp",
        residual_lags: int = 24,
        residual_degree: int = 3,
        known_future_idx: tuple[int, ...] = DEFAULT_KNOWN_FUTURE,
    ) -> None:
        super().__init__()
        if anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}")
        self.output_window = output_window
        self.past_len = input_window - output_window
        self.levels = levels
        self.anchor = anchor
        self.attention_dim = attention_dim
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        self.register_buffer(
            "known_idx", torch.tensor(known_future_idx, dtype=torch.long)
        )
        m, d, a = levels + 1, latent_dim, attention_dim
        n_t, n_k = len(target_idx), len(known_future_idx)
        self.encoder = nn.Parameter(torch.randn(m, d, n_nodes) / n_nodes)
        self.key = nn.Parameter(torch.randn(m, a, d) / math.sqrt(d))
        self.position = nn.Parameter(0.1 * torch.randn(m, self.past_len, a))
        self.query_input = nn.Parameter(torch.zeros(m, a, n_k))
        self.query_state = nn.Parameter(torch.randn(m, a, d) / math.sqrt(d))
        self.query_step = nn.Parameter(0.1 * torch.randn(m, output_window, a))
        self.decoder = nn.Parameter(0.1 * torch.randn(m, n_t, d))
        self.direct = nn.Parameter(torch.zeros(n_t, n_k))
        self.bias = nn.Parameter(torch.zeros(n_t))
        # Extraction (`equation`): a fixed `(m, K, τ)` attention replaces
        # the computed one, and `attention_log` collects the attention of
        # every forward pass (sum over the batch, count).
        self.fixed_attention: torch.Tensor | None = None
        self.attention_log: list[tuple[torch.Tensor, int]] | None = None
        self.residual = None
        self.residual_coeffs = None
        if residual_kind not in {"mlp", "chebyshev"}:
            raise ValueError(f"unknown residual_kind: {residual_kind!r}")
        if residual_kind == "chebyshev" and residual_hidden > 0:
            self.residual_coeffs = nn.Parameter(
                torch.zeros(
                    n_t,
                    output_window,
                    min(residual_lags, self.past_len),
                    residual_degree,
                )
            )
        elif residual_hidden > 0:
            self.residual = nn.Sequential(
                nn.Linear(self.past_len, residual_hidden),
                nn.Tanh(),
                nn.Linear(residual_hidden, output_window),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Latents `(b, m, τ, d)`, attention `(b, m, K, τ)`, u, centre."""
        past = x[:, : self.past_len]
        mu = past.mean(dim=1, keepdim=True)  # (b, 1, n)
        centred = past - mu
        if self.levels > 0:
            comps = atrous(centred, self.levels)  # (b, τ, n, m)
        else:
            comps = centred.unsqueeze(-1)
        latent = torch.einsum("btnm,mdn->bmtd", comps, self.encoder)
        known = x[:, self.past_len :, self.known_idx]
        u = known - mu[:, :, self.known_idx]  # (b, K, n_k)
        keys = torch.einsum("bmtd,mad->bmta", latent, self.key)
        keys += self.position.unsqueeze(0)
        last = latent[:, :, -1]  # (b, m, d)
        query = (
            torch.einsum("bkn,man->bmka", u, self.query_input)
            + torch.einsum("bmd,mad->bma", last, self.query_state).unsqueeze(2)
            + self.query_step.unsqueeze(0)
        )
        scores = torch.einsum("bmka,bmta->bmkt", query, keys)
        attention = torch.softmax(scores / math.sqrt(self.attention_dim), -1)
        return latent, attention, u, mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        past = x[:, : self.past_len]
        latent, attention, u, mu = self._encode(x)
        if self.fixed_attention is not None:
            attention = self.fixed_attention.expand_as(attention)
        if self.attention_log is not None:
            self.attention_log.append((attention.detach().sum(0), len(x)))
        context = torch.einsum("bmkt,bmtd->bmkd", attention, latent)
        out = torch.einsum("bmkd,mid->bki", context, self.decoder)
        out = out + u @ self.direct.T + self.bias
        target_past = past[..., self.target_idx] - mu[..., self.target_idx]
        if self.residual_coeffs is not None:
            out += self._chebyshev_residual(target_past)
        elif self.residual is not None:
            residual = self.residual(target_past.transpose(1, 2))
            out += residual.transpose(1, 2)
        if self.anchor == "last":
            # The context is centred on the mean: shift it to the last
            # value so that the bands only carry the change from it.
            target = past[..., self.target_idx]
            return out + target[:, -1:]
        return out + mu[:, :, self.target_idx]

    def _chebyshev_residual(self, target_past: torch.Tensor) -> torch.Tensor:
        """`(b, K, i)` explicit polynomial correction, as in `GOTAS`."""
        lags = self.residual_coeffs.shape[2]
        z = target_past[:, -lags:].flip(1).clamp(-1, 1)  # (b, l, i)
        terms = [z]
        if self.residual_coeffs.shape[3] > 1:
            terms.append(2 * z * z - 1)
        for _ in range(3, self.residual_coeffs.shape[3] + 1):
            terms.append(2 * z * terms[-1] - terms[-2])
        basis = torch.stack(terms, dim=-1)  # (b, l, i, d)
        return torch.einsum("blid,ikld->bki", basis, self.residual_coeffs)

    def group_l1(self, alpha: float = 0.5) -> torch.Tensor:
        """Sparse graph: per-edge group of `C_m E_m` plus L1 on the maps."""
        per_term = (
            self.encoder.abs().sum()
            + self.decoder.abs().sum()
            + self.direct.abs().sum()
        )
        per_edge = (self.edge_gain().pow(2).sum(0) + 1e-12).sqrt().sum()
        return alpha * per_term + (1 - alpha) * per_edge

    def edge_gain(self) -> torch.Tensor:
        """`(m, n_targets, n_nodes)`: `C_m E_m`, the graph of each band."""
        return torch.einsum("mid,mdn->min", self.decoder, self.encoder)

    @torch.no_grad()
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """`(batch, m, K, τ)` attention maps; `τ = 0` is the oldest lag."""
        return self._encode(x)[1]

    @torch.no_grad()
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        """`(batch, K, n_targets, n_nodes, m, τ)` gains on the band lags.

        `out[b, k, i] = sum_{j,m,τ} kernel * s^m_j(τ) + D u + anchor`
        (plus the residual MLP when enabled), with `s^m_j` the band `m`
        of node `j`, centred on the window mean.
        """
        attention = self._encode(x)[1]
        return torch.einsum("bmkt,min->bkinmt", attention, self.edge_gain())

    @torch.no_grad()
    def equation(  # noqa: PLR0914 - one step per gain block
        self, attention: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact affine equation of the model with a fixed attention.

        With `attention` `(m, K, τ)` fixed (e.g. the validation mean,
        from `attention_log`) and without the residual MLP, the model
        is affine in its input:

            y_i(t+k) = a_i + b_ik + sum_{j,τ} h[k,i,τ,j] s_j(t-P+τ)
                                + sum_{l,q} g[k,i,l,q] u_q(t+l)

        where `a_i` is the anchor (the window mean or the last value of
        output `i`, a fixed rule) and `h` holds only the learned part,
        with the window mean and the bands folded in. Every gain is
        read off by feeding unit inputs. A Chebyshev residual, when
        present, adds its explicit term `residual_coeffs` on top; an MLP
        residual has no closed form and is left out.

        Returns:
            `b` `(K, n_targets)`, `h` `(K, n_targets, P, n_nodes)` and
            `g` `(K, n_targets, K, n_known)`; `τ = P - 1` is the last
            hour before the forecast.
        """
        residual, fixed = self.residual, self.fixed_attention
        coeffs = self.residual_coeffs
        self.residual, self.fixed_attention = None, attention
        self.residual_coeffs = None
        try:
            p, k = self.past_len, self.output_window
            n = self.encoder.shape[-1]
            n_k = len(self.known_idx)
            device = self.bias.device
            basis = torch.zeros(1 + p * n + k * n_k, p + k, n, device=device)
            rows = torch.arange(p * n, device=device)
            basis[1 + rows, rows // n, rows % n] = 1.0
            rows = torch.arange(k * n_k, device=device)
            cols = self.known_idx[rows % n_k]
            basis[1 + p * n + rows, p + rows // n_k, cols] = 1.0
            out = self(basis)  # (1 + ..., K, i)
        finally:
            self.residual, self.fixed_attention = residual, fixed
            self.residual_coeffs = coeffs
        offset = out[0]
        gains = (out[1:] - offset).permute(1, 2, 0)  # (K, i, inputs)
        h = gains[..., : p * n].reshape(k, -1, p, n).clone()
        g = gains[..., p * n :].reshape(k, -1, k, n_k)
        for i, node in enumerate(self.target_idx.tolist()):
            if self.anchor == "last":
                h[:, i, -1, node] -= 1.0
            else:
                h[:, i, :, node] -= 1.0 / p
        return offset, h, g

    @staticmethod
    def subgraph(
        h: torch.Tensor,
        g: torch.Tensor,
        known_idx: torch.Tensor,
        keep: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Edge strength `(n_targets, n_nodes)` and the kept edges.

        The strength of `j -> i` is the norm of all its gains in the
        equation (every lag, step and known future value). An edge is
        kept when it reaches `keep` times the strongest edge into `i`.
        """
        strength = h.pow(2).sum((0, 2))  # (i, n)
        strength[:, known_idx] += g.pow(2).sum((0, 2))
        strength = strength.sqrt()
        kept = strength >= keep * strength.amax(1, keepdim=True)
        return strength, kept
