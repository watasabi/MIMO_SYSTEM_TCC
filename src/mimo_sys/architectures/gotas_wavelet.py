"""GOTAS-W: the GOTAS graph with one filter per edge and wavelet component.

Plan: `docs/18_plano-gotas-w.md`.

GOTAS takes every node to the frequency domain and multiplies each edge
by a response `H_ij(f)`. GOTAS-W keeps the graph and the recursive
simulation but replaces the FFT by a causal Haar à trous decomposition
(the same as `training.features.atrous_components`): each node `j` is
split into `levels` details and one approximation, which add up to the
node exactly, and every edge `j -> i` has a short filter for each
component `m`, with lags spaced by the component's scale `d_m`:

    y_i[t] = b_i + sum_j sum_m sum_l a[i, j, m, l] * c_{j,m}[t - l * d_m]

Row `t` of a component only uses rows `<= t`, so the forecast never
reads the future. Without the optional window limit (`detail_limit`)
the model stays linear in the inputs: `impulse_response` returns the
single kernel each edge amounts to, so the extraction of GOTAS (impulse
responses, transfer functions) still applies.

Shape:
    - Input: `(batch, input_window, n_nodes)`
    - Output: `(batch, output_window, len(target_idx))`
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["GOTASWavelet", "atrous"]

# freq_b1..3, vazao_entrada in `training.data.FEATURE_COLS` order.
DEFAULT_KNOWN_FUTURE = (0, 1, 2, 3)
CENTRES = ("mean", "median", "last")
MAD_TO_STD = 1.4826


def atrous(x: torch.Tensor, levels: int) -> torch.Tensor:
    """Causal Haar à trous of `(batch, time, nodes)` along time.

    Same recursion as `training.features.atrous_components`: the first
    rows repeat the first value as history, and the components add up
    to `x` exactly.

    Returns:
        `(batch, time, nodes, levels + 1)`: `[w_1, ..., w_J, c_J]`.
    """
    c = x
    comps = []
    for j in range(1, levels + 1):
        lag = 2 ** (j - 1)
        first = c[:, :1].expand(-1, min(lag, c.shape[1]), -1)
        shifted = torch.cat([first, c[:, :-lag]], dim=1)[:, : c.shape[1]]
        smooth = 0.5 * (c + shifted)
        comps.append(c - smooth)
        c = smooth
    comps.append(c)
    return torch.stack(comps, dim=-1)


class GOTASWavelet(nn.Module):
    """Graph of per-component filters between all series.

    Args:
        n_nodes: Number of series.
        input_window: Length of `x` along time: the past window, plus
            the known future horizon when `future_in_input`.
        output_window: Forecast horizon.
        target_idx: Output nodes, in return order.
        levels: Detail levels of the à trous; `levels + 1` components.
        lags: Filter taps per component, spaced by its scale.
        residual_hidden: Width of the nonlinear residual MLP on each
            output's own past (0 disables), as in `GOTAS`.
        future_in_input: The last `output_window` steps of `x` are the
            horizon (known inputs, zeroed outputs).
        known_future_idx: Nodes whose future values are real inputs.
        centre: What each window is centred on: the past `mean` (as in
            `GOTAS`), the past `median` (robust to a failure inside the
            window), or `last`: the outputs on their last observed value
            and the inputs on their mean.
        detail_limit: When `> 0`, the details of the past are limited to
            `± detail_limit` robust standard deviations of each window
            (median absolute deviation), which cuts sensor spikes scale
            by scale. Makes the model nonlinear in the inputs.
    """

    def __init__(  # noqa: PLR0913 - every knob is a config field
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
        *,
        levels: int = 4,
        lags: int = 4,
        residual_hidden: int = 0,
        future_in_input: bool = True,
        known_future_idx: tuple[int, ...] = DEFAULT_KNOWN_FUTURE,
        centre: str = "mean",
        detail_limit: float = 0.0,
    ) -> None:
        super().__init__()
        if centre not in CENTRES:
            raise ValueError(f"centre must be one of {CENTRES}")
        self.n_nodes = n_nodes
        self.output_window = output_window
        self.future_in_input = future_in_input
        self.seq_len = input_window + output_window * (not future_in_input)
        self.past_len = self.seq_len - output_window
        self.levels = levels
        self.centre = centre
        self.detail_limit = detail_limit
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        future_mask = torch.zeros(n_nodes)
        future_mask[list(known_future_idx)] = 1.0
        self.register_buffer("future_mask", future_mask)

        # Component m (details 1..J, then the approximation) has scale
        # d_m = 2^(m-1); the approximation shares the last detail's.
        scales = [2**m for m in range(levels)] + [2 ** (levels - 1)]
        offsets = torch.tensor(
            [[lag * d for lag in range(lags)] for d in scales],
            dtype=torch.long,
        )
        self.register_buffer("offsets", offsets)  # (component, lag)
        n_targets = len(target_idx)
        # (target, sender, component, lag)
        self.weight = nn.Parameter(
            0.01 * torch.randn(n_targets, n_nodes, levels + 1, lags)
        )
        self.bias = nn.Parameter(torch.zeros(n_targets))
        self.residual = None
        if residual_hidden > 0:
            self.residual = nn.Sequential(
                nn.Linear(self.past_len, residual_hidden),
                nn.Tanh(),
                nn.Linear(residual_hidden, output_window),
            )
            # Start purely linear: the residual must earn its share.
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        self._last_linear: torch.Tensor | None = None
        self._last_residual: torch.Tensor | None = None

    def _centre(self, past: torch.Tensor) -> torch.Tensor:
        """`(batch, 1, n_nodes)` value each window is centred on."""
        if self.centre == "median":
            return past.median(dim=1, keepdim=True).values
        mu = past.mean(dim=1, keepdim=True)
        if self.centre == "last":
            mu = mu.clone()
            mu[:, :, self.target_idx] = past[:, -1:, self.target_idx]
        return mu

    def _limit_details(self, past: torch.Tensor) -> torch.Tensor:
        """Past with each detail clipped to `± detail_limit` robust stds."""
        comps = atrous(past, self.levels)
        details = comps[..., :-1]
        centre = details.median(dim=1, keepdim=True).values
        mad = (details - centre).abs().median(dim=1, keepdim=True).values
        bound = self.detail_limit * MAD_TO_STD * mad.clamp_min(1e-6)
        limited = torch.maximum(torch.minimum(details, bound), -bound)
        return limited.sum(dim=-1) + comps[..., -1]

    def _step(self, norm: torch.Tensor, pos: int) -> torch.Tensor:
        """`(batch, n_targets)` one-step output at position `pos`."""
        comps = atrous(norm[:, : pos + 1], self.levels)  # (b, t, n, m)
        index = (pos - self.offsets).clamp_min(0)  # (m, l)
        b, _, n, m = comps.shape
        lagged = torch.gather(
            comps.permute(0, 2, 3, 1),  # (b, n, m, t)
            3,
            index[None, None].expand(b, n, m, -1),
        )  # (b, n, m, l)
        return torch.einsum("bnml,inml->bi", lagged, self.weight) + self.bias

    def _prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Centred sequence (unknown horizon cells zeroed) and its centre."""
        if self.future_in_input:
            seq = x
        else:
            pad = x.new_zeros(x.shape[0], self.output_window, x.shape[2])
            seq = torch.cat([x, pad], dim=1)
        mu = self._centre(seq[:, : self.past_len])
        horizon_mask = torch.ones_like(seq)
        horizon_mask[:, self.past_len :] = (
            self.future_mask if self.future_in_input else 0.0
        )
        norm = (seq - mu) * horizon_mask
        if self.detail_limit > 0:
            norm = torch.cat(
                [
                    self._limit_details(norm[:, : self.past_len]),
                    norm[:, self.past_len :],
                ],
                dim=1,
            )
        return norm, mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recursive simulation over the horizon, as in `GOTAS.forward`."""
        norm, mu = self._prepare(x)
        n_targets = len(self.target_idx)
        scatter = torch.zeros(n_targets, self.n_nodes, device=x.device)
        scatter[torch.arange(n_targets), self.target_idx] = 1.0
        keep = 1.0 - scatter.sum(dim=0)
        steps = []
        for k in range(self.output_window):
            pos = self.past_len + k
            step = self._step(norm, pos)
            steps.append(step)
            row = norm[:, pos] * keep + step @ scatter
            norm = torch.cat(
                [norm[:, :pos], row[:, None], norm[:, pos + 1 :]], dim=1
            )
        linear = torch.stack(steps, dim=1)  # (b, H, i)
        self._last_linear = linear
        self._last_residual = None
        out = linear
        if self.residual is not None:
            target_past = norm[:, : self.past_len, self.target_idx]
            residual = self.residual(target_past.transpose(1, 2))
            self._last_residual = residual.transpose(1, 2)
            out = linear + self._last_residual
        return out + mu[:, :, self.target_idx]

    def group_l1(self, alpha: float = 0.5) -> torch.Tensor:
        """Sparse-group lasso: single taps, and whole (edge, component).

        `alpha` weighs single taps against whole components of an edge,
        so a component an edge does not use dies as a group.
        """
        per_tap = self.weight.abs().sum()
        per_group = self.weight.pow(2).sum(dim=-1).sqrt().sum()
        return alpha * per_tap + (1 - alpha) * per_group

    @torch.no_grad()
    def component_strength(self) -> torch.Tensor:
        """`(n_targets, n_nodes, n_components)` norm of each filter."""
        return self.weight.norm(dim=-1)

    @torch.no_grad()
    def impulse_response(self, i: int, j: int) -> torch.Tensor:
        """Kernel of edge `j -> target_idx[i]`, as one linear filter.

        Index `tau` is the weight of `s_j[t - tau]` in `y_i[t]`: the sum
        over components of each filter convolved with the component's
        own à trous kernel.
        """
        n_comp, n_lags = self.offsets.shape
        support = 2**self.levels
        length = int(self.offsets.max()) + support
        # Zeros before the impulse: `atrous` repeats the first row as
        # history, so an impulse at row 0 would look like a step.
        impulse = torch.zeros(1, support + length, 1)
        impulse[0, support, 0] = 1.0
        psi = atrous(impulse, self.levels)[0, support:, 0]  # (time, comp)
        kernel = torch.zeros(length)
        for m in range(n_comp):
            for lag in range(n_lags):
                shift = int(self.offsets[m, lag])
                kernel[shift:] += (
                    self.weight[i, j, m, lag] * psi[: length - shift, m]
                )
        return kernel

    def nonlinearity_share(self) -> float:
        """Share of the last output energy from the nonlinear residual."""
        if self._last_residual is None:
            return 0.0
        lin = self._last_linear.pow(2).sum()
        res = self._last_residual.pow(2).sum()
        return (res / (lin + res + 1e-12)).item()
