"""Spectral transfer-function GNN: each edge is a learned frequency response.

Plan and motivation: `docs/changes/2026-10-20-spectral-tf-net.md`.

Node `j`'s sequence (its past window, plus — for the exogenous inputs —
the known future horizon) is taken to the frequency domain, each edge
`j -> i` multiplies it by a learned complex response `H_ij(f)`
(FilterNet, 2411.01623; FITS, 2307.03756), the contributions are
summed over senders, and the inverse FFT brings the result back to
time. That is the **one-step** model, iterated over the horizon with
the predictions written back (recursive ARX simulation, see
`forward`):

    y_i[t] = iRFFT( sum_j H_ij(f) * RFFT(s_j) )[t] + b_i

a MIMO linear transfer-function model (the ARX/ARMAX class of
the reference paper) written as a graph of filters, so extraction
needs no symbolic regression:

- **Route A**: `impulse_response(i, j)` is `iRFFT(H_ij)`, the literal
  convolution kernel of the equation.
- **Route B**: `band_strength()` is `|H_ij|` per frequency band — a
  coupling graph *per band* — and `group_l1()` is a sparse-group lasso
  over (edge, band) (iCSNN, 2610.06630), so whole edges or single bands
  of an edge die during training.

`H_ij` is parametrised by `n_bands` complex values at log-spaced band
centres, linearly interpolated over the FFT bins: few parameters
(n≈1.5k training rows) and a smooth response by construction.

Pump-regime switching (`n_regimes > 1`, FreqMoE-style, 2501.15125):
a softmax gate on the pump channels mixes `n_regimes` banks of `H`, so
each operating regime gets its own — still readable — transfer
function (`regime_weights`).

Centring (no scaling) is done inside, on the **past part only**: the future
part of the target channels is zero by contract
(`training.data.with_future_exog`), so an external RevIN would see
those zeros. Train this model with `revin=None`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["SpectralTFNet"]

# freq_b1..3, vazao_entrada in `training.data.FEATURE_COLS` order.
DEFAULT_KNOWN_FUTURE = (0, 1, 2, 3)
DEFAULT_PUMPS = (0, 1, 2)
MIN_BANDS = 2


def _band_interpolation(n_bins: int, n_bands: int) -> torch.Tensor:
    """`(n_bins, n_bands)` linear interpolation from log-spaced centres.

    Centre 0 sits on DC, the rest are log-spaced up to Nyquist, so the
    slow components (where the flow's drift lives) get as many
    parameters as the fast ones. Every bin gets weights summing to 1.
    """
    if n_bands < MIN_BANDS or n_bands > n_bins:
        raise ValueError(f"n_bands must be in [{MIN_BANDS}, {n_bins}]")
    centres = torch.cat(
        [
            torch.zeros(1),
            torch.logspace(0, math.log10(n_bins - 1), n_bands - 1),
        ]
    )
    centres[-1] = n_bins - 1
    bins = torch.arange(n_bins, dtype=torch.float32)
    left = torch.searchsorted(centres, bins, right=True) - 1
    left = left.clamp(0, n_bands - 2)
    span = (centres[left + 1] - centres[left]).clamp_min(1e-6)
    frac = ((bins - centres[left]) / span).clamp(0.0, 1.0)
    weights = torch.zeros(n_bins, n_bands)
    rows = torch.arange(n_bins)
    weights[rows, left] = 1 - frac
    weights[rows, left + 1] += frac
    return weights


class SpectralTFNet(nn.Module):
    """Graph of learned frequency responses between all series.

    Args:
        n_nodes: Number of series.
        input_window: Length of `x` along time — the past window, plus
            the known future horizon when `future_in_input`.
        output_window: Forecast horizon.
        target_idx: Output nodes, in return order.
        n_bands: Complex parameters per edge (per regime).
        n_regimes: Banks of responses mixed by the pump gate; 1 turns
            the gate off.
        residual_hidden: Width of a small nonlinear residual MLP,
            shared by the targets and applied to each one's own past
            (0 disables).
        future_in_input: The last `output_window` steps of `x` are the
            horizon (known inputs, zeroed outputs). When False, `x` is
            the past only and the horizon is zero-padded.
        known_future_idx: Nodes whose future values are real inputs.
        pump_idx: Nodes fed to the regime gate.
        input_nonlinearity_degree: Hammerstein input edges: each pump
            channel `u` goes through `u + sum_k c_k T_k(2u - 1)`
            (Chebyshev, `k = 2..degree`, the pump curve) before the
            linear filters; 0 or 1 disables it. Starts at `c = 0`, i.e.
            linear, and the polynomial is read off `hammerstein_coeffs`.
            Chebyshev basis as in foreblocks' KAN layers
            (`models/kan/poly/chebyshev.py`, MIT, L. Seman).
        regime_per_step: Pick the response bank at **every horizon
            step** from `gate_idx` at that step (a switched ARX: one
            transfer-function matrix per regime), instead of one mix per
            window from the window mean of `pump_idx`.
        gate_idx: Nodes read by the per-step gate (default `pump_idx`),
            e.g. a known-future pressure setpoint.
        residual_kind: `mlp` (small MLP, no closed form) or `chebyshev`:
            `r_i[h] = sum_{l,k} c[i, h, l, k] T_k(z_i(t - l))` over the
            last `residual_lags` centred values of output `i`, degree
            `residual_degree` — an explicit polynomial NARX correction,
            zero-initialised like the MLP.

    Shape:
        - Input: `(batch, input_window, n_nodes)`
        - Output: `(batch, output_window, len(target_idx))`
    """

    def __init__(  # noqa: PLR0913 - every knob is a config field
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
        *,
        n_bands: int = 8,
        n_regimes: int = 1,
        residual_hidden: int = 0,
        future_in_input: bool = True,
        known_future_idx: tuple[int, ...] = DEFAULT_KNOWN_FUTURE,
        pump_idx: tuple[int, ...] = DEFAULT_PUMPS,
        input_nonlinearity_degree: int = 0,
        regime_per_step: bool = False,
        gate_idx: tuple[int, ...] | None = None,
        residual_kind: str = "mlp",
        residual_lags: int = 24,
        residual_degree: int = 3,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.output_window = output_window
        self.future_in_input = future_in_input
        self.seq_len = input_window + output_window * (not future_in_input)
        self.past_len = self.seq_len - output_window
        self.n_fft = 2 * self.seq_len  # zero-padded: linear, not circular
        self.n_regimes = n_regimes
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        self.register_buffer(
            "pump_idx", torch.tensor(pump_idx, dtype=torch.long)
        )
        future_mask = torch.zeros(n_nodes)
        future_mask[list(known_future_idx)] = 1.0
        self.register_buffer("future_mask", future_mask)

        n_bins = self.n_fft // 2 + 1
        self.register_buffer("interp", _band_interpolation(n_bins, n_bands))
        n_targets = len(target_idx)
        # (regime, target, sender, band, re/im)
        self.response = nn.Parameter(
            0.01 * torch.randn(n_regimes, n_targets, n_nodes, n_bands, 2)
        )
        self.bias = nn.Parameter(torch.zeros(n_targets))
        self.regime_per_step = regime_per_step and n_regimes > 1
        gate_nodes = tuple(pump_idx if gate_idx is None else gate_idx)
        # Derived from the constructor: not saved, so older checkpoints
        # (without it) still load.
        self.register_buffer(
            "gate_idx",
            torch.tensor(gate_nodes, dtype=torch.long),
            persistent=False,
        )
        self.gate = (
            nn.Linear(len(gate_nodes), n_regimes) if n_regimes > 1 else None
        )
        self.hammerstein_coeffs = None
        if input_nonlinearity_degree > 1:
            # (pump, k = 2..degree)
            self.hammerstein_coeffs = nn.Parameter(
                torch.zeros(len(pump_idx), input_nonlinearity_degree - 1)
            )
        self.residual = None
        self.residual_coeffs = None
        if residual_kind not in {"mlp", "chebyshev"}:
            raise ValueError(f"unknown residual_kind: {residual_kind!r}")
        if residual_kind == "chebyshev" and residual_hidden > 0:
            self.residual_coeffs = nn.Parameter(
                torch.zeros(
                    n_targets, output_window, residual_lags, residual_degree
                )
            )
        elif residual_hidden > 0:
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
        self._last_mixed: torch.Tensor | None = None
        self._last_mu: torch.Tensor | None = None
        self._last_past: torch.Tensor | None = None

    def _sequence(self, x: torch.Tensor) -> torch.Tensor:
        """`(batch, seq_len, n_nodes)` with the horizon in place."""
        if self.future_in_input:
            return self._hammerstein(x)
        pad = x.new_zeros(x.shape[0], self.output_window, x.shape[2])
        return self._hammerstein(torch.cat([x, pad], dim=1))

    def _hammerstein(self, seq: torch.Tensor) -> torch.Tensor:
        """Static polynomial on the pump channels (identity if off)."""
        if self.hammerstein_coeffs is None:
            return seq
        u = seq[..., self.pump_idx]
        z = (2 * u - 1).clamp(-1, 1)
        terms = [torch.ones_like(z), z]
        for _ in range(2, self.hammerstein_coeffs.shape[1] + 2):
            terms.append(2 * z * terms[-1] - terms[-2])
        basis = torch.stack(terms[2:], dim=-1)  # (..., pump, k)
        curved = u + (basis * self.hammerstein_coeffs).sum(-1)
        return seq.index_copy(-1, self.pump_idx, curved)

    def regime_weights(self, x: torch.Tensor) -> torch.Tensor:
        """`(batch, n_regimes)` gate over the response banks.

        Reads the pump channels over the whole sequence (with the known
        future, that is the pump schedule of the forecast horizon).
        """
        if self.gate is None:
            return x.new_ones(x.shape[0], 1)
        pumps = self._sequence(x)[:, :, self.gate_idx].mean(dim=1)
        return torch.softmax(self.gate(pumps), dim=-1)

    def _chebyshev_residual(self, target_past: torch.Tensor) -> torch.Tensor:
        """`(b, H, i)` explicit polynomial correction (see `residual_kind`)."""
        lags = self.residual_coeffs.shape[2]
        z = target_past[:, -lags:].flip(1).clamp(-1, 1)  # (b, l, i), l=1..L
        terms = [z]
        if self.residual_coeffs.shape[3] > 1:
            terms.append(2 * z * z - 1)
        for _ in range(3, self.residual_coeffs.shape[3] + 1):
            terms.append(2 * z * terms[-1] - terms[-2])
        basis = torch.stack(terms, dim=-1)  # (b, l, i, k)
        return torch.einsum("blik,ihlk->bhi", basis, self.residual_coeffs)

    def frequency_response(self) -> torch.Tensor:
        """`(regime, target, sender, n_bins)` complex `H_ij(f)`."""
        bands = torch.view_as_complex(self.response.contiguous())
        return bands @ self.interp.T.to(bands.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: PLR0914
        """Recursive ARX simulation over the horizon, in the frequency domain.

        A single shift-invariant filter cannot forecast several steps
        ahead from the series' own past (step `h` would need its own
        lag), so the filter is the **one-step** model and is iterated:
        at step `k` the graph of filters is evaluated at position
        `past_len + k`, and the predicted outputs are written back
        into the sequence before the next step. The self-edges `H_ii`
        are then the autoregressive polynomials (the poles), the
        edges from the inputs the exogenous ones — a transfer-function
        matrix proper. Known future inputs enter step by step.
        """
        seq = self._sequence(x)
        past = seq[:, : self.past_len]
        # Centre only, never divide by the window's std: the pumps sit
        # constant (off) for whole windows, std ~ 0, and a pump switching
        # on in the known horizon would blow up to ~1e5. The
        # preprocessor already puts every channel on a [0, 1]-ish scale.
        mu = past.mean(dim=1, keepdim=True)
        norm = seq - mu
        # Future outputs (and, without `future_in_input`, everything in
        # the horizon) are unknown: zero after centring, then filled
        # step by step with the model's own predictions.
        horizon_mask = torch.ones_like(norm)
        horizon_mask[:, self.past_len :] = (
            self.future_mask if self.future_in_input else 0.0
        )
        norm *= horizon_mask

        response = self.frequency_response()  # (k, i, j, f)
        gate = self.regime_weights(x).to(response.dtype)  # (b, k)
        mixed = torch.einsum("bk,kijf->bijf", gate, response)
        if self.regime_per_step:
            # Gate read at each horizon position (known-future setpoint).
            step_gate = torch.softmax(self.gate(seq[..., self.gate_idx]), -1)
        n_targets = len(self.target_idx)
        scatter = torch.zeros(n_targets, self.n_nodes, device=x.device)
        scatter[torch.arange(n_targets), self.target_idx] = 1.0
        keep = 1.0 - scatter.sum(dim=0)  # 0 on target channels

        steps = []
        for k in range(self.output_window):
            pos = self.past_len + k
            spec = torch.fft.rfft(norm, n=self.n_fft, dim=1)  # (b, f, j)
            if self.regime_per_step:
                out_spec = torch.einsum("kijf,bfj->bkfi", response, spec)
                full = torch.fft.irfft(out_spec, n=self.n_fft, dim=2)
                g = step_gate[:, pos].to(full.dtype)  # (b, k)
                step = torch.einsum("bk,bki->bi", g, full[:, :, pos])
                step += self.bias
            else:
                out_spec = torch.einsum("bijf,bfj->bfi", mixed, spec)
                full = torch.fft.irfft(out_spec, n=self.n_fft, dim=1)
                step = full[:, pos] + self.bias  # (b, i)
            steps.append(step)
            row = norm[:, pos] * keep + step @ scatter
            norm = torch.cat(
                [norm[:, :pos], row[:, None], norm[:, pos + 1 :]], dim=1
            )
        linear = torch.stack(steps, dim=1)  # (b, H, i)

        self._last_linear = linear
        self._last_residual = None
        self._last_mixed = mixed
        self._last_mu = mu
        self._last_past = past
        out = linear
        if self.residual_coeffs is not None:
            target_past = norm[:, : self.past_len, self.target_idx]
            self._last_residual = self._chebyshev_residual(target_past)
            out = linear + self._last_residual
        elif self.residual is not None:
            target_past = norm[:, : self.past_len, self.target_idx]
            residual = self.residual(target_past.transpose(1, 2))
            self._last_residual = residual.transpose(1, 2)
            # Not `+=`: that would add the residual into `_last_linear`
            # (same tensor) and inflate `nonlinearity_share`'s linear part.
            out = linear + self._last_residual

        return out + mu[:, :, self.target_idx]

    def backcast(self, x: torch.Tensor) -> torch.Tensor:
        """Same `H_ij(f)` applied to reconstruct the *observed* past.

        Auto-consistency check for an ARX filter: `H_ii`/`H_ij` is one
        shift-invariant kernel used at every position, so it must also
        explain `y_i`'s own past from the other series' past — not
        just the (harder, error-compounding) horizon it is iterated
        over in `forward`. Unlike the horizon, every past position is
        already observed, so this needs no step-by-step write-back:
        one filter pass over the whole past window.

        Call after `forward()` (reuses its cached `mixed`/`mu`/`past`).

        Returns:
            `(batch, past_len, len(target_idx))`, same (centred) units
            as `forward`'s output.
        """
        if self._last_mixed is None:
            raise RuntimeError("call forward() before backcast()")
        past = self._last_past
        norm = past - self._last_mu
        spec = torch.fft.rfft(norm, n=self.n_fft, dim=1)
        out_spec = torch.einsum("bijf,bfj->bfi", self._last_mixed, spec)
        full = torch.fft.irfft(out_spec, n=self.n_fft, dim=1)
        return full[:, : self.past_len] + self.bias

    def backcast_loss(self, x: torch.Tensor) -> torch.Tensor:
        """MSE of `backcast()` against the observed past targets.

        Without this term `H_ij` is only ever evaluated causally
        forward of the window it was fit on; this adds a cheap,
        non-recursive gradient signal that keeps the filter consistent
        with the past it was given, instead of only the horizon it
        has to extrapolate into.
        """
        target_past = self._sequence(x)[:, : self.past_len, self.target_idx]
        return nn.functional.mse_loss(self.backcast(x), target_past)

    @torch.no_grad()
    def impulse_response(
        self, i: int, j: int, regime: int = 0
    ) -> torch.Tensor:
        """`(n_fft,)` kernel of edge `j -> target_idx[i]` (route A).

        Index `tau` is the weight of `s_j[t - tau]` in `y_i[t]`; indices
        past `n_fft // 2` wrap to negative lags (the known future
        inputs inside the sequence). In the model's normalised units.
        """
        h = self.frequency_response()[regime, i, j]
        return torch.fft.irfft(h, n=self.n_fft)

    @torch.no_grad()
    def band_strength(self) -> torch.Tensor:
        """`(n_targets, n_nodes, n_bands)` `|H_ij|` per band (route B).

        Averaged over regimes; per-regime graphs are `self.response`.
        """
        return self.response.norm(dim=-1).mean(dim=0)

    def group_l1(self, alpha: float = 0.5) -> torch.Tensor:
        """Sparse-group lasso over (edge, band).

        `alpha` weighs single bands (an edge active only in some
        bands) against whole edges (`||H_ij||` over all bands and
        regimes). Pass through `TrainConfig.message_l1_weight`.
        """
        band_mag = self.response.norm(dim=-1)  # (k, i, j, m)
        per_band = band_mag.sum()
        per_edge = band_mag.pow(2).sum(dim=(0, 3)).sqrt().sum()
        return alpha * per_band + (1 - alpha) * per_edge

    def nonlinearity_share(self) -> float:
        """Share of the last output energy from the nonlinear residual."""
        if self._last_residual is None:
            return 0.0
        lin = self._last_linear.pow(2).sum()
        res = self._last_residual.pow(2).sum()
        return (res / (lin + res + 1e-12)).item()
