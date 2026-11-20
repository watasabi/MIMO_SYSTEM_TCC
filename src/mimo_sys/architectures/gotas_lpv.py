"""GOTAS-LPV: GOTAS-W with operating-point-dependent filters.

Plan: `docs/18_plano-gotas-w.md` (exploração depois da Etapa 1).

Two extensions of `GOTASWavelet`, both still read off as equations:

- **Linear parameter-varying (LPV) edges.** Each filter tap is an affine
  function of scheduling signals `p_s` read at the forecast step (by
  default the three pump frequencies, known over the horizon):

      a_ijml(t) = a_ijml + sum_s b_ijmls * p_s(t)

  so the transfer function of an edge changes with the pumps' operating
  point instead of being fixed over the whole series (the Fourier and
  plain ARX assumption). `lpv_strength` gives the size of each
  modulation.
- **Hammerstein (KAN-like) input curves.** Each known input `u` goes
  through `u + sum_k h_k T_k(2u - 1)` (Chebyshev, `k = 2..degree`) before
  the decomposition: one learned univariate curve per input, a closed
  polynomial, as the static pump curve. Starts linear (`h = 0`).

Both are off by default, and then the model is exactly `GOTASWavelet`.
"""

from __future__ import annotations

import torch
from torch import nn

from mimo_sys.architectures.gotas_wavelet import GOTASWavelet, atrous

__all__ = ["GOTASLPV"]

DEFAULT_SCHEDULING = (0, 1, 2)  # freq_b1..3
GROUP_EPS = 1e-12


class GOTASLPV(GOTASWavelet):
    """`GOTASWavelet` plus LPV edges and Hammerstein input curves.

    Args:
        lpv: Make every tap affine in the scheduling signals.
        scheduling_idx: Nodes read as scheduling signals, at the step
            being predicted (known inputs only).
        hammerstein_degree: Chebyshev degree of the input curves; 0 or 1
            disables them.
        **kwargs: `GOTASWavelet` arguments.
    """

    def __init__(
        self,
        *args: object,
        lpv: bool = False,
        scheduling_idx: tuple[int, ...] = DEFAULT_SCHEDULING,
        hammerstein_degree: int = 0,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        known = torch.nonzero(self.future_mask).flatten().tolist()
        if lpv and not set(scheduling_idx) <= set(known):
            raise ValueError("scheduling signals must be known inputs")
        self.register_buffer(
            "scheduling_idx", torch.tensor(scheduling_idx, dtype=torch.long)
        )
        self.register_buffer(
            "input_idx", torch.tensor(known, dtype=torch.long)
        )
        self.lpv_weight = None
        if lpv:
            # (target, sender, component, lag, scheduling signal)
            self.lpv_weight = nn.Parameter(
                torch.zeros(*self.weight.shape, len(scheduling_idx))
            )
        self.hammerstein = None
        if hammerstein_degree > 1:
            # (known input, k = 2..degree)
            self.hammerstein = nn.Parameter(
                torch.zeros(len(known), hammerstein_degree - 1)
            )
        self._schedule: torch.Tensor | None = None

    def _curves(self, seq: torch.Tensor) -> torch.Tensor:
        """Known inputs through their Chebyshev curves (identity at 0)."""
        if self.hammerstein is None:
            return seq
        u = seq[..., self.input_idx]
        z = (2 * u - 1).clamp(-1, 1)
        terms = [torch.ones_like(z), z]
        for _ in range(2, self.hammerstein.shape[1] + 2):
            terms.append(2 * z * terms[-1] - terms[-2])
        basis = torch.stack(terms[2:], dim=-1)  # (..., input, k)
        curved = u + (basis * self.hammerstein).sum(-1)
        return seq.index_copy(-1, self.input_idx, curved)

    def _prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.future_in_input:
            pad = x.new_zeros(x.shape[0], self.output_window, x.shape[2])
            x = torch.cat([x, pad], dim=1)
        # The schedule is read raw (preprocessor scale), before the curve.
        self._schedule = x[..., self.scheduling_idx]
        seq = self._curves(x)
        future_in_input = self.future_in_input
        self.future_in_input = True  # `x` already holds the horizon
        try:
            return super()._prepare(seq)
        finally:
            self.future_in_input = future_in_input

    def _step(self, norm: torch.Tensor, pos: int) -> torch.Tensor:
        if self.lpv_weight is None:
            return super()._step(norm, pos)
        comps = atrous(norm[:, : pos + 1], self.levels)
        index = (pos - self.offsets).clamp_min(0)
        b, _, n, m = comps.shape
        lagged = torch.gather(
            comps.permute(0, 2, 3, 1),
            3,
            index[None, None].expand(b, n, m, -1),
        )  # (b, n, m, l)
        schedule = self._schedule[:, pos]  # (b, s)
        taps = self.weight + torch.einsum(
            "inmls,bs->binml", self.lpv_weight, schedule
        )
        return torch.einsum("bnml,binml->bi", lagged, taps) + self.bias

    def group_l1(self, alpha: float = 0.5) -> torch.Tensor:
        """Base sparse-group lasso plus one group per modulated filter."""
        penalty = super().group_l1(alpha)
        if self.lpv_weight is None:
            return penalty
        w = self.lpv_weight
        # `+ GROUP_EPS`: as modulações começam em zero, e a derivada de
        # `sqrt` em zero é infinita (NaN no primeiro passo).
        groups = (w.pow(2).sum(dim=(-2, -1)) + GROUP_EPS).sqrt().sum()
        return penalty + alpha * w.abs().sum() + (1 - alpha) * groups

    @torch.no_grad()
    def lpv_strength(self) -> torch.Tensor:
        """`(n_targets, n_nodes, n_components, n_schedule)` modulation size."""
        if self.lpv_weight is None:
            raise RuntimeError("model built without lpv")
        return self.lpv_weight.norm(dim=3)
