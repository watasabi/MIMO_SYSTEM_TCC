"""Per-output correction heads, to decouple shared-backbone capacity.

`MPNNForecaster`/`SymbolicGraphNetwork` share `update_fn` across every
node, including the 3 outputs; `StemGNN`/`Seq2SeqLatentGNN`/
`FourierGNN` share their final projection across outputs too. When one
output is structurally harder than the rest (`nivel_res` integrates
the inflow/outflow imbalance instead of being near-algebraic in the
inputs — `docs/1_analise-dados-reservatorio.md` §4-5), training it
well can pull the shared backbone away from what the easy outputs
need: wrapping the architecture in `LeakyIntegratorWrapper` alone
dropped `pressao`'s test R² from 0.64 to as low as -1.8 in early
experiments on this branch, even though the integrator only touches
`nivel_res`'s channel.

`PerOutputHeadWrapper` does not touch the backbone: it adds a small
per-output MLP that refines the backbone's raw forecast, each with its
own parameters, so an output that needs more non-linear massaging to
fit a delta/integrated target does not have to share that capacity
(or fight for it) with the easy outputs.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["PerOutputHeadWrapper"]


class PerOutputHeadWrapper(nn.Module):
    """Add an independent residual correction MLP per output channel.

    Args:
        model: Forecaster with the shared contract
            `(batch, input_window, n_nodes) -> (batch, horizon,
            n_targets)`.
        n_targets: Number of outputs (`len(target_idx)`).
        hidden: Hidden width of each per-output head. The head takes
            the backbone's raw scalar forecast at each horizon step
            (not a hidden representation — the architectures don't
            share one in a common shape) and outputs a residual
            correction, so a head starting at zero reduces to the
            backbone's own forecast.
    """

    def __init__(
        self, model: nn.Module, n_targets: int, hidden: int = 16
    ) -> None:
        super().__init__()
        self.model = model
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(n_targets)
            ]
        )
        # Correction starts at (near) zero: early training reproduces
        # the backbone's own forecast exactly, then each head learns
        # to deviate where its output needs to.
        for head in self.heads:
            last = head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Backbone forecast plus a per-output residual correction."""
        out = self.model(x, **kwargs)
        corrected = [
            out[..., i : i + 1] + head(out[..., i : i + 1])
            for i, head in enumerate(self.heads)
        ]
        return torch.cat(corrected, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapped model's `encode`, unchanged by the output heads."""
        return self.model.encode(x)

    def encoder_modules(self) -> list[nn.Module]:
        """Wrapped model's encoder modules."""
        return self.model.encoder_modules()
