"""Which lags matter: gradient saliency and native attention, aggregated.

Before fixing `input_window` for the symbolic-regression route (which
needs it small — `docs/4_rotas-de-extracao.md`), train an attention
model with a large `input_window` and read off which lags it actually
uses. Two independent signals:

- `gradient_saliency`: `|∂forecast/∂x[lag]|`, model-agnostic — works on
  any architecture, including ones with no attention mechanism.
- `attention_by_lag`: the native decoder→encoder attention of
  `Seq2SeqLatentGNN` (`StemGNN`'s node attention has no lag axis, so it
  is not covered — see its README for why).

Both aggregate over every test window, not a single sample, so the
result is not an artifact of one forecast.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


def gradient_saliency(
    model: nn.Module,
    windows: torch.Tensor,
    target_idx: list[int] | None = None,
    batch_size: int = 32,
) -> npt.NDArray:
    """Mean `|∂forecast/∂x|` per lag and input node, over `windows`.

    Args:
        model: Forecaster `(B, input_window, n_nodes) -> (B, H,
            n_targets)`, any architecture.
        windows: `(n, input_window, n_nodes)` test inputs.
        target_idx: Indices *into the model's output* (0-based, not
            node indices) to sum gradients over; defaults to all of
            the model's outputs.
        batch_size: Batched to keep the backward pass's memory bounded.

    Returns:
        `(input_window, n_nodes)`: higher means that lag/node moves the
        forecast more, averaged over `windows` and, within a window,
        over horizon and output steps.
    """
    model.eval()
    total = None
    n_seen = 0
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size].clone()
        batch.requires_grad_(True)
        out = model(batch)
        if target_idx is not None:
            out = out[..., target_idx]
        out.sum().backward()
        grad = batch.grad.abs().sum(dim=0).detach()
        total = grad if total is None else total + grad
        n_seen += len(batch)
    return (total / n_seen).cpu().numpy()


def attention_by_lag(model: nn.Module, windows: torch.Tensor) -> npt.NDArray:
    """Mean decoder→encoder attention per input lag, `Seq2SeqLatentGNN`.

    Args:
        model: A `Seq2SeqLatentGNN` (or `RevINWrapper` around one).
        windows: `(n, input_window, n_nodes)` test inputs.

    Returns:
        `(input_window,)`: attention mass each lag receives, averaged
        over every decoder step and every window.
    """
    core = model.model if hasattr(model, "model") else model
    model.eval()
    encoded_store: list[torch.Tensor] = []
    decoded_store: list[torch.Tensor] = []
    h_enc = core.encoder.register_forward_hook(
        lambda _m, _i, o: encoded_store.append(o[0].detach())
    )
    h_dec = core.decoder.register_forward_hook(
        lambda _m, _i, o: decoded_store.append(o[0].detach())
    )
    try:
        with torch.no_grad():
            model(windows)
    finally:
        h_enc.remove()
        h_dec.remove()
    encoded = encoded_store[-1]  # (n, input_window, hidden)
    weights = torch.cat(
        [
            torch.softmax(torch.bmm(d, encoded.transpose(1, 2)), dim=-1)
            for d in decoded_store
        ],
        dim=1,
    )  # (n, horizon, input_window)
    return weights.mean(dim=(0, 1)).cpu().numpy()


def top_lags(
    importance: npt.NDArray, k: int, input_window: int
) -> npt.NDArray:
    """Indices of the `k` most important lags, as distance from the end.

    Args:
        importance: `(input_window,)` or `(input_window, n_nodes)`
            (summed over nodes first).
        k: How many lags to keep.
        input_window: Length of `importance`'s first axis.

    Returns:
        Sorted ascending `(k,)` array of lags-before-the-last-step
        (0 = most recent), e.g. for picking a small, non-contiguous
        `input_window` for the distillation route.
    """
    per_lag = importance if importance.ndim == 1 else importance.sum(axis=1)
    most_recent_first = per_lag[::-1]
    chosen = np.argsort(most_recent_first)[-k:]
    return np.sort(chosen)
