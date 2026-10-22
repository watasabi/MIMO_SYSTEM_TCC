"""Uncertainty for the interpretable model: on the equation and the forecast.

Two independent pieces (`docs/12_incerteza-e-explicabilidade.md`):

- **Laplace posterior of the linear transfer functions** of a
  `SpectralTFNet` (generalised Gauss-Newton on the 12-step recursive
  simulation, the same output the model is trained on), giving a
  credible interval for any coefficient read off the equation (an edge's
  static gain, the mass-balance taps);
- **forecast intervals**: an explicit heteroscedastic `log sigma` (a
  linear function of horizon step, hour and pump moves, so it says *when*
  the forecast is less reliable) recalibrated by split conformal per
  step and output, which fixes the coverage without a distribution
  assumption.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.func import functional_call, jacrev

LINEAR_PARAMS = ("response", "bias")


def laplace_linear_posterior(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    prior_precision: float = 1.0,
    batch: int = 8,
    chunk: int = 64,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Gaussian posterior `N(theta_hat, Sigma)` of the linear parameters.

    `Sigma^-1 = sum_n J_n^T W J_n + lambda I`, with `J_n` the Jacobian of
    window `n`'s 12-step forecast w.r.t. `response` and `bias`, `W` the
    inverse residual variance of each output (estimated on `x`, `y`)
    and a weak isotropic prior. The model must have no nonlinear
    residual (the interpretable, linear-only model).

    Args:
        model: Trained `SpectralTFNet`, eval mode, residual disabled.
        x: `(n, seq_len, n_nodes)` training windows.
        y: `(n, horizon, n_targets)` their targets.
        prior_precision: `lambda`.
        batch: Windows per Jacobian evaluation.
        chunk: Output rows differentiated at once (memory bound).

    Returns:
        `(theta_hat, Sigma)`, flattened in `LINEAR_PARAMS` order.
    """
    params = {k: getattr(model, k).detach() for k in LINEAR_PARAMS}
    frozen = {
        k: v.detach()
        for k, v in {
            **dict(model.named_parameters()),
            **dict(model.named_buffers()),
        }.items()
        if k not in params
    }
    with torch.no_grad():
        noise = ((model(x) - y) ** 2).mean(dim=(0, 1))  # per output
    weight = 1.0 / noise

    def forecast(p: dict[str, torch.Tensor], xb: torch.Tensor) -> torch.Tensor:
        return functional_call(model, {**frozen, **p}, (xb,))

    n_par = sum(v.numel() for v in params.values())
    precision = prior_precision * torch.eye(n_par, dtype=torch.float64)
    for start in range(0, len(x), batch):
        xb = x[start : start + batch]
        # `chunk`: vjps vectorised at once; the recursive 12-step graph
        # is large, so materialising all of them runs out of memory.
        jac = jacrev(forecast, chunk_size=chunk)(params, xb)
        j = torch.cat(
            [jac[k].reshape(*jac[k].shape[:3], -1) for k in LINEAR_PARAMS],
            dim=-1,
        ).double()  # (b, H, i, P)
        w = weight.double()[None, None, :, None]
        precision += torch.einsum("bhip,bhiq->pq", j * w, j)
    theta = torch.cat([params[k].reshape(-1) for k in LINEAR_PARAMS])
    sigma = torch.linalg.inv(precision)
    return theta.double().numpy(), sigma.numpy()


def coefficient_ci(
    model: nn.Module,
    sigma: npt.NDArray,
    coefficient: Callable[[nn.Module], torch.Tensor],
) -> tuple[float, float]:
    """Mean and standard deviation of a scalar read off the equation.

    Delta method: `Var(c) = g^T Sigma g`, `g = dc / dtheta`. Exact for
    coefficients that are linear in the transfer functions (impulse
    response taps and their sums).

    Args:
        model: The model `sigma` was computed for.
        sigma: From `laplace_linear_posterior`.
        coefficient: `model -> scalar tensor`, differentiable w.r.t.
            `response` and `bias`.
    """
    params = [getattr(model, k) for k in LINEAR_PARAMS]
    for p in params:
        p.requires_grad_(True)
    value = coefficient(model)
    grads = torch.autograd.grad(value, params, allow_unused=True)
    g = torch.cat(
        [
            (gr if gr is not None else torch.zeros_like(p)).reshape(-1)
            for gr, p in zip(grads, params, strict=True)
        ]
    ).double()
    var = float(g @ torch.as_tensor(sigma) @ g)
    return float(value.detach()), float(np.sqrt(max(var, 0.0)))


def impulse_response_graph(
    model: nn.Module, i: int, j: int, regime: int = 0
) -> torch.Tensor:
    """`SpectralTFNet.impulse_response` without `no_grad` (for gradients)."""
    bands = torch.view_as_complex(model.response.contiguous())
    h = bands[regime, i, j] @ model.interp.T.to(bands.dtype)
    return torch.fft.irfft(h, n=model.n_fft)


def fit_log_sigma(
    features: npt.NDArray, errors: npt.NDArray, ridge: float = 1e-3
) -> npt.NDArray:
    """Explicit `log sigma = features @ beta` from forecast errors.

    Least squares of `log |e|` (plus the constant that makes it unbiased
    for a Gaussian, `E log|z| = -0.635`) on the features; `beta` is the
    readable equation of the uncertainty.

    Args:
        features: `(n, p)`, a constant column included.
        errors: `(n,)`.
    """
    target = np.log(np.abs(errors) + 1e-9) + 0.635
    a = features.T @ features + ridge * np.eye(features.shape[1])
    return np.linalg.solve(a, features.T @ target)


def conformal_quantile(scores: npt.NDArray, alpha: float) -> float:
    """Split-conformal quantile with the `(n + 1)` finite-sample correction."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def interval_metrics(
    y: npt.NDArray, lower: npt.NDArray, upper: npt.NDArray, alpha: float
) -> dict[str, float]:
    """Coverage, mean width and the interval (Winkler) score."""
    width = upper - lower
    below = np.clip(lower - y, 0, None)
    above = np.clip(y - upper, 0, None)
    score = width + (2 / alpha) * (below + above)
    return {
        "cobertura": float(np.mean((y >= lower) & (y <= upper))),
        "largura": float(width.mean()),
        "winkler": float(score.mean()),
    }
