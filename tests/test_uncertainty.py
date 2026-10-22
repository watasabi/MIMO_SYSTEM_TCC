"""Uncertainty tools: conformal coverage, Laplace posterior, delta method."""

import numpy as np
import pytest
import torch

from mimo_sys.architectures import SpectralTFNet
from mimo_sys.training.uncertainty import (
    coefficient_ci,
    conformal_quantile,
    fit_log_sigma,
    impulse_response_graph,
    interval_metrics,
    laplace_linear_posterior,
)

RNG = np.random.default_rng(0)
NOMINAL_SLACK = 0.03


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_conformal_cobre_o_nominal_em_dados_iid(alpha: float) -> None:
    calib = np.abs(RNG.standard_t(4, 2000))
    test = RNG.standard_t(4, 20000)
    q = conformal_quantile(calib, alpha)
    m = interval_metrics(
        test, -q * np.ones_like(test), q * np.ones_like(test), alpha
    )
    assert abs(m["cobertura"] - (1 - alpha)) < NOMINAL_SLACK


def test_log_sigma_recupera_heterocedasticidade() -> None:
    x = RNG.uniform(0, 1, 5000)
    features = np.stack([np.ones_like(x), x], axis=1)
    errors = RNG.normal(0, np.exp(0.5 + 1.5 * x))
    beta = fit_log_sigma(features, errors)
    np.testing.assert_allclose(beta, [0.5, 1.5], atol=0.1)


def _small_model() -> SpectralTFNet:
    torch.manual_seed(0)
    return SpectralTFNet(
        n_nodes=3,
        input_window=30,
        output_window=4,
        target_idx=[2],
        n_bands=3,
        known_future_idx=(0, 1),
    ).eval()


def _data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 30, 3)
    y = torch.rand(n, 4, 1)
    return x, y


def test_laplace_e_psd_e_encolhe_com_mais_dados() -> None:
    model = _small_model()
    _, few = laplace_linear_posterior(model, *_data(8))
    _, many = laplace_linear_posterior(model, *_data(64))
    assert np.allclose(few, few.T)
    assert np.linalg.eigvalsh(few).min() > 0
    assert np.trace(many) < np.trace(few)


def test_delta_method_usa_o_gradiente_certo() -> None:
    """dc/dtheta confere com diferenças finitas num coeficiente da equação."""
    model = _small_model()
    _, sigma = laplace_linear_posterior(model, *_data(16))

    def coef(m: torch.nn.Module) -> torch.Tensor:
        return impulse_response_graph(m, 0, 0)[0:5].sum()

    _, sd = coefficient_ci(model, sigma, coef)
    eps = 1e-3
    grads = []
    with torch.no_grad():
        for name in ("response", "bias"):
            p = getattr(model, name)
            flat = p.view(-1)
            for k in range(flat.numel()):
                flat[k] += eps
                up = float(coef(model))
                flat[k] -= 2 * eps
                down = float(coef(model))
                flat[k] += eps
                grads.append((up - down) / (2 * eps))
    g = np.array(grads)
    expected = np.sqrt(g @ sigma @ g)
    assert sd == pytest.approx(expected, rel=1e-3)
