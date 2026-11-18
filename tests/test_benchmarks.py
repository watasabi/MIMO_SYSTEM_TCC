"""Tests of the temporal cross-validation pieces and the new baselines."""

import numpy as np
import pytest
import torch

from mimo_sys.architectures import (
    GRUForecaster,
    LSTMForecaster,
    MLPForecaster,
)
from mimo_sys.training.benchmarks import (
    TREE_LAGS,
    TreeForecaster,
    armax_forecast,
    fit_armax,
    forecast_metrics,
    seasonal_scale,
    temporal_folds,
    tree_features,
    valid_origins,
)
from mimo_sys.training.data import N_NODES, N_TARGETS, TARGET_IDX

N_ROWS = 2208
PAST = 8
HORIZON = 4
WINDOW = PAST + HORIZON
BATCH = 5
KNOWN = [0, 1, 2, 3]
ARMAX_TOLERANCE = 0.05


def test_folds_are_chronological_and_tests_do_not_overlap() -> None:
    folds = temporal_folds(N_ROWS)
    assert folds[-1].test_end == N_ROWS
    for fold in folds:
        assert 0 < fold.train_end < fold.val_end < fold.test_end
    for before, after in zip(folds, folds[1:], strict=False):
        assert before.test_end == after.test_end - 220
        assert before.val_end <= after.val_end - 220
    with pytest.raises(ValueError, match="do not fit"):
        temporal_folds(N_ROWS - 1)


def test_valid_origins_drop_every_horizon_touching_a_failure() -> None:
    bad = np.zeros(30, dtype=bool)
    bad[10] = True
    keep = valid_origins(bad, start=5, n_origins=10, horizon=3)
    rows = 5 + np.arange(10)
    assert rows[~keep].tolist() == [8, 9, 10]


def test_forecast_metrics_match_hand_values() -> None:
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    pred = y + 1.0
    metrics = forecast_metrics(y, pred, scale=np.array([2.0]))
    assert metrics["mae"][0] == pytest.approx(1.0)
    assert metrics["rmse"][0] == pytest.approx(1.0)
    assert metrics["mase"][0] == pytest.approx(0.5)
    assert metrics["r2"][0] == pytest.approx(1 - 4 / 5)
    # Same correlation and spread: only the mean ratio (3.5 / 2.5) counts.
    assert metrics["kge"][0] == pytest.approx(1 - 0.4)
    perfect = forecast_metrics(y, y, scale=np.array([2.0]))
    assert perfect["kge"][0] == pytest.approx(1.0)
    assert perfect["smape"][0] == pytest.approx(0.0)


def test_seasonal_scale_uses_the_previous_season() -> None:
    y = np.arange(30.0)[:, None]
    assert seasonal_scale(y, season=24)[0] == pytest.approx(24.0)


@pytest.mark.parametrize("cls", [MLPForecaster, LSTMForecaster, GRUForecaster])
def test_baselines_shape_and_persistence_start(cls: type) -> None:
    model = cls(N_NODES, WINDOW, HORIZON, TARGET_IDX)
    x = torch.randn(BATCH, WINDOW, N_NODES)
    assert model(x).shape == (BATCH, HORIZON, N_TARGETS)
    # With every weight at zero, only the last observed value is left.
    for parameter in model.parameters():
        parameter.data.zero_()
    last = x[:, PAST - 1, TARGET_IDX].unsqueeze(1)
    assert torch.allclose(model(x), last.expand(-1, HORIZON, -1))


@pytest.mark.parametrize("cls", [MLPForecaster, LSTMForecaster, GRUForecaster])
def test_baselines_use_the_known_future_inputs(cls: type) -> None:
    model = cls(N_NODES, WINDOW, HORIZON, TARGET_IDX).eval()
    x = torch.randn(BATCH, WINDOW, N_NODES)
    changed = x.clone()
    changed[:, PAST:, 0] += 1.0
    assert not torch.allclose(model(changed), model(x))


def test_tree_forecaster_adds_the_last_observed_value() -> None:
    x = np.random.default_rng(0).normal(size=(BATCH, WINDOW, N_NODES))
    features = tree_features(x, PAST, KNOWN)
    n_lags = sum(lag <= PAST for lag in TREE_LAGS)
    assert features.shape == (BATCH, n_lags * N_NODES + HORIZON * len(KNOWN))
    model = TreeForecaster(
        lambda f: np.zeros((len(f), HORIZON * N_TARGETS)),
        PAST,
        HORIZON,
        KNOWN,
    )
    out = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
    last = x[:, PAST - 1, TARGET_IDX]
    assert np.allclose(out, np.repeat(last[:, None], HORIZON, axis=1))


def simulated_arx(n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Three stable first-order outputs driven by four inputs."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(n, 4))
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + 0.5 * u[t - 1, :3]
    return y + 10.0, u + 5.0


def test_armax_recovers_a_noiseless_system() -> None:
    y, u = simulated_arx()
    model = fit_armax(y[:400], u[:400], (1, 1, 1))
    origins = np.arange(420, 500, 10)
    pred = armax_forecast(model, y, u, origins, HORIZON)
    true = np.stack([y[o : o + HORIZON] for o in origins])
    assert np.abs(pred - true).max() < ARMAX_TOLERANCE


def test_armax_forecast_never_reads_future_outputs() -> None:
    y, u = simulated_arx()
    model = fit_armax(y[:400], u[:400], (2, 2, 2))
    origins = np.array([450])
    reference = armax_forecast(model, y, u, origins, HORIZON)
    changed = y.copy()
    changed[450:] += 100.0
    assert np.array_equal(
        armax_forecast(model, changed, u, origins, HORIZON), reference
    )


def test_armax_pumps_scenario_hides_the_future_inflow() -> None:
    y, u = simulated_arx()
    model = fit_armax(y[:400], u[:400], (2, 2, 1))
    origins = np.array([450])
    reference = armax_forecast(
        model, y, u, origins, HORIZON, inflow_known=False
    )
    changed = u.copy()
    changed[450:, 3] += 50.0
    assert np.array_equal(
        armax_forecast(
            model, y, changed, origins, HORIZON, inflow_known=False
        ),
        reference,
    )
