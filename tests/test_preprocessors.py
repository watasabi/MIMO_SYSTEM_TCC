"""Testes do TimeSeriesPreprocessor."""

import numpy as np
import pytest

from mimo_sys.preprocessors import TimeSeriesPreprocessor

WINDOW = 24
HORIZON = 12
FEATS = [1, 2]


@pytest.fixture
def series() -> np.ndarray:
    """Passeio aleatório positivo com 3 colunas."""
    rng = np.random.default_rng(0)
    return np.cumsum(rng.normal(size=(300, 3)), axis=0) + 50.0


def _pp(**overrides: object) -> TimeSeriesPreprocessor:
    kwargs = {
        "window_size": WINDOW,
        "horizon": HORIZON,
        "remove_outliers": False,
        "apply_filter": False,
        "apply_ewt": False,
        "detrend": False,
        "plot": False,
    }
    kwargs.update(overrides)
    return TimeSeriesPreprocessor(**kwargs)


def _raw_targets(
    data: np.ndarray, pp: TimeSeriesPreprocessor, n: int
) -> np.ndarray:
    return np.stack(
        [data[a : a + HORIZON][:, FEATS] for a in pp.anchor_indices(n)]
    )


def test_transform_com_ewt_reproduz_fit_transform(
    series: np.ndarray,
) -> None:
    """transform com EWT usa as fronteiras ajustadas no treino."""
    pp = _pp(normalize=False, apply_ewt=True, detrend=True)
    x_fit, _, _ = pp.fit_transform(series)
    x_transform, _ = pp.transform(series)
    np.testing.assert_allclose(x_transform, x_fit, atol=1e-8)


def test_transform_com_filtro_reproduz_fit_transform(
    series: np.ndarray,
) -> None:
    """transform aplica filtro e normalização na mesma ordem do fit."""
    pp = _pp(apply_filter=True, scaler="minmax")
    x_fit, _, _ = pp.fit_transform(series)
    x_transform, _ = pp.transform(series)
    np.testing.assert_allclose(x_transform, x_fit, atol=1e-8)


@pytest.mark.parametrize("scaler", ["standard", "minmax", "robust"])
@pytest.mark.parametrize(
    "steps",
    [
        {},
        {"differencing": True},
        {"apply_ewt": True, "detrend": True},
        {"log_transform": True, "differencing": True},
    ],
)
@pytest.mark.parametrize("step_size", [1, 5])
def test_round_trip_reconstroi_segmento_novo(
    series: np.ndarray, scaler: str, steps: dict, step_size: int
) -> None:
    """Janela -> inverse_transform(reference=seg) devolve o sinal bruto."""
    train, test = series[:200], series[200:]
    pp = _pp(scaler=scaler, step_size=step_size, **steps)
    pp.fit_transform(train, feats=FEATS)
    _, y_test = pp.transform(test)
    y_test = y_test[:, :, FEATS]

    anchors = pp.anchor_indices(len(y_test))
    rebuilt = pp.inverse_transform(
        y_test, anchors, feats=FEATS, reference=test
    )
    np.testing.assert_allclose(
        rebuilt, _raw_targets(test, pp, len(y_test)), atol=1e-6
    )


def test_step_size_reduz_janelas(series: np.ndarray) -> None:
    """step_size=k mantém uma a cada k janelas."""
    x1, _, _ = _pp(step_size=1).fit_transform(series)
    x5, _, _ = _pp(step_size=5).fit_transform(series)
    np.testing.assert_allclose(x5, x1[::5])
