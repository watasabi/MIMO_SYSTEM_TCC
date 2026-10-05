"""Testes do TimeSeriesPreprocessor."""

import matplotlib
import numpy as np

from mimo_sys.preprocessors import TimeSeriesPreprocessor

matplotlib.use("Agg")


def test_transform_com_ewt_reproduz_fit_transform() -> None:
    """transform com EWT usa as fronteiras ajustadas no treino."""
    rng = np.random.default_rng(0)
    data = np.cumsum(rng.normal(size=(300, 3)), axis=0)
    pp = TimeSeriesPreprocessor(
        window_size=24,
        horizon=12,
        normalize=False,
        remove_outliers=False,
        apply_filter=False,
    )
    x_fit, _, _ = pp.fit_transform(data)
    x_transform, _ = pp.transform(data)
    np.testing.assert_allclose(x_transform, x_fit, atol=1e-8)
