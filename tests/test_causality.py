"""Production contract: causality, exact reconstruction, no train/serve skew.

See `docs/changes/2026-10-22-decomposicao-saidas.md`. A forecast at
origin `t0` may only depend on the rows before `t0` and on the horizon
values of the columns known ahead of time.
"""

import numpy as np
import pytest
from ewtpy import EWT1D

from mimo_sys.preprocessors import TimeSeriesPreprocessor
from mimo_sys.preprocessors.timeseries import ewt_causal_components
from mimo_sys.training.data import INPUT_COLS, load_splits
from mimo_sys.training.evaluation import forecast_at_origin
from mimo_sys.training.features import (
    atrous_components,
    ewt_online_components,
    seasonal_components,
)

RNG = np.random.default_rng(0)
PUMPS = ("freq_b1", "freq_b2", "freq_b3")
EXACT = 1e-10
# A clearly visible effect, far above float noise.
VISIBLE = 1e-3


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    t = np.arange(600)
    return (
        10
        + 3 * np.sin(2 * np.pi * t / 24)
        + np.sin(2 * np.pi * t / 7)
        + 0.01 * t
        + RNG.normal(0, 0.3, t.size)
    )


def _boundaries(x: np.ndarray, n: int = 3) -> np.ndarray:
    return EWT1D(x, N=n)[2]


@pytest.mark.parametrize(
    "decompose",
    [
        lambda x: atrous_components(x, 4),
        lambda x: ewt_online_components(x, _boundaries(x[:400]), window=96),
        lambda x: seasonal_components(x, 24),
    ],
    ids=["atrous", "ewt_c", "seasonal24"],
)
def test_componentes_somam_exato_e_sao_causais(
    series: np.ndarray, decompose
) -> None:
    """Soma = sinal (R3) e mexer em t >= t0 não muda nada antes (R1)."""
    comps = decompose(series)
    np.testing.assert_allclose(comps.sum(axis=1), series, atol=EXACT)
    t0 = 450
    perturbed = series.copy()
    perturbed[t0:] += RNG.normal(0, 100, series.size - t0)
    np.testing.assert_array_equal(decompose(perturbed)[:t0], comps[:t0])


def test_ewt_do_ewtpy_nao_soma_o_sinal(series: np.ndarray) -> None:
    """Documenta: componentes do EWT1D não reconstroem; o EWT-c sim."""
    ewt, _, bounds = EWT1D(series, N=3)
    assert np.abs(ewt.sum(axis=1) - series).max() > VISIBLE
    window = series[-168:]
    fixed = ewt_causal_components(window, bounds)
    np.testing.assert_allclose(fixed.sum(axis=1), window, atol=EXACT)


def _pp(**steps: bool) -> TimeSeriesPreprocessor:
    options = {
        "remove_outliers": False,
        "apply_ewt": False,
        "detrend": False,
        "apply_filter": False,
        **steps,
    }
    return TimeSeriesPreprocessor(
        window_size=24, horizon=6, plot=False, **options
    )


@pytest.mark.parametrize(
    "steps",
    [
        {"apply_ewt": True, "detrend": True},
        {"apply_filter": True},
    ],
    ids=["ewt_detrend", "savgol_centrado"],
)
def test_preprocessor_nao_causal_vaza_e_e_barrado(
    series: np.ndarray, steps: dict
) -> None:
    """EWT/detrend e Savgol centrado mudam o passado ao mexer no futuro.

    São documentados como vazamento: em produção `forecast_at_origin`
    recusa esses preprocessadores em vez de vazar.
    """
    data = np.stack([series, series[::-1]], axis=1)
    pp = _pp(**steps)
    pp.fit_transform(data[:400])
    seg = data[400:]
    t0 = 100
    perturbed = seg.copy()
    perturbed[t0:] += 50
    before = pp.transform_series(seg)[:t0]
    after = pp.transform_series(perturbed)[:t0]
    assert np.abs(after - before).max() > VISIBLE
    with pytest.raises(ValueError, match="non-causal"):
        forecast_at_origin(
            model=lambda x: x[..., :6, :1],
            preprocessor=pp,
            history=seg[:t0],
            future_known=seg[t0 : t0 + 6, :1],
            known_idx=[0],
            input_window=24,
            output_window=6,
        )


def test_preprocessor_baseline_e_causal(series: np.ndarray) -> None:
    """MinMax puro (o `BASELINE_PREPROCESSOR`) é elemento a elemento."""
    data = np.stack([series, series[::-1]], axis=1)
    pp = _pp()
    pp.fit_transform(data[:400])
    seg = data[400:]
    perturbed = seg.copy()
    perturbed[100:] += 50
    np.testing.assert_array_equal(
        pp.transform_series(perturbed)[:100], pp.transform_series(seg)[:100]
    )


def test_round_trip_com_ewt_so_fecha_lendo_a_tendencia_futura(
    series: np.ndarray,
) -> None:
    """Complementa `test_round_trip_reconstroi_segmento_novo`.

    Aquele teste é exato porque `reference=seg` soma a tendência EWT do
    horizonte calculada com os valores futuros. Com só o passado (o
    que existe em produção) a reconstrução não é exata.
    """
    data = series[:, None]
    pp = _pp(apply_ewt=True, detrend=True)
    pp.fit_transform(data[:400])
    seg = data[400:]
    _, y = pp.transform(seg)
    t0 = 100
    truth = seg[t0 : t0 + 6, 0]
    window = y[t0 - 24][None]
    future = pp.inverse_transform(window, np.array([t0]), reference=seg)
    past_only = pp.inverse_transform(
        window, np.array([t0]), reference=seg[:t0]
    )
    np.testing.assert_allclose(future[0, :, 0], truth, atol=1e-6)
    assert np.abs(past_only[0, :, 0] - truth).max() > VISIBLE


@pytest.fixture
def mlflow_tmp(tmp_path):
    """MLflow tracking isolated in a temporary SQLite store."""
    import mlflow  # noqa: PLC0415

    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(previous)


VARIANTS = {
    "baseline_full": {"future_known": tuple(INPUT_COLS)},
    "baseline_pumps": {"future_known": PUMPS},
    "atrous_unit_hibrido": {
        "future_known": PUMPS,
        "target_decomposition": "atrous",
        "component_loss_weight": 0.5,
    },
    "atrous_parent": {
        "future_known": tuple(INPUT_COLS),
        "target_decomposition": "atrous",
        "component_scale": "parent",
        "component_loss_weight": 0.0,
    },
    "ewt_c": {
        "future_known": PUMPS,
        "target_decomposition": "ewt",
        "decomposition_levels": 3,
        "ewt_window": 96,
    },
    "seasonal24": {
        "future_known": PUMPS,
        "target_decomposition": "seasonal24",
    },
    "pumps_setpoint": {
        "future_known": PUMPS,
        "setpoint_feature": True,
    },
    "pumps_limpo_preve_entrada": {
        "future_known": PUMPS,
        "clean_train": True,
        "predict_inflow": True,
    },
}


def _quick_config(name: str):
    from mimo_sys.training.train import TrainConfig  # noqa: PLC0415

    return TrainConfig(
        model="spectral_tf_net",
        model_kwargs={"n_bands": 4, "residual_hidden": 8},
        input_window=48,
        output_window=12,
        future_exog=True,
        mass_balance_feature=True,
        mass_balance_window=168,
        cascade_weight=0.5,
        uncertainty_weighted=True,
        max_epochs=1,
        accelerator="cpu",
        experiment="test",
        save=False,
        **VARIANTS[name],
    )


def _assert_exact_reconstruction(cfg, result) -> None:
    """Componentes verdadeiras -> `reconstruct` -> saídas verdadeiras."""
    import torch  # noqa: PLC0415

    from mimo_sys.training.train import (  # noqa: PLC0415
        _windows,  # noqa: PLC2701
        target_layout,
    )

    layout = target_layout(cfg)
    if not layout.decomposed:
        return
    _, windows, _, _ = _windows(cfg, load_splits(cfg.data_prefix))
    X, y = (torch.as_tensor(a, dtype=torch.float32) for a in windows["val"])
    n_comp = len(layout.pred_cols)
    rebuilt = result.module.cpu().reconstruct(y[..., 3 : 3 + n_comp], X)
    np.testing.assert_allclose(rebuilt.numpy(), y[..., :3].numpy(), atol=1e-4)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_previsao_em_producao_e_causal_e_exata(
    mlflow_tmp: None, name: str
) -> None:
    """Treina rápido e verifica o contrato de produção ponta a ponta.

    1. Reconstrução exata: as componentes verdadeiras, passadas pelo
       `Forecaster.reconstruct`, devolvem as saídas verdadeiras (R3).
    2. Perturbar tudo o que não é conhecido a partir de `t0` não muda
       a previsão em `t0` (R1/R2).
    3. Replay: recalcular as colunas derivadas só com as linhas
       recebidas até `t0` (horizonte desconhecido = NaN) dá a mesma
       previsão que a avaliação em lote (R4).
    """
    from mimo_sys.training.train import (  # noqa: PLC0415
        derive_columns,
        known_columns,
        target_layout,
        train,
    )

    cfg = _quick_config(name)
    result = train(cfg)
    _assert_exact_reconstruction(cfg, result)

    splits = load_splits(cfg.data_prefix)
    raw = np.concatenate([splits["train"], splits["val"], splits["test"]])
    t0, h = len(raw) - 100, cfg.output_window
    known_idx = known_columns(cfg)

    def forecast(rows: np.ndarray) -> np.ndarray:
        derived = derive_columns(cfg, rows, splits["train"])
        pred, aux = forecast_at_origin(
            result.module.model.cpu().eval(),
            result.preprocessor,
            derived[:t0],
            derived[t0 : t0 + h, known_idx],
            known_idx=known_idx,
            input_window=cfg.input_window,
            output_window=h,
            layout=target_layout(cfg),
            return_aux=True,
        )
        return np.concatenate([pred, aux], axis=1)

    reference = forecast(raw)
    assert np.isfinite(reference).all()

    known_inputs = [INPUT_COLS.index(c) for c in cfg.future_known]
    unknown = [c for c in range(raw.shape[1]) if c not in known_inputs]
    perturbed = raw.copy()
    noise = RNG.normal(0, 100, (len(raw) - t0, len(unknown)))
    perturbed[t0:, unknown] += noise
    perturbed[t0 + h :, known_inputs] += 100
    np.testing.assert_allclose(forecast(perturbed), reference, atol=EXACT)

    received = raw[: t0 + h].copy()
    received[t0:, unknown] = np.nan
    np.testing.assert_allclose(forecast(received), reference, atol=EXACT)


def test_clean_train_tira_janelas_com_falha_no_alvo() -> None:
    """Nenhum alvo de treino contém falha; a entrada fica interpolada.

    Desescalados, os alvos mantidos batem exatamente com o dado bruto
    nas origens sem falha no horizonte; logo nenhum rótulo é inventado.
    """
    from mimo_sys.training.features import bad_sample_mask  # noqa: PLC0415
    from mimo_sys.training.train import (  # noqa: PLC0415
        TrainConfig,
        _windows,  # noqa: PLC2701
    )

    splits = load_splits()
    raw = splits["train"]
    bad = bad_sample_mask(raw)
    cfg = TrainConfig(input_window=48, output_window=12, clean_train=True)
    pp, windows, _, _ = _windows(cfg, splits)
    L, H = cfg.input_window, cfg.output_window
    starts = [
        i
        for i in range(len(raw) - L - H + 1)
        if not bad[i + L : i + L + H].any()
    ]
    y = windows["train"][1]
    assert len(y) == len(starts) < len(raw) - L - H + 1
    width = len(pp.scaler.scale_)
    flat = np.zeros((y.shape[0] * H, width))
    flat[:, [4, 5, 6]] = y.reshape(-1, 3)
    unscaled = pp.scaler.inverse_transform(flat)[:, [4, 5, 6]]
    expected = np.stack([raw[i + L : i + L + H, 4:7] for i in starts])
    np.testing.assert_allclose(
        unscaled.reshape(expected.shape), expected, atol=1e-9
    )


def test_fit_level_blend_recupera_pesos_conhecidos() -> None:
    from mimo_sys.training.mass_balance_cascade import (  # noqa: PLC0415
        apply_level_blend,
        fit_level_blend,
    )

    truth = RNG.normal(size=(500, 12))
    direct = truth + RNG.normal(0, 1, truth.shape)
    cascade = truth + RNG.normal(0, 1, truth.shape)
    weights = np.linspace(1, 0, 12)
    blended = apply_level_blend(weights, direct, cascade)
    # truth == blended exactly when the cascade/direct errors cancel:
    fitted = fit_level_blend(blended, direct, cascade)
    np.testing.assert_allclose(fitted, weights, atol=EXACT)
