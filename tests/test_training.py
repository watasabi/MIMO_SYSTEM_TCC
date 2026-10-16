"""Contract tests for `mimo_sys.training` (no forecast quality)."""

import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mimo_sys.architectures import (
    MPNNForecaster,
    Seq2SeqLatentGNN,
    StemGNN,
    SymbolicGraphNetwork,
)
from mimo_sys.training.augmentations import (
    ChannelMask,
    Compose,
    FreqMask,
    Jitter,
    MagnitudeWarp,
    Mixup,
    Scaling,
    SpanMask,
    TimeWarp,
)
from mimo_sys.training.callbacks import EMA, EncoderFreeze
from mimo_sys.training.contrastive import ContrastivePretrainer, nt_xent
from mimo_sys.training.forecaster import Forecaster
from mimo_sys.training.losses import (
    MASELoss,
    SpectralLoss,
    build_loss,
    seasonal_naive_scale,
)
from mimo_sys.training.scalers import RevINWrapper

SEED = 42
N_NODES = 7
TARGET_IDX = [4, 5, 6]
INPUT_IDX = [0, 1, 2, 3]
INPUT_WINDOW = 12
OUTPUT_WINDOW = 4
BATCH = 8
ENCODED_DIMS = 3  # (batch, positions, features)

ARCHS = {
    "mpnn": MPNNForecaster,
    "sgn": SymbolicGraphNetwork,
    "stemgnn": StemGNN,
    "seq2seq": Seq2SeqLatentGNN,
}


def build(name: str, **overrides: int) -> torch.nn.Module:
    """Architecture with the small test geometry."""
    return ARCHS[name](
        n_nodes=N_NODES,
        input_window=INPUT_WINDOW,
        output_window=OUTPUT_WINDOW,
        target_idx=TARGET_IDX,
        **overrides,
    )


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic `(x, y)` batch."""
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, INPUT_WINDOW, N_NODES)
    y = torch.randn(BATCH, OUTPUT_WINDOW, len(TARGET_IDX))
    return x, y


def loader(n: int = 32) -> DataLoader:
    """Tiny synthetic dataset."""
    torch.manual_seed(SEED)
    x = torch.randn(n, INPUT_WINDOW, N_NODES)
    y = torch.randn(n, OUTPUT_WINDOW, len(TARGET_IDX))
    return DataLoader(TensorDataset(x, y), batch_size=BATCH)


def quick_trainer(**kwargs: object) -> L.Trainer:
    """CPU trainer without logging or checkpoints."""
    defaults = {
        "accelerator": "cpu",
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    defaults.update(kwargs)
    return L.Trainer(**defaults)


# ------------------------------------------------------- augmentations

AUGMENTATIONS = [
    Jitter(0.1),
    Scaling(TARGET_IDX),
    MagnitudeWarp(TARGET_IDX),
    Mixup(0.4),
    TimeWarp(),
    SpanMask(),
    ChannelMask(INPUT_IDX, p=0.5),
    FreqMask(),
]


@pytest.mark.parametrize(
    "augment", AUGMENTATIONS, ids=lambda a: type(a).__name__
)
def test_augmentation_preserva_shape(
    augment: object, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    x_aug, y_aug = augment(x, y)
    assert x_aug.shape == x.shape
    assert y_aug.shape == y.shape
    assert torch.isfinite(x_aug).all()


@pytest.mark.parametrize(
    "augment",
    [Jitter(0.1), TimeWarp(), SpanMask(), FreqMask()],
    ids=lambda a: type(a).__name__,
)
def test_augmentation_so_entrada_nao_muda_alvo(
    augment: object, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, y = batch
    _, y_aug = augment(x, y)
    torch.testing.assert_close(y_aug, y)


def test_scaling_aplica_mesmo_fator_em_x_e_y() -> None:
    """Se y repete o último passo de x, continua repetindo após Scaling."""
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, INPUT_WINDOW, N_NODES)
    y = x[:, -1:, TARGET_IDX].clone()
    x_aug, y_aug = Scaling(TARGET_IDX, sigma=0.5)(x, y)
    torch.testing.assert_close(y_aug, x_aug[:, -1:, TARGET_IDX])


def test_channel_mask_nunca_mascara_alvos(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = batch
    x_aug, _ = ChannelMask(INPUT_IDX, p=1.0)(x, y)
    torch.testing.assert_close(x_aug[..., TARGET_IDX], x[..., TARGET_IDX])
    assert not torch.allclose(x_aug[..., INPUT_IDX], x[..., INPUT_IDX])


def test_span_mask_preserva_ultimo_passo(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = batch
    x_aug, _ = SpanMask(max_frac=0.9)(x, y)
    torch.testing.assert_close(x_aug[:, -1], x[:, -1])


def test_time_warp_identidade_com_sigma_zero(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = batch
    x_aug, _ = TimeWarp(sigma=0.0)(x, y)
    torch.testing.assert_close(x_aug, x, atol=1e-5, rtol=1e-5)


def test_compose_com_p_zero_e_identidade(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, y = batch
    x_aug, y_aug = Compose(AUGMENTATIONS, p=0.0)(x, y)
    torch.testing.assert_close(x_aug, x)
    torch.testing.assert_close(y_aug, y)


# ---------------------------------------------------------------- losses


def test_spectral_e_mase_zero_quando_igual(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, y = batch
    assert SpectralLoss()(y, y).item() == pytest.approx(0.0)
    scale = seasonal_naive_scale(torch.randn(100, 3), season=24)
    assert MASELoss(scale)(y, y).item() == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["mse", "mae", "huber", "smape"])
def test_build_loss_positiva(
    name: str, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    _, y = batch
    assert build_loss(name)(y + 1.0, y).item() > 0


def test_nt_xent_menor_para_vistas_iguais() -> None:
    torch.manual_seed(SEED)
    z = torch.randn(16, 8)
    same = nt_xent(z, z.clone())
    other = nt_xent(z, torch.randn(16, 8))
    assert same < other


# ----------------------------------------------------------------- RevIN


@pytest.mark.parametrize("mode", ["standard", "robust", "minmax"])
def test_revin_inverte_normalizacao(
    mode: str, batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, _ = batch
    revin = RevINWrapper(build("mpnn"), N_NODES, TARGET_IDX, mode=mode)
    with torch.no_grad():
        revin.weight.uniform_(0.5, 2.0)
        revin.bias.normal_()
    z, loc, scale = revin.normalize(x)
    rebuilt = revin.denormalize(z[..., TARGET_IDX], loc, scale)
    torch.testing.assert_close(rebuilt, x[..., TARGET_IDX], atol=1e-4, rtol=0)


def test_revin_mantem_contrato(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, _ = batch
    out = RevINWrapper(build("stemgnn"), N_NODES, TARGET_IDX)(x)
    assert out.shape == (BATCH, OUTPUT_WINDOW, len(TARGET_IDX))


# ---------------------------------------------------------------- encode


@pytest.mark.parametrize(
    ("name", "overrides", "width"),
    [
        ("seq2seq", {"hidden_size": 16}, 16),
        ("mpnn", {"n_layers": 2, "hidden_size": 16}, None),
        ("sgn", {"n_layers": 2, "hidden_size": 16}, None),
    ],
)
def test_encode_shape(
    name: str,
    overrides: dict,
    width: int | None,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, _ = batch
    model = build(name, **overrides)
    encoded = model.encode(x)
    assert encoded.dim() == ENCODED_DIMS
    assert encoded.shape[0] == BATCH
    if width is not None:
        assert encoded.shape[-1] == width
    assert model.encoder_modules()


def test_mpnn_de_uma_camada_nao_tem_encoder(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    with pytest.raises(ValueError, match="n_layers >= 2"):
        build("mpnn").encode(batch[0])


# ------------------------------------------------------------ Forecaster


@pytest.mark.parametrize("name", list(ARCHS))
@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "augment": Compose(AUGMENTATIONS, p=1.0),
            "spectral_weight": 0.1,
            "message_l1_weight": 1e-2,
            "teacher_forcing_ratio": 0.5,
            "scheduler": "cosine_warmup",
        },
        {"scheduler": "onecycle", "loss": build_loss("huber")},
    ],
    ids=["base", "todas_flags", "onecycle_huber"],
)
def test_forecaster_treina_todas_arquiteturas(
    name: str, options: dict
) -> None:
    torch.manual_seed(SEED)
    module = Forecaster(build(name), max_epochs=1, **options)
    quick_trainer(fast_dev_run=2).fit(module, loader(), loader())


def test_forecaster_com_revin_e_backcast() -> None:
    torch.manual_seed(SEED)
    model = RevINWrapper(build("stemgnn"), N_NODES, TARGET_IDX)
    module = Forecaster(model, backcast_weight=1.0)
    quick_trainer(fast_dev_run=2).fit(module, loader(), loader())


# ------------------------------------------------------------- callbacks


def test_ema_diverge_dos_pesos_crus() -> None:
    torch.manual_seed(SEED)
    module = Forecaster(build("mpnn"), lr=1e-2, scheduler="none")
    to_vec = torch.nn.utils.parameters_to_vector
    initial = to_vec(module.model.parameters()).detach().clone()
    ema = EMA(decay=0.9)
    quick_trainer(max_epochs=2, callbacks=[ema]).fit(
        module, loader(), loader()
    )
    averaged = to_vec(ema.ema.module.parameters())
    final = to_vec(module.model.parameters())
    torch.testing.assert_close(final, averaged)
    assert not torch.allclose(averaged, initial)


def test_encoder_freeze_congela_e_libera() -> None:
    torch.manual_seed(SEED)
    module = Forecaster(build("seq2seq", hidden_size=8), scheduler="none")
    encoder = module.model.encoder
    before = [p.detach().clone() for p in encoder.parameters()]
    quick_trainer(max_epochs=1, callbacks=[EncoderFreeze(1)]).fit(
        module, loader(), loader()
    )
    for old, new in zip(before, encoder.parameters(), strict=True):
        torch.testing.assert_close(old, new)

    quick_trainer(max_epochs=2, callbacks=[EncoderFreeze(0)]).fit(
        module, loader(), loader()
    )
    assert all(p.requires_grad for p in encoder.parameters())


# ----------------------------------------------------------- contrastive


def test_pretreino_contrastivo_mexe_so_no_encoder() -> None:
    torch.manual_seed(SEED)
    model = build("seq2seq", hidden_size=8)
    encoder_before = [p.detach().clone() for p in model.encoder.parameters()]
    head_before = [p.detach().clone() for p in model.head.parameters()]
    data = loader()
    pretrainer = ContrastivePretrainer(
        model,
        Compose([Jitter(0.1), Scaling(TARGET_IDX)], p=1.0),
        TARGET_IDX,
        sample=next(iter(data))[0],
    )
    quick_trainer(max_epochs=1).fit(pretrainer, data)
    moved = [
        not torch.allclose(old, new)
        for old, new in zip(encoder_before, model.encoder.parameters())
    ]
    assert any(moved)
    for old, new in zip(head_before, model.head.parameters(), strict=True):
        torch.testing.assert_close(old, new)


# ------------------------------------------------------- end to end


@pytest.fixture
def mlflow_tmp(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """MLflow tracking isolated in a temporary SQLite store."""
    import mlflow  # noqa: PLC0415

    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    mlflow.set_tracking_uri(previous)


def test_train_ponta_a_ponta_com_todas_tecnicas(mlflow_tmp: None) -> None:
    from mimo_sys.training import TrainConfig, train  # noqa: PLC0415
    from mimo_sys.training.data import N_TARGETS  # noqa: PLC0415

    result = train(
        TrainConfig(
            model="seq2seq_latent_gnn",
            model_kwargs={"hidden_size": 8},
            revin="standard",
            augment=True,
            mixup_alpha=0.2,
            spectral_weight=0.1,
            ema_decay=0.99,
            pretrain_epochs=1,
            freeze_encoder_epochs=1,
            max_epochs=2,
            accelerator="cpu",
            experiment="test",
            save=False,
        )
    )
    assert result.y_pred.shape == result.y_true.shape
    assert result.y_pred.shape[1] == N_TARGETS
    assert result.metrics["r2"].notna().all()
    assert result.best_val_mse > 0


def test_tune_dois_trials(mlflow_tmp: None) -> None:
    optuna = pytest.importorskip("optuna")
    from mimo_sys.training import TrainConfig  # noqa: PLC0415
    from mimo_sys.training.tuning import suggest_config, tune  # noqa: PLC0415

    base = TrainConfig(max_epochs=1, accelerator="cpu", experiment="test")
    study = tune("mpnn", n_trials=2, base=base)
    assert len(study.trials) == 2  # noqa: PLR2004
    best = suggest_config(
        optuna.trial.FixedTrial(study.best_params), "mpnn", base
    )
    assert best.model_kwargs["hidden_size"] in {32, 64, 128}


# ------------------------------------------------------- explainability


def test_gradient_saliency_shape_e_nao_negativo() -> None:
    from mimo_sys.training.explainability import (  # noqa: PLC0415
        gradient_saliency,
    )

    model = build("seq2seq", hidden_size=8)
    windows = torch.randn(10, INPUT_WINDOW, N_NODES)
    sal = gradient_saliency(model, windows)
    assert sal.shape == (INPUT_WINDOW, N_NODES)
    assert (sal >= 0).all()


def test_gradient_saliency_lag_recente_pesa_mais_num_caso_trivial() -> None:
    """Num modelo que só usa o último passo, a saliência concentra lá."""
    from mimo_sys.training.explainability import (  # noqa: PLC0415
        gradient_saliency,
    )

    class LastStepOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(N_NODES, len(TARGET_IDX))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(x[:, -1:, :]).expand(-1, OUTPUT_WINDOW, -1)

    model = LastStepOnly()
    windows = torch.randn(16, INPUT_WINDOW, N_NODES)
    sal = gradient_saliency(model, windows, target_idx=list(range(2)))
    per_lag = sal.sum(axis=1)
    assert per_lag[-1] == per_lag.max()
    assert per_lag[:-1].max() == 0


def test_attention_by_lag_soma_um_e_tem_input_window() -> None:
    from mimo_sys.training.explainability import (  # noqa: PLC0415
        attention_by_lag,
    )

    model = build("seq2seq", hidden_size=8)
    windows = torch.randn(6, INPUT_WINDOW, N_NODES)
    attn = attention_by_lag(model, windows)
    assert attn.shape == (INPUT_WINDOW,)
    assert attn.sum() == pytest.approx(1.0, abs=1e-4)


def test_top_lags_recupera_lags_dominantes() -> None:
    from mimo_sys.training.explainability import top_lags  # noqa: PLC0415

    importance = np.zeros(10)
    importance[[2, 7, 9]] = [5.0, 3.0, 1.0]
    chosen = top_lags(importance, k=3, input_window=10)
    # lag-before-last-step: índice 9 -> 0, 7 -> 2, 2 -> 7
    assert set(chosen.tolist()) == {0, 2, 7}


# ------------------------------------------------------- spectral loss


def test_decaying_freq_weights_decresce_e_fica_positivo() -> None:
    from mimo_sys.training.losses import decaying_freq_weights  # noqa: PLC0415

    w = decaying_freq_weights(24, cutoff_period=12.0, decay=2.0)
    tolerance = 1e-6
    assert w[0] == pytest.approx(1.0)
    assert (w > 0).all()
    assert (np.diff(w.numpy()) <= tolerance).all()


def test_spectral_loss_com_pesos_zero_em_igual(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    from mimo_sys.training.losses import (  # noqa: PLC0415
        SpectralLoss,
        decaying_freq_weights,
    )

    _, y = batch
    weights = decaying_freq_weights(y.shape[1], cutoff_period=6.0)
    loss = SpectralLoss(freq_weights=weights)
    assert loss(y, y).item() == pytest.approx(0.0)
    assert loss(y + 1.0, y).item() > 0


# -------------------------------------------------- leaky integrator


def test_leaky_integrator_mantem_contrato(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    from mimo_sys.training.integrator import (  # noqa: PLC0415
        LeakyIntegratorWrapper,
    )

    x, _ = batch
    model = LeakyIntegratorWrapper(
        build("mpnn"), target_idx=TARGET_IDX, state_idx=[5]
    )
    out = model(x)
    assert out.shape == (BATCH, OUTPUT_WINDOW, len(TARGET_IDX))


def test_leaky_integrator_nao_afeta_sa_das_nao_integradas(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    from mimo_sys.training.integrator import (  # noqa: PLC0415
        LeakyIntegratorWrapper,
    )

    x, _ = batch
    torch.manual_seed(SEED)
    raw_model = build("mpnn")
    torch.manual_seed(SEED)
    wrapped = LeakyIntegratorWrapper(
        build("mpnn"), target_idx=TARGET_IDX, state_idx=[5]
    )
    wrapped.model.load_state_dict(raw_model.state_dict())
    out_raw = raw_model.eval()(x)
    out_wrapped = wrapped.eval()(x)
    torch.testing.assert_close(out_wrapped[..., 0], out_raw[..., 0])
    torch.testing.assert_close(out_wrapped[..., 2], out_raw[..., 2])
    assert not torch.allclose(out_wrapped[..., 1], out_raw[..., 1])


def test_leaky_integrator_leak_proximo_de_um_e_soma_cumulativa() -> None:
    from mimo_sys.training.integrator import (  # noqa: PLC0415
        LeakyIntegratorWrapper,
    )

    class ConstantDelta(torch.nn.Module):
        def forward(self, x: torch.Tensor, **kw: object) -> torch.Tensor:  # noqa: PLR6301
            return torch.ones(x.shape[0], OUTPUT_WINDOW, len(TARGET_IDX))

    model = LeakyIntegratorWrapper(
        ConstantDelta(),
        target_idx=TARGET_IDX,
        state_idx=[5],
        init_leak=1.0 - 1e-7,
    )
    x = torch.zeros(1, INPUT_WINDOW, N_NODES)
    x[0, -1, 5] = 10.0
    out = model(x)
    expected = 10.0 + torch.arange(1, OUTPUT_WINDOW + 1, dtype=torch.float32)
    torch.testing.assert_close(out[0, :, 1], expected, atol=1e-3, rtol=0)


# -------------------------------------------------- per-output heads


def test_per_output_head_identidade_na_inicializacao(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Correction heads start at zero: output == backbone's own forecast."""
    from mimo_sys.training.output_heads import (  # noqa: PLC0415
        PerOutputHeadWrapper,
    )

    x, _ = batch
    torch.manual_seed(SEED)
    raw_model = build("mpnn").eval()
    wrapped = PerOutputHeadWrapper(raw_model, n_targets=len(TARGET_IDX)).eval()
    torch.testing.assert_close(wrapped(x), raw_model(x))


def test_per_output_head_mantem_contrato(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    from mimo_sys.training.output_heads import (  # noqa: PLC0415
        PerOutputHeadWrapper,
    )

    x, _ = batch
    model = PerOutputHeadWrapper(build("mpnn"), n_targets=len(TARGET_IDX))
    out = model(x)
    assert out.shape == (BATCH, OUTPUT_WINDOW, len(TARGET_IDX))
    out.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing


# -------------------------------------------------- mass balance feature


def test_mass_balance_feature_e_cumsum_causal() -> None:
    from mimo_sys.training.features import (  # noqa: PLC0415
        add_mass_balance_feature,
    )

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(50, N_NODES))
    inflow_idx, outflow_idx = 3, 4
    out = add_mass_balance_feature(raw, inflow_idx, outflow_idx)
    assert out.shape == (50, N_NODES + 1)
    np.testing.assert_allclose(out[:, :N_NODES], raw)
    expected = np.cumsum(raw[:, inflow_idx] - raw[:, outflow_idx])
    np.testing.assert_allclose(out[:, N_NODES], expected)
    # causal: value at row i only depends on rows <= i
    out_prefix = add_mass_balance_feature(raw[:30], inflow_idx, outflow_idx)
    np.testing.assert_allclose(out[:30, N_NODES], out_prefix[:, N_NODES])


# -------------------------------------------------- uncertainty weighting


def test_uncertainty_weighted_loss_pondera_e_tem_gradiente() -> None:
    from mimo_sys.training.losses import (  # noqa: PLC0415
        UncertaintyWeightedLoss,
    )

    loss_fn = UncertaintyWeightedLoss(n_outputs=3)
    per_output = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    total = loss_fn(per_output)
    assert total.item() > 0
    total.backward()
    assert per_output.grad is not None
    assert loss_fn.log_vars.grad is not None
    assert loss_fn.weights().shape == (3,)


def test_uncertainty_weighted_loss_nao_zera_peso_do_termo_dificil() -> None:
    """The `+ s_i` term stops the optimiser from ignoring a hard term."""
    from mimo_sys.training.losses import (  # noqa: PLC0415
        UncertaintyWeightedLoss,
    )

    loss_fn = UncertaintyWeightedLoss(n_outputs=2)
    optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.1)
    easy_hard = torch.tensor([0.01, 10.0])
    for _ in range(200):
        optimizer.zero_grad()
        loss_fn(easy_hard).backward()
        optimizer.step()
    weights = loss_fn.weights()
    floor = 1e-3
    # the hard term's weight shrinks (down-weighted) but never collapses
    # to (near) zero relative to the easy one's.
    assert weights[1] < weights[0]
    assert weights[1] > floor


# -------------------------------------------------------- ensemble


def test_average_predictions() -> None:
    from mimo_sys.training.ensemble import average_predictions  # noqa: PLC0415

    a = np.ones((5, 3))
    b = np.zeros((5, 3))
    np.testing.assert_allclose(average_predictions([a, b]), 0.5)


def test_fit_ensemble_weights_recupera_membro_perfeito() -> None:
    """Se um membro acerta exatamente, o nnls deve pesar só nele."""
    from mimo_sys.training.ensemble import (  # noqa: PLC0415
        fit_ensemble_weights,
        weighted_predictions,
    )

    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(50, 2))
    perfect = y_true.copy()
    noisy = y_true + rng.normal(scale=5.0, size=y_true.shape)
    weights = fit_ensemble_weights([perfect, noisy], y_true)
    assert weights.shape == (2, 2)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-6)
    combined = weighted_predictions([perfect, noisy], weights)
    np.testing.assert_allclose(combined, y_true, atol=1e-6)


def test_select_ensemble_subset_descarta_membro_ruim() -> None:
    from mimo_sys.training.ensemble import (  # noqa: PLC0415
        select_ensemble_subset,
    )

    rng = np.random.default_rng(0)
    y_val = rng.normal(size=(50, 2))
    preds = {
        "good_a": y_val + rng.normal(scale=0.05, size=y_val.shape),
        "good_b": y_val + rng.normal(scale=0.05, size=y_val.shape),
        "bad": rng.normal(scale=10.0, size=y_val.shape),
    }
    subset = select_ensemble_subset(preds, y_val)
    assert "bad" not in subset
    assert "good_a" in subset
    assert "good_b" in subset


# -------------------------------------------------------- features


def test_add_ewma_feature_e_causal_e_suave() -> None:
    from mimo_sys.training.features import add_ewma_feature  # noqa: PLC0415

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(40, 3))
    out = add_ewma_feature(raw, col_idx=1, span=5)
    assert out.shape == (40, 4)
    np.testing.assert_allclose(out[:, :3], raw)
    out_prefix = add_ewma_feature(raw[:20], col_idx=1, span=5)
    np.testing.assert_allclose(out[:20, 3], out_prefix[:, 3])
    # smoother than the raw column: lower variance of first differences
    assert np.diff(out[:, 3]).std() < np.diff(raw[:, 1]).std()


def test_greedy_forward_selection_descarta_membro_ruim() -> None:
    from mimo_sys.training.ensemble import (  # noqa: PLC0415
        greedy_forward_selection,
    )

    rng = np.random.default_rng(0)
    y_val = rng.normal(size=(50, 2))
    preds = {
        "good_a": y_val + rng.normal(scale=0.05, size=y_val.shape),
        "good_b": y_val + rng.normal(scale=0.05, size=y_val.shape),
        "bad": rng.normal(scale=10.0, size=y_val.shape),
    }
    subset = greedy_forward_selection(preds, y_val)
    assert "bad" not in subset
    assert "good_a" in subset
