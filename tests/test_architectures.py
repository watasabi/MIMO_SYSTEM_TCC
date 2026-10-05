"""Contract tests for the MIMO architectures.

These check shapes, seed determinism and the properties each
architecture is supposed to provide (message bottleneck, orthogonal
graph Fourier basis, k-NN masking, noise injection only while
training). They do not check forecast quality — that belongs to the
training notebooks.
"""

import pytest
import torch

from mimo_sys.architectures import (
    MPNNForecaster,
    Seq2SeqLatentGNN,
    StemGNN,
    SymbolicGraphNetwork,
)
from mimo_sys.architectures.seq2seq_latent_gnn import GaussianSmoothing
from mimo_sys.architectures.stemgnn import (
    chebyshev_basis,
    normalised_laplacian,
)
from mimo_sys.architectures.symbolic_graph_network import (
    SavitzkyGolayFilter,
)

SEED = 42
N_NODES = 7
TARGET_IDX = [4, 5, 6]
INPUT_WINDOW = 12
OUTPUT_WINDOW = 4
BATCH = 5


def build(name: str, **overrides: int) -> torch.nn.Module:
    """Instantiate an architecture with the shared test geometry."""
    kwargs = {
        "n_nodes": N_NODES,
        "input_window": INPUT_WINDOW,
        "output_window": OUTPUT_WINDOW,
        "target_idx": TARGET_IDX,
    }
    kwargs.update(overrides)
    builders = {
        "mpnn": MPNNForecaster,
        "stemgnn": StemGNN,
        "seq2seq": Seq2SeqLatentGNN,
        "sgn": SymbolicGraphNetwork,
    }
    return builders[name](**kwargs)


@pytest.fixture
def window() -> torch.Tensor:
    """A deterministic batch of input windows."""
    torch.manual_seed(SEED)
    return torch.randn(BATCH, INPUT_WINDOW, N_NODES)


@pytest.mark.parametrize("name", ["mpnn", "stemgnn", "seq2seq", "sgn"])
def test_output_shape(name: str, window: torch.Tensor) -> None:
    torch.manual_seed(SEED)
    model = build(name).eval()
    out = model(window)
    assert out.shape == (BATCH, OUTPUT_WINDOW, len(TARGET_IDX))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["mpnn", "stemgnn", "seq2seq", "sgn"])
def test_seed_determinism(name: str, window: torch.Tensor) -> None:
    torch.manual_seed(SEED)
    first = build(name).eval()(window)
    torch.manual_seed(SEED)
    second = build(name).eval()(window)
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("name", ["mpnn", "sgn"])
def test_single_step_horizon(name: str, window: torch.Tensor) -> None:
    torch.manual_seed(SEED)
    model = build(name, output_window=1).eval()
    assert model(window).shape == (BATCH, 1, len(TARGET_IDX))


def test_mpnn_has_no_self_messages(window: torch.Tensor) -> None:
    torch.manual_seed(SEED)
    model = build("mpnn").eval()
    model(window)
    assert model.last_messages is not None
    diagonal = model.last_messages.diagonal(dim1=1, dim2=2)
    assert torch.count_nonzero(diagonal) == 0


def test_mpnn_edge_strength_is_a_topology_matrix(
    window: torch.Tensor,
) -> None:
    torch.manual_seed(SEED)
    model = build("mpnn").eval()
    strength = model.edge_strength(window)
    assert strength.shape == (N_NODES, N_NODES)
    assert torch.count_nonzero(strength.diagonal()) == 0
    assert (strength >= 0).all()


def test_mpnn_message_l1_needs_a_forward_pass() -> None:
    model = build("mpnn")
    with pytest.raises(RuntimeError, match="forward"):
        model.message_l1()


def test_mpnn_message_io_shapes(window: torch.Tensor) -> None:
    torch.manual_seed(SEED)
    model = build("mpnn", message_dim=2).eval()
    inputs, messages = model.collect_message_io(window)
    n_edges = BATCH * N_NODES * (N_NODES - 1)
    assert inputs.shape == (n_edges, 2 * INPUT_WINDOW)
    assert messages.shape == (n_edges, 2)


def test_mpnn_update_io_shapes(window: torch.Tensor) -> None:
    torch.manual_seed(SEED)
    model = build("mpnn", message_dim=2).eval()
    inputs, outputs = model.collect_update_io(window)
    assert inputs.shape == (BATCH * N_NODES, INPUT_WINDOW + 2)
    assert outputs.shape == (BATCH * N_NODES, OUTPUT_WINDOW)


def test_mpnn_rejects_zero_layers() -> None:
    with pytest.raises(ValueError, match="n_layers"):
        build("mpnn", n_layers=0)


def test_chebyshev_basis_follows_the_recurrence() -> None:
    torch.manual_seed(SEED)
    adjacency = torch.rand(N_NODES, N_NODES)
    order = 4
    basis = chebyshev_basis(adjacency, order=order)
    laplacian = normalised_laplacian(adjacency)

    assert basis.shape == (order, N_NODES, N_NODES)
    torch.testing.assert_close(basis[0], torch.eye(N_NODES))
    torch.testing.assert_close(basis[1], laplacian)
    for k in range(2, order):
        expected = 2 * laplacian @ basis[k - 1] - basis[k - 2]
        torch.testing.assert_close(basis[k], expected)


def test_chebyshev_basis_rejects_order_below_two() -> None:
    with pytest.raises(ValueError, match="order"):
        chebyshev_basis(torch.rand(N_NODES, N_NODES), order=1)


def test_stemgnn_is_reproducible(window: torch.Tensor) -> None:
    """Guards the eigenbasis ambiguity described in `stemgnn`.

    The learned attention is near-uniform, so the Laplacian is
    near-degenerate; an eigendecomposition-based transform drifts
    between calls of the same weights, the Chebyshev one does not.
    """
    torch.manual_seed(SEED)
    model = build("stemgnn").eval()
    with torch.no_grad():
        torch.testing.assert_close(model(window), model(window))


def test_stemgnn_learns_an_adjacency(window: torch.Tensor) -> None:
    torch.manual_seed(SEED)
    model = build("stemgnn").eval()
    model(window)
    assert model.last_adjacency is not None
    assert model.last_adjacency.shape == (N_NODES, N_NODES)


def test_seq2seq_teacher_forcing_needs_a_target(
    window: torch.Tensor,
) -> None:
    torch.manual_seed(SEED)
    model = build("seq2seq").eval()
    with pytest.raises(ValueError, match="teacher forcing"):
        model(window, teacher_forcing_ratio=0.5)


def test_seq2seq_accepts_teacher_forcing(
    window: torch.Tensor,
) -> None:
    torch.manual_seed(SEED)
    model = build("seq2seq")
    target = torch.randn(BATCH, OUTPUT_WINDOW, len(TARGET_IDX))
    out = model(window, target=target, teacher_forcing_ratio=1.0)
    assert out.shape == target.shape


def test_seq2seq_knn_mask_keeps_only_top_k(
    window: torch.Tensor,
) -> None:
    top_k = 2
    torch.manual_seed(SEED)
    model = build("seq2seq", top_k=top_k).eval()
    model.gcn.static_adjacency.data.zero_()
    # One sample at a time: `last_adjacency` averages over the batch,
    # and each sample masks a different set of neighbours.
    model(window[:1])
    adjacency = model.gcn.last_adjacency
    assert adjacency is not None
    nonzero_per_row = torch.count_nonzero(adjacency, dim=-1)
    assert (nonzero_per_row <= top_k).all()


def test_seq2seq_rejects_an_empty_fourier_band() -> None:
    with pytest.raises(ValueError, match="band is empty"):
        build("seq2seq", k_low=3, k_high=3)


def test_gaussian_smoothing_passes_short_horizons_through() -> None:
    smoothing = GaussianSmoothing(len(TARGET_IDX), kernel_size=5)
    short = torch.randn(BATCH, 1, len(TARGET_IDX))
    torch.testing.assert_close(smoothing(short), short)


def test_savgol_preserves_a_polynomial() -> None:
    """A degree-2 filter must leave a degree-2 ramp untouched."""
    filtered = SavitzkyGolayFilter(1, window=5, order=2)
    ramp = torch.arange(20, dtype=torch.float32) ** 2
    series = ramp.view(1, -1, 1)
    out = filtered(series)
    torch.testing.assert_close(
        out[:, 2:-2, :], series[:, 2:-2, :], atol=1e-3, rtol=1e-3
    )


def test_sgn_injects_noise_only_while_training(
    window: torch.Tensor,
) -> None:
    torch.manual_seed(SEED)
    model = build("sgn", noise_scale=1.0)

    model.eval()
    torch.manual_seed(SEED)
    first = model(window)
    torch.manual_seed(SEED + 1)
    second = model(window)
    torch.testing.assert_close(first, second)

    model.train()
    torch.manual_seed(SEED)
    noisy_first = model(window)
    torch.manual_seed(SEED + 1)
    noisy_second = model(window)
    assert not torch.allclose(noisy_first, noisy_second)


def test_sgn_rejects_negative_noise_scale() -> None:
    with pytest.raises(ValueError, match="noise_scale"):
        build("sgn", noise_scale=-1.0)


@pytest.mark.parametrize("name", ["mpnn", "stemgnn", "seq2seq", "sgn"])
def test_backward_reaches_every_parameter(
    name: str, window: torch.Tensor
) -> None:
    """A full backward pass must populate every gradient.

    This is what guards the non-in-place residual in `StemGNN`: an
    in-place write there breaks autograd instead of merely being slow.
    """
    torch.manual_seed(SEED)
    model = build(name)
    target = torch.randn(BATCH, OUTPUT_WINDOW, len(TARGET_IDX))
    loss = torch.nn.functional.mse_loss(model(window), target)
    # StemGNN trains forecast and backcast jointly; without the
    # reconstruction term its last backcast head gets no gradient.
    auxiliary = (
        model.backcast_loss(window)
        if hasattr(model, "backcast_loss")
        else torch.zeros(())
    )
    (loss + auxiliary).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"sem gradiente: {missing}"
