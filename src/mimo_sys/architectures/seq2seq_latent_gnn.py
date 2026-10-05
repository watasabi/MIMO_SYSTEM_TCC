"""Fourier-enhanced Seq2Seq latent GNN (Seman, Stefenon, Yow, Coelho &
Mariani, Eng. Appl. AI 167:113939, 2026).

Reference implementation of `docs/pappers/Seq2SeqLatentGNN_2026.pdf` —
the reservoir-domain work of the same group as the target paper, kept as
the "what the group already did" baseline. Like StemGNN it is a
black box for extraction purposes: the GCN message sees a latent that
has already been through a spectral filter, so its arguments have no
physical name.

The adaptive hybrid detrending of the original (Savitzky-Golay plus
polynomial regression) is preprocessing, not architecture, and lives in
`mimo_sys.preprocessors`; it is deliberately not duplicated here.
"""

import math

import torch
from torch import nn

__all__ = [
    "FourierFilter",
    "GaussianSmoothing",
    "LatentCorrelationGCN",
    "Seq2SeqLatentGNN",
]


class FourierFilter(nn.Module):
    """Low-rank spectral filter keeping modes `[k_low, k_high)`.

    Learnable complex gains are applied to the retained modes and every
    other mode is zeroed, which is the paper's `kappa1, kappa2` band.

    Args:
        n_nodes: Number of series (one gain per node per mode).
        input_window: Lag window length.
        k_low: First retained frequency index.
        k_high: One past the last retained index. Clipped to the number
            of available `rfft` bins.
    """

    def __init__(
        self,
        n_nodes: int,
        input_window: int,
        k_low: int = 0,
        k_high: int = 8,
    ) -> None:
        super().__init__()
        n_bins = input_window // 2 + 1
        if not 0 <= k_low < n_bins:
            raise ValueError("k_low outside the available rfft bins")
        self.k_low = k_low
        self.k_high = min(k_high, n_bins)
        n_kept = self.k_high - self.k_low
        if n_kept < 1:
            raise ValueError("the retained band is empty")
        self.gain_real = nn.Parameter(torch.ones(n_nodes, n_kept))
        self.gain_imag = nn.Parameter(torch.zeros(n_nodes, n_kept))
        self.input_window = input_window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of `(batch, nodes, window)`. Same shape out."""
        freq = torch.fft.rfft(x, dim=-1)
        gain = torch.complex(self.gain_real, self.gain_imag)
        kept = freq[..., self.k_low : self.k_high] * gain.unsqueeze(0)
        filtered = torch.zeros_like(freq)
        filtered[..., self.k_low : self.k_high] = kept
        return torch.fft.irfft(filtered, n=self.input_window, dim=-1)


class LatentCorrelationGCN(nn.Module):
    """Dynamic attention adjacency with a k-NN mask, plus a static one.

    `A_dyn = softmax(Q K^T / sqrt(d))` keeps only each node's `top_k`
    strongest neighbours; a learnable `A_static` is added on top, as in
    the paper's `A_dyn = QK^T/sqrt(d) + A_static`.

    Args:
        n_nodes: Number of series.
        in_dim: Input feature width per node.
        out_dim: Output feature width per node.
        top_k: Neighbours retained per node by the mask.
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        out_dim: int,
        top_k: int = 3,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= n_nodes:
            raise ValueError("top_k must be in [1, n_nodes]")
        self.to_query = nn.Linear(in_dim, in_dim)
        self.to_key = nn.Linear(in_dim, in_dim)
        self.project = nn.Linear(in_dim, out_dim)
        self.static_adjacency = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        self.top_k = top_k
        self.in_dim = in_dim
        # Filled by the last forward pass: [nodes, nodes].
        self.last_adjacency: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of `(batch, nodes, in_dim)`.

        Returns:
            `(batch, nodes, out_dim)`.
        """
        query = self.to_query(x)
        key = self.to_key(x)
        scale = math.sqrt(self.in_dim)
        scores = torch.matmul(query, key.transpose(1, 2)) / scale

        kth = scores.topk(self.top_k, dim=-1).values[..., -1:]
        masked = scores.masked_fill(scores < kth, float("-inf"))
        # One expression, not `+=`: an in-place write on the softmax
        # output breaks autograd.
        adjacency = torch.softmax(
            masked, dim=-1
        ) + self.static_adjacency.unsqueeze(0)

        self.last_adjacency = adjacency.detach().mean(dim=0)
        return self.project(torch.matmul(adjacency, x))


class GaussianSmoothing(nn.Module):
    """Fixed Gaussian smoothing along the forecast horizon.

    Acts as the identity when the horizon is shorter than the kernel,
    so single-step forecasting is unaffected.

    Args:
        n_targets: Number of output series (depthwise channels).
        kernel_size: Kernel width (odd).
        sigma: Gaussian standard deviation in samples.
    """

    def __init__(
        self,
        n_targets: int,
        kernel_size: int = 3,
        sigma: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        offsets = torch.arange(kernel_size) - kernel_size // 2
        kernel = torch.exp(-(offsets**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        self.register_buffer(
            "kernel", kernel.view(1, 1, -1).repeat(n_targets, 1, 1)
        )
        self.kernel_size = kernel_size
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of `(batch, horizon, n_targets)`. Same shape out."""
        if x.shape[1] < self.kernel_size:
            return x
        channels = x.transpose(1, 2)
        smoothed = nn.functional.conv1d(
            channels,
            self.kernel,
            padding=self.kernel_size // 2,
            groups=self.n_targets,
        )
        return smoothed.transpose(1, 2)


class Seq2SeqLatentGNN(nn.Module):
    """Fourier filter -> latent GCN -> Seq2Seq LSTM with attention.

    Args:
        n_nodes: Number of series.
        input_window: Lag window length.
        output_window: Forecast horizon.
        target_idx: Indices of the output nodes, in return order.
        hidden_size: LSTM hidden size and GCN output width.
        num_layers: LSTM layers in encoder and decoder.
        top_k: Neighbours kept by the GCN mask.
        k_low: First retained Fourier mode.
        k_high: One past the last retained Fourier mode.
        smoothing_kernel: Width of the output Gaussian smoothing (odd).

    Shape:
        - Input: `(batch, input_window, n_nodes)`
        - Output: `(batch, output_window, len(target_idx))`
    """

    def __init__(
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        top_k: int = 3,
        k_low: int = 0,
        k_high: int = 8,
        smoothing_kernel: int = 3,
    ) -> None:
        super().__init__()
        n_targets = len(target_idx)
        self.fourier = FourierFilter(n_nodes, input_window, k_low, k_high)
        self.gcn = LatentCorrelationGCN(
            n_nodes, input_window, input_window, top_k
        )
        self.encoder = nn.LSTM(
            n_nodes, hidden_size, num_layers, batch_first=True
        )
        self.decoder = nn.LSTM(
            n_targets, hidden_size, num_layers, batch_first=True
        )
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.head = nn.Linear(hidden_size, n_targets)
        self.smoothing = GaussianSmoothing(n_targets, smoothing_kernel)
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )
        self.output_window = output_window
        self.n_targets = n_targets

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Run the forecast, optionally with teacher forcing.

        Args:
            x: `(batch, input_window, n_nodes)`.
            target: `(batch, output_window, n_targets)`, required when
                `teacher_forcing_ratio > 0`.
            teacher_forcing_ratio: Probability of feeding the measured
                value instead of the previous prediction, per step.

        Returns:
            `(batch, output_window, n_targets)`.
        """
        if teacher_forcing_ratio > 0.0 and target is None:
            raise ValueError("teacher forcing needs target")

        nodes_first = x.transpose(1, 2)  # [batch, nodes, window]
        filtered = self.fourier(nodes_first)
        correlated = self.gcn(filtered)
        encoded, (hidden, cell) = self.encoder(correlated.transpose(1, 2))

        step = x[:, -1:, :].index_select(2, self.target_idx)
        outputs = []
        for position in range(self.output_window):
            decoded, (hidden, cell) = self.decoder(step, (hidden, cell))
            scores = torch.bmm(decoded, encoded.transpose(1, 2))
            weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(weights, encoded)
            combined = torch.tanh(
                self.attention_combine(torch.cat([decoded, context], dim=-1))
            )
            prediction = self.head(combined)
            outputs.append(prediction)

            use_truth = (
                target is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )
            if use_truth:
                step = target[:, position : position + 1, :]
            else:
                step = prediction

        return self.smoothing(torch.cat(outputs, dim=1))
