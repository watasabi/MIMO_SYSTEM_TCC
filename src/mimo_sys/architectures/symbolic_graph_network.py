"""Symbolic Graph Network — robust PDE discovery from noisy sparse data
(arXiv:2603.22380).

Reference implementation of
`docs/pappers/SymbolicGraphNetworks_2026.pdf`. The architecture is an
MPNN with a very low-dimensional message (`d_msg` of 1 or 2) plus two
stabilisers aimed squarely at noisy, short records:

1. **Geometric warm start** — a Savitzky-Golay polynomial filter on the
   raw window, so the network does not have to learn gradients from
   jitter (the paper's cold-start problem).
2. **Dynamic noise injection** — Gaussian perturbation of the node
   features during training, scaled by the *estimated* observation
   noise, so only operator forms that survive perturbation are kept.

Extraction is the paper's two-stage protocol, and it is why this class
exists next to `MPNNForecaster`: Stage I distils the message function
`psi` (`collect_message_io`), Stage II distils the update function `phi`
over the already-reduced aggregated message (`collect_update_io`). The
final expression is `phi . Agg . psi`.

With 600 estimation samples at `Ts = 1 h` the noise-robustness is the
relevant part of this paper for the reservoir problem, not the PDE
framing — there is no spatial grid here, the graph is the 7 system
variables.
"""

import torch
from scipy.signal import savgol_coeffs
from torch import nn

from mimo_sys.architectures.mpnn import MPNNForecaster

__all__ = ["SavitzkyGolayFilter", "SymbolicGraphNetwork"]


class SavitzkyGolayFilter(nn.Module):
    """Savitzky-Golay smoothing as a fixed depthwise convolution.

    Implemented with `scipy.signal.savgol_coeffs` baked into a frozen
    `conv1d` kernel so it runs on-device inside `forward`, rather than
    as a numpy preprocessing step.

    Args:
        n_nodes: Number of series (depthwise channels).
        window: Filter length (odd, greater than `order`).
        order: Polynomial order of the local least-squares fit.
    """

    def __init__(self, n_nodes: int, window: int = 5, order: int = 2) -> None:
        super().__init__()
        if window % 2 == 0:
            raise ValueError("window must be odd")
        if order >= window:
            raise ValueError("order must be smaller than window")
        coefficients = torch.tensor(
            savgol_coeffs(window, order).copy(), dtype=torch.float32
        )
        self.register_buffer(
            "kernel",
            coefficients.view(1, 1, -1).repeat(n_nodes, 1, 1),
        )
        self.window = window
        self.n_nodes = n_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of `(batch, window_len, n_nodes)`. Same shape out.

        Returns the input unchanged when the lag window is shorter than
        the filter, so short-window configurations stay usable.
        """
        if x.shape[1] < self.window:
            return x
        channels = x.transpose(1, 2)
        smoothed = nn.functional.conv1d(
            channels,
            self.kernel,
            padding=self.window // 2,
            groups=self.n_nodes,
        )
        return smoothed.transpose(1, 2)


class SymbolicGraphNetwork(nn.Module):
    """MPNN with S-G warm start and dynamic noise injection.

    Args:
        n_nodes: Number of series.
        input_window: Lag window length.
        output_window: Forecast horizon.
        target_idx: Indices of the output nodes, in return order.
        hidden_size: Width of the message/update MLPs.
        message_dim: Message bottleneck. The paper restricts this to 1
            or 2 on the manifold-hypothesis argument; 2 is the default
            here for parity with `MPNNForecaster`.
        n_layers: Rounds of message passing.
        savgol_window: Savitzky-Golay filter length (odd).
        savgol_order: Savitzky-Golay polynomial order.
        noise_scale: Multiplier on the estimated noise level for the
            training-time perturbation. 0 disables injection.

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
        message_dim: int = 2,
        n_layers: int = 1,
        savgol_window: int = 5,
        savgol_order: int = 2,
        noise_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be >= 0")
        self.smoothing = SavitzkyGolayFilter(
            n_nodes, savgol_window, savgol_order
        )
        self.mpnn = MPNNForecaster(
            n_nodes=n_nodes,
            input_window=input_window,
            output_window=output_window,
            target_idx=target_idx,
            hidden_size=hidden_size,
            message_dim=message_dim,
            n_layers=n_layers,
        )
        self.noise_scale = noise_scale

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Warm-start the window and, while training, perturb it.

        The noise level is estimated from the high-frequency residual
        the Savitzky-Golay filter removed, which is what makes the
        injection adaptive instead of a fixed hyperparameter.
        """
        smoothed = self.smoothing(x)
        if not self.training or self.noise_scale == 0.0:
            return smoothed
        sigma = (x - smoothed).std()
        noise = torch.randn_like(smoothed) * sigma * self.noise_scale
        return smoothed + noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the graph simulator and return the forecast."""
        return self.mpnn(self._prepare(x))

    def message_l1(self) -> torch.Tensor:
        """Mean absolute message of the last forward pass."""
        return self.mpnn.message_l1()

    @torch.no_grad()
    def edge_strength(self, x: torch.Tensor) -> torch.Tensor:
        """Mean message norm per edge, `(n_nodes, n_nodes)`."""
        return self.mpnn.edge_strength(self.smoothing(x))

    @torch.no_grad()
    def collect_message_io(
        self, x: torch.Tensor, layer: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage I: input/output pairs of the message function."""
        return self.mpnn.collect_message_io(self.smoothing(x), layer)

    @torch.no_grad()
    def collect_update_io(
        self, x: torch.Tensor, layer: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage II: input/output pairs of the update function."""
        return self.mpnn.collect_update_io(self.smoothing(x), layer)
