"""Non-graph neural baselines: MLP, LSTM and GRU.

They follow the shared contract with the known future inputs in the
window: `x` is `(batch, past + horizon, n_nodes)`, with the unknown
horizon cells zeroed by `data.with_future_exog`. Every model predicts
the change from the last observed value of each output, so that it
starts from persistence instead of having to learn the output's level.
"""

import torch
from torch import nn

__all__ = ["GRUForecaster", "LSTMForecaster", "MLPForecaster"]


class _Baseline(nn.Module):
    """Geometry and the last-observed-value skip shared by the baselines."""

    def __init__(
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
    ) -> None:
        super().__init__()
        if input_window <= output_window:
            raise ValueError("input_window must include past and horizon")
        self.n_nodes = n_nodes
        self.input_window = input_window
        self.output_window = output_window
        self.n_targets = len(target_idx)
        self.last_observed = input_window - output_window - 1
        self.register_buffer(
            "target_idx", torch.as_tensor(target_idx), persistent=False
        )

    def _from_last(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Add each output's last observed value to the predicted change."""
        last = x[:, self.last_observed, self.target_idx]
        return last.unsqueeze(1) + delta


class MLPForecaster(_Baseline):
    """Multilayer perceptron over the flattened window.

    Args:
        n_nodes: Series in the window.
        input_window: Past plus horizon rows.
        output_window: Forecast horizon.
        target_idx: Output nodes, in the returned order.
        hidden_size: Width of the hidden layers.
        n_layers: Hidden layers.
        dropout: Dropout after each hidden layer.
    """

    def __init__(  # noqa: PLR0913 - architecture knobs
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
        *,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(n_nodes, input_window, output_window, target_idx)
        layers: list[nn.Module] = []
        size = input_window * n_nodes
        for _ in range(n_layers):
            layers += [
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            size = hidden_size
        layers.append(nn.Linear(size, output_window * self.n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast `(batch, output_window, n_targets)`."""
        delta = self.net(x.flatten(1)).reshape(
            -1, self.output_window, self.n_targets
        )
        return self._from_last(x, delta)


class _RecurrentForecaster(_Baseline):
    """Recurrent net read at the horizon rows, one output per step."""

    cell: type[nn.LSTM] | type[nn.GRU]

    def __init__(  # noqa: PLR0913 - architecture knobs
        self,
        n_nodes: int,
        input_window: int,
        output_window: int,
        target_idx: list[int],
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(n_nodes, input_window, output_window, target_idx)
        self.rnn = self.cell(
            n_nodes,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, self.n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast `(batch, output_window, n_targets)`."""
        states, _ = self.rnn(x)
        delta = self.head(states[:, -self.output_window :])
        return self._from_last(x, delta)


class LSTMForecaster(_RecurrentForecaster):
    """Long Short-Term Memory baseline."""

    cell = nn.LSTM


class GRUForecaster(_RecurrentForecaster):
    """Gated Recurrent Unit baseline."""

    cell = nn.GRU
