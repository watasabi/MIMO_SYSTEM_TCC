"""Message-passing GNN with a message bottleneck, built for symbolic
distillation (Cranmer et al., 2020; SymTorch, 2026).

This is the extraction vehicle ("Modelo B") of
`docs/3_investigacao-arquiteturas.md`. The whole point of the design is
that `message_fn` and `update_fn` are isolated submodules whose inputs
are **named physical variables** (the lag window of each node), so a
symbolic regressor can be run on each one separately.

Keep `input_window` small (3, matching the ARMAX order of the reference
paper) when the goal is extraction: `message_fn` sees `2 * input_window`
scalars, and symbolic regression degrades quickly past a handful of
inputs.
"""

import torch
from torch import nn

__all__ = ["MPNNForecaster"]


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP with ReLU, the unit a symbolic regressor replaces."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class MPNNForecaster(nn.Module):
    """Fully-connected MPNN over the system variables.

    Each of the `n_nodes` series is one node; its feature vector is its
    own lag window, so every argument of `message_fn` has a name
    (`u1(t-1)`, `y2(t-3)`, ...). Messages are summed over senders and
    the update function maps (own window, aggregated message) to the
    forecast horizon.

    The graph is fully connected and the adjacency is **not** learned.
    Coupling topology is read off `edge_strength()` instead: with an L1
    penalty on the messages (see `message_l1`), edges that carry no
    physics collapse towards zero, which is the same mechanism that
    makes the message vector itself interpretable.

    Args:
        n_nodes: Number of series (nodes). 7 for the São José reservoir
            (`u1..u4` + `y1..y3`).
        input_window: Lag window length used as node features.
        output_window: Forecast horizon.
        target_idx: Indices of the nodes that are model outputs
            (`y1..y3`), in the order they should be returned.
        hidden_size: Width of the message/update MLPs.
        message_dim: Message bottleneck width. Cranmer et al. recover
            force laws with 2-3; SymTorch's pruning regularisation finds
            it automatically, but it is explicit here so the notebook
            can sweep it.
        n_layers: Rounds of message passing. 1 keeps `update_fn`
            distillable in one shot; >1 stacks separate MLP pairs and
            needs one symbolic regression per round.

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
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_nodes = n_nodes
        self.input_window = input_window
        self.output_window = output_window
        self.message_dim = message_dim
        self.n_layers = n_layers
        self.register_buffer(
            "target_idx", torch.tensor(target_idx, dtype=torch.long)
        )

        node_dim = input_window
        self.message_fn = nn.ModuleList()
        self.update_fn = nn.ModuleList()
        for layer in range(n_layers):
            self.message_fn.append(
                _mlp(2 * node_dim, hidden_size, message_dim)
            )
            is_last = layer == n_layers - 1
            out_dim = output_window if is_last else node_dim
            self.update_fn.append(
                _mlp(node_dim + message_dim, hidden_size, out_dim)
            )

        # Filled by the last forward pass: [batch, recv, send, msg].
        self.last_messages: torch.Tensor | None = None

    @staticmethod
    def _pairwise_inputs(h: torch.Tensor) -> torch.Tensor:
        """Build the `(h_i, h_j)` pairs fed to `message_fn`.

        Index `[b, i, j]` is receiver `i`, sender `j`, concatenated as
        `(h_i, h_j)` — the same argument order as Cranmer's `phi^e`.
        """
        n = h.shape[1]
        recv = h.unsqueeze(2).expand(-1, -1, n, -1)
        send = h.unsqueeze(1).expand(-1, n, -1, -1)
        return torch.cat([recv, send], dim=-1)

    def _pass(
        self, h: torch.Tensor, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One round of message passing.

        Returns:
            `(messages, aggregated, h_next)` with shapes
            `(batch, recv, send, message_dim)`,
            `(batch, nodes, message_dim)` and `(batch, nodes, out_dim)`.
        """
        n = h.shape[1]
        pairs = self._pairwise_inputs(h)
        messages = self.message_fn[layer](pairs)
        # No self-messages: a node already sees its own window.
        eye = torch.eye(n, device=h.device, dtype=torch.bool)
        messages = messages.masked_fill(eye.view(1, n, n, 1), 0.0)
        aggregated = messages.sum(dim=2)
        h_next = self.update_fn[layer](torch.cat([h, aggregated], dim=-1))
        return messages, aggregated, h_next

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run message passing and return the forecast."""
        # [batch, window, nodes] -> [batch, nodes, window]
        h = x.transpose(1, 2)
        messages = None
        for layer in range(self.n_layers):
            messages, _, h = self._pass(h, layer)

        self.last_messages = messages
        out = h.index_select(1, self.target_idx)
        return out.transpose(1, 2)

    def _features_at(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Node features entering `layer`, `(batch, nodes, dim)`."""
        h = x.transpose(1, 2)
        for previous in range(layer):
            _, _, h = self._pass(h, previous)
        return h

    def message_l1(self) -> torch.Tensor:
        """Mean absolute message of the last forward pass.

        Add this to the training loss (Cranmer et al. use weight 1e-2).
        It is what forces `message_fn` to become a rotation of the true
        interaction instead of an arbitrary high-dimensional encoding.
        """
        if self.last_messages is None:
            raise RuntimeError("call forward() before message_l1()")
        return self.last_messages.abs().mean()

    @torch.no_grad()
    def edge_strength(self, x: torch.Tensor) -> torch.Tensor:
        """Mean message norm per edge, as a coupling-topology matrix.

        Args:
            x: Batch of input windows, `(batch, input_window, n_nodes)`.

        Returns:
            `(n_nodes, n_nodes)` tensor; entry `[i, j]` is how much node
            `j` pushes into node `i`. Diagonal is zero by construction.
        """
        self.forward(x)
        assert self.last_messages is not None
        return self.last_messages.norm(dim=-1).mean(dim=0)

    @torch.no_grad()
    def collect_message_io(
        self, x: torch.Tensor, layer: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Input/output pairs of `message_fn`, for symbolic regression.

        Self-edges are dropped. Columns of the returned inputs are, in
        order, the receiver's lag window then the sender's — name them
        accordingly when handing them to PySR/SymTorch.

        Args:
            x: Batch of input windows, `(batch, input_window, n_nodes)`.
            layer: Which message-passing round to sample.

        Returns:
            `(inputs, messages)` with shapes
            `(n_edges, 2 * input_window)` and `(n_edges, message_dim)`.
        """
        h = self._features_at(x, layer)
        n = h.shape[1]
        pairs = self._pairwise_inputs(h)
        messages = self.message_fn[layer](pairs)
        off_diag = ~torch.eye(n, device=h.device, dtype=torch.bool)
        mask = off_diag.view(1, n, n).expand(h.shape[0], -1, -1)
        return pairs[mask], messages[mask]

    @torch.no_grad()
    def collect_update_io(
        self, x: torch.Tensor, layer: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Input/output pairs of `update_fn`, for symbolic regression.

        Run this after the message function has been distilled: the
        aggregated message is already a low-dimensional scalar-ish
        quantity, so this regression is the cheap one.

        Args:
            x: Batch of input windows, `(batch, input_window, n_nodes)`.
            layer: Which round to sample. Defaults to the last, whose
                output is the forecast itself.

        Returns:
            `(inputs, outputs)` with shapes
            `(batch * n_nodes, input_window + message_dim)` and
            `(batch * n_nodes, out_dim)`. Input columns are the node's
            own lag window followed by the aggregated message.
        """
        if layer is None:
            layer = self.n_layers - 1
        h = self._features_at(x, layer)
        _, aggregated, h_next = self._pass(h, layer)
        inputs = torch.cat([h, aggregated], dim=-1)
        return (
            inputs.reshape(-1, inputs.shape[-1]),
            h_next.reshape(-1, h_next.shape[-1]),
        )
